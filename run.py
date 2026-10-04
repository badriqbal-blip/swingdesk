#!/usr/bin/env python3
"""SwingDesk — multi-factor swing-trade research & risk engine.

  python run.py scan                      # scan all markets in config.yaml, write CSV/JSON/HTML to ./output
  python run.py scan --markets us,crypto --top 12 --ml --with-backtest
  python run.py scan --demo               # offline smoke run on synthetic data
  python run.py backtest --markets us     # full event-driven backtest + signal diagnostics
  python run.py validate                  # which tickers actually return data
  python run.py regime                    # regime read-out only
  python run.py size --market us --entry 190 --stop 182   # ad-hoc position sizing
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

import numpy as np
import pandas as pd

from swingdesk import data as D
from swingdesk.backtest import decile_analysis, ic_analysis, run_backtest
from swingdesk.config import load_config
from swingdesk.events import earnings_assessment, macro_warnings
from swingdesk.factors import (FUNDAMENTAL_FACTORS, SENTIMENT_FACTORS, breadth_series, build_panel,
                               fundamental_factors, latest_cross_section)
from swingdesk.regime import compute_regime, regime_snapshot
from swingdesk.report import print_backtest, print_scan, save_outputs
from swingdesk.risk import plan_trade, returns_matrix, select_portfolio
from swingdesk.scoring import (SETUP_INFO, classify_setup, composite, conviction_from_composite, explain,
                               group_scores, score_panel, zscore_flat)
from swingdesk.sentiment import fear_greed_adjustment, score_headlines
from swingdesk.universe import MARKET_META, aed_per_unit, get_universe, min_adv


# ───────────────────────────────────────── data loading ──────────────────────────────────────────
def load_market(market: str, cfg: dict, demo: bool):
    tickers = get_universe(cfg, market)
    years = int(cfg["lookback_years"])
    cache, ttl = cfg["cache_dir"], cfg["cache_ttl_hours"]
    aux: dict = {}
    if demo:
        if market == "crypto":
            prices = D.demo_prices(tickers, 900, seed=3, vol=0.045, level=40, daily=True)
        elif market == "uae":
            prices = D.demo_prices(tickers, 800, seed=5, vol=0.016, level=6)
        else:
            prices = D.demo_prices(tickers, 800, seed=1, vol=0.02, level=120)
            aux["bench"] = D.demo_prices(["SPY"], 800, seed=11, vol=0.011, level=500)["SPY"]["Close"]
            vix = D.demo_prices(["VIX"], 800, seed=13, vol=0.06, level=18)["VIX"]["Close"]
            aux["vix"] = (vix / vix.mean() * 18).clip(10, 60)
            cr = D.demo_prices(["HYG", "IEF"], 800, seed=17, vol=0.006, level=80)
            aux["credit"] = cr["HYG"]["Close"] / cr["IEF"]["Close"]
        source = "synthetic demo data (offline)"
    else:
        if market == "crypto":
            prices, src = D.fetch_crypto_ohlcv(tickers, years, cfg["crypto"]["exchange"],
                                               cfg["crypto"].get("fallback_exchanges", []), cache, ttl)
            source = f"{src} daily spot OHLCV"
        else:
            extra = []
            if market == "us":
                extra = [cfg["benchmarks"]["us"], cfg["aux"]["vix"], cfg["aux"]["credit_risk_on"], cfg["aux"]["credit_risk_off"]]
            allp = D.fetch_equity_ohlcv(tickers + [e for e in extra if e not in tickers], years, cache, ttl)
            prices = {t: allp[t] for t in tickers if t in allp}
            if market == "us":
                b = allp.get(cfg["benchmarks"]["us"])
                aux["bench"] = b["Close"] if b is not None else None
                v = allp.get(cfg["aux"]["vix"])
                aux["vix"] = v["Close"] if v is not None else None
                h, i = allp.get(cfg["aux"]["credit_risk_on"]), allp.get(cfg["aux"]["credit_risk_off"])
                aux["credit"] = (h["Close"] / i["Close"]).dropna() if h is not None and i is not None else None
            source = "Yahoo Finance (yfinance), daily adjusted"
    bench_name = cfg["benchmarks"].get(market) or "equal-weight universe"
    if market == "crypto" and cfg["benchmarks"].get("crypto") in prices:
        bench = prices[cfg["benchmarks"]["crypto"]]["Close"]
    else:
        bench = aux.get("bench")
    if bench is None or bench.dropna().empty:
        bench, bench_name = D.equal_weight_index(prices), "equal-weight universe"
    missing = [t for t in tickers if t not in prices]
    return prices, bench, bench_name, aux, source, missing


# ───────────────────────────────────────────── scan ──────────────────────────────────────────────
def scan_market(market: str, cfg: dict, args) -> dict | None:
    meta = MARKET_META[market]
    t0 = time.time()
    print(f"\n[{market}] loading data…")
    prices, bench, bench_name, aux, source, missing = load_market(market, cfg, args.demo)
    if len(prices) < 5:
        print(f"[{market}] only {len(prices)} instruments with data — skipping. Missing: {missing[:10]}")
        return None
    if missing:
        print(f"[{market}] no data for {len(missing)} tickers: {', '.join(missing[:12])}{'…' if len(missing) > 12 else ''}")
    print(f"[{market}] {len(prices)} instruments, computing features…")

    H = int(cfg["holding_days"])
    panel = build_panel(prices, bench, meta["atr_ideal_pct"], H, meta["periods_per_year"])
    breadth = breadth_series(panel)
    regime_df = compute_regime(bench, breadth, aux.get("vix"), aux.get("credit"), meta["periods_per_year"])
    scored = score_panel(panel, regime_df["regime"], cfg["weights"])
    snap = regime_snapshot(regime_df, bench_name)
    regime_now = snap["regime"]

    latest = latest_cross_section(scored)
    latest = latest[latest["composite"].notna()].copy()
    today = pd.Timestamp.today().normalize()
    notes: list[str] = []

    # ── fundamentals (equities) ───────────────────────────────────────────────────────────────
    fund = None
    if market != "crypto" and not args.no_fundamentals and not args.demo:
        top_f = latest["composite"].sort_values(ascending=False).head(int(cfg["scan"]["fundamentals_top"])).index.tolist()
        print(f"[{market}] fundamentals for top {len(top_f)}…")
        fund = D.fetch_fundamentals(top_f, cfg["cache_dir"], cfg["fundamentals_ttl_hours"])
        ff = fundamental_factors(fund, latest["close"])
        zf = zscore_flat(ff, list(FUNDAMENTAL_FACTORS))
        gsf = group_scores(zf, {"fundamentals": FUNDAMENTAL_FACTORS})
        latest["g_fundamentals"] = gsf["g_fundamentals"].reindex(latest.index)
        for col in ["sector", "shortName", "shortPercentOfFloat", "recommendationKey"]:
            if col in fund.columns:
                latest[col] = fund[col].reindex(latest.index)
        if "quoteType" in fund.columns:
            is_etf = fund["quoteType"].reindex(latest.index).astype(str).str.upper().eq("ETF")
            latest.loc[is_etf.fillna(False), "sector"] = "ETF"
    elif args.demo and market != "crypto":
        rng = np.random.default_rng(42)
        latest["g_fundamentals"] = rng.normal(0, 0.8, len(latest))
        latest["sector"] = rng.choice(["Technology", "Financials", "Energy", "Healthcare", "Industrials"], len(latest))

    # ── news / crowding sentiment ─────────────────────────────────────────────────────────────
    sent_rows: dict[str, dict] = {}
    news_items: dict[str, list] = {}
    if not args.no_news:
        top_n = latest["composite"].sort_values(ascending=False).head(int(cfg["scan"]["news_top"])).index.tolist()
        if market == "crypto" and not args.demo:
            fr = D.fetch_funding_rates(top_n)
            for t, v in fr.items():
                sent_rows.setdefault(t, {})["funding_contrarian"] = -v
            if fr:
                notes.append(f"Perp funding read for {len(fr)} names (positive funding = crowded longs → contrarian penalty)")
        if market != "crypto":
            print(f"[{market}] news for top {len(top_n)}…")
            for t in top_n:
                items = D.demo_news(t) if args.demo else D.fetch_news(t)
                s = score_headlines(items)
                if s["n"]:
                    sent_rows.setdefault(t, {}).update({"news_sent": s["news_sent"], "news_attention": s["news_attention"]})
                    news_items[t] = s["items"]
    if sent_rows:
        sf = pd.DataFrame.from_dict(sent_rows, orient="index")
        zs = zscore_flat(sf, [c for c in SENTIMENT_FACTORS if c in sf.columns])
        gss = group_scores(zs, {"sentiment": {c: w for c, w in SENTIMENT_FACTORS.items() if c in zs.columns}})
        latest["g_sentiment"] = gss["g_sentiment"].reindex(latest.index)
        if "news_sent" in sf.columns:
            latest["news_sent"] = sf["news_sent"].reindex(latest.index)

    # ── market-level sentiment gauge ──────────────────────────────────────────────────────────
    fg = None
    if not args.demo:
        fg = D.fetch_crypto_fear_greed() if market == "crypto" else (D.fetch_cnn_fear_greed() if market == "us" else None)
    fg_mult, fg_note = fear_greed_adjustment(fg, regime_now)
    if fg_note:
        notes.append(fg_note)
    if market == "us":
        notes.extend(macro_warnings(cfg["events"].get("macro_dates"), today, int(cfg["events"]["earnings_window_days"])))

    # ── final composite with all available groups, regime weights ─────────────────────────────
    gcols = [c for c in latest.columns if c.startswith("g_")]
    latest["composite_final"] = composite(latest[gcols], regime_now, cfg["weights"])
    latest["conviction"] = conviction_from_composite(latest["composite_final"])

    # ── optional ML blend ─────────────────────────────────────────────────────────────────────
    ml_metrics = None
    if args.ml:
        from swingdesk.ml import walk_forward_ml

        print(f"[{market}] walk-forward ML…")
        mlres = walk_forward_ml(scored, H, int(cfg["ml"]["step_days"]), int(cfg["ml"]["min_train_days"]))
        ml_metrics = mlres["metrics"]
        if len(mlres["latest_prob"]):
            latest["ml_prob"] = mlres["latest_prob"].reindex(latest.index)
            b = float(cfg["ml"]["blend"])
            has = latest["ml_prob"].notna()
            latest.loc[has, "conviction"] = (1 - b) * latest.loc[has, "conviction"] + b * 100 * latest.loc[has, "ml_prob"]

    # ── setups, events, plans ─────────────────────────────────────────────────────────────────
    capital = float(cfg["capital_aed"])
    unit = aed_per_unit(cfg, market)
    adv_floor = min_adv(cfg, market)
    ev_cfg = cfg["events"]
    plans = []
    earn_targets = set(latest["conviction"].sort_values(ascending=False).head(int(cfg["scan"]["news_top"])).index) if market != "crypto" else set()
    for t, r in latest.iterrows():
        setup = classify_setup(r)
        info = SETUP_INFO[setup]
        ea = {"flag": "", "size_mult": 1.0, "veto": False, "earnings_date": None}
        if t in earn_targets and not args.demo and not args.no_news:
            ea = earnings_assessment(D.fetch_next_earnings(t, cfg["cache_dir"]), today, int(ev_cfg["earnings_window_days"]), ev_cfg["earnings_policy"])
        size_mult = fg_mult * ea["size_mult"] * (0.5 if info["kind"] == "momentum" else 1.0)
        plan = plan_trade(r, setup, cfg, regime_now, capital, unit, market, size_mult) or {}
        liquid = float(r.get("adv20", 0) or 0) >= adv_floor
        price_ok = market != "us" or float(r["close"]) >= float(cfg["liquidity"].get("min_price_us", 0))
        conv = float(r["conviction"])
        if not info["eligible"]:
            conv -= 10
        if ea["flag"]:
            conv -= 10
        if not liquid:
            conv -= 8
        plans.append({
            "ticker": t, "setup": setup, "eligible": bool(info["eligible"] and liquid and price_ok and plan),
            "conviction": float(np.clip(conv, 1, 99)), "earnings_flag": ea["flag"], "earnings_date": ea["earnings_date"],
            "veto": bool(ea["veto"]), "liquid": liquid, "why": explain(r), **{k: plan.get(k, np.nan) for k in
            ["entry", "stop", "stop_pct", "t1", "t2", "units", "pos_native", "pos_aed", "risk_aed", "risk_pct", "cost_aed", "hold_days", "entry_note", "exit_rule"]},
            "plan_flags": plan.get("plan_flags", []) + ([] if liquid else ["below liquidity floor"]),
        })
    cands = pd.DataFrame(plans).set_index("ticker")
    keep = ["close", "atr", "atr_pct", "rsi14", "rsi2", "adx", "dist21_atr", "pct_52w_high", "mom_1m", "mom_3m", "mom_6m", "mom_12_1",
            "rvol", "adv20", "squeeze_on", "breakout20", "breakout55", "uptrend", "composite", "comp_rank", "composite_final",
            "g_trend", "g_momentum", "g_timing", "g_volatility", "g_volume", "g_fundamentals", "g_sentiment", "news_sent", "ml_prob",
            "sector", "shortName", "shortPercentOfFloat", "date"]
    cands = cands.join(latest[[c for c in keep if c in latest.columns]])
    cands = cands.sort_values("conviction", ascending=False)

    rets = returns_matrix(prices, cands.index.tolist(), 60)
    cands, port = select_portfolio(cands, rets, cfg, regime_now, capital)

    # ── diagnostics (is the ranking informative at all?) ──────────────────────────────────────
    diag = {"deciles": decile_analysis(scored), "ic": ic_analysis(scored), "years": cfg["lookback_years"]}
    bt = None
    if args.with_backtest:
        print(f"[{market}] backtest…")
        bt = run_backtest(scored, prices, regime_df, cfg, adv_floor, meta["periods_per_year"], bench)

    sparks = {t: prices[t]["Close"].tail(60).tolist() for t in cands.head(15).index if t in prices}
    print(f"[{market}] done in {time.time() - t0:.1f}s")
    return {"market": market, "label": meta["label"], "source": source, "regime": snap, "notes": notes, "candidates": cands,
            "portfolio": port, "diagnostics": diag, "ml_metrics": ml_metrics, "backtest": bt, "sparks": sparks,
            "news": news_items, "fear_greed": fg, "missing": missing}


# ─────────────────────────────────────────── commands ────────────────────────────────────────────
def cmd_scan(cfg, args):
    markets = [m.strip() for m in (args.markets or ",".join(cfg["markets"])).split(",") if m.strip()]
    results = []
    for m in markets:
        if m not in MARKET_META:
            print(f"unknown market '{m}' (choose from {list(MARKET_META)})")
            continue
        res = scan_market(m, cfg, args)
        if res:
            results.append(res)
    for res in results:
        print_scan(res, cfg, args.top)
    if results:
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
        paths = save_outputs(results, cfg, args.outdir, stamp)
        print("\nFiles written:")
        for k, p in paths.items():
            print(f"  {k:<14} {p}")
    print("\nResearch tool, not investment advice. Rankings are relative; stops and sizes are what make it survivable.")


def cmd_backtest(cfg, args):
    markets = [m.strip() for m in (args.markets or ",".join(cfg["markets"])).split(",") if m.strip()]
    for m in markets:
        meta = MARKET_META[m]
        prices, bench, bench_name, aux, source, missing = load_market(m, cfg, args.demo)
        if len(prices) < 5:
            print(f"[{m}] not enough data"); continue
        H = int(cfg["holding_days"])
        panel = build_panel(prices, bench, meta["atr_ideal_pct"], H, meta["periods_per_year"])
        regime_df = compute_regime(bench, breadth_series(panel), aux.get("vix"), aux.get("credit"), meta["periods_per_year"])
        scored = score_panel(panel, regime_df["regime"], cfg["weights"])
        dec, ic = decile_analysis(scored), ic_analysis(scored)
        print(f"\n═══ {meta['label']} — {source}; {len(prices)} instruments; regime now {regime_df['regime'].iloc[-1]}")
        if ic:
            print(f" Rank IC {ic['mean_ic']:+.3f} (t≈{ic['ic_t_stat']:.1f}, positive {ic['pct_positive']*100:.0f}% of dates, n={ic['n_dates']})")
        if dec is not None:
            print(" Forward return by composite decile (%, next 15 bars):")
            print(dec.round(2).to_string())
        bt = run_backtest(scored, prices, regime_df, cfg, min_adv(cfg, m), meta["periods_per_year"], bench)
        if bt:
            print_backtest(bt, meta["label"])
            if args.ml:
                from swingdesk.ml import walk_forward_ml

                mm = walk_forward_ml(scored, H, int(cfg["ml"]["step_days"]), int(cfg["ml"]["min_train_days"]))["metrics"]
                print(f" ML walk-forward: {mm}")
            yr = regime_df["regime"].value_counts(normalize=True).round(2).to_dict()
            print(f" Regime mix over the window: {yr}")
        else:
            print(" Backtest skipped: not enough history.")
    print("\nBacktests here are in-sample on hand-picked current constituents (survivorship bias) and assume fills at the next open ± slippage.")


def cmd_validate(cfg, args):
    for m in [x.strip() for x in (args.markets or ",".join(cfg["markets"])).split(",")]:
        prices, bench, bench_name, aux, source, missing = load_market(m, cfg, args.demo)
        print(f"\n[{m}] {source}: {len(prices)} of {len(get_universe(cfg, m))} tickers returned ≥{D.MIN_BARS} bars")
        if missing:
            print(f"   missing: {', '.join(missing)}")
        if prices:
            last = max(d.index[-1] for d in prices.values())
            print(f"   latest bar: {last.date()}; benchmark: {bench_name}")


def cmd_regime(cfg, args):
    for m in [x.strip() for x in (args.markets or ",".join(cfg["markets"])).split(",")]:
        meta = MARKET_META[m]
        prices, bench, bench_name, aux, source, missing = load_market(m, cfg, args.demo)
        if len(prices) < 5:
            print(f"[{m}] not enough data"); continue
        panel = build_panel(prices, bench, meta["atr_ideal_pct"], int(cfg["holding_days"]), meta["periods_per_year"])
        reg = compute_regime(bench, breadth_series(panel), aux.get("vix"), aux.get("credit"), meta["periods_per_year"])
        snap = regime_snapshot(reg, bench_name)
        print(f"\n[{m}] {snap['regime']} (score {snap['score']:+.2f}) — " + ", ".join(f"{k}={v}" for k, v in snap.items() if k not in ("regime", "score")))
        print("   last 10 bars:", " ".join(reg['regime'].tail(10).map({'RISK_ON': '▲', 'NEUTRAL': '◆', 'RISK_OFF': '▼', 'CRISIS': '✖'}).tolist()))


def cmd_size(cfg, args):
    unit = aed_per_unit(cfg, args.market)
    rk = cfg["risk"]
    regime = args.regime or "NEUTRAL"
    risk_pct = float(rk["risk_per_trade_pct"][regime])
    cap = float(cfg["capital_aed"])
    R = args.entry - args.stop
    if R <= 0:
        print("stop must be below entry for a long"); return
    risk_aed = cap * risk_pct / 100
    units = (risk_aed / unit) / R
    pos_aed = units * args.entry * unit
    cap_aed = cap * float(rk["max_position_pct"]) / 100
    if pos_aed > cap_aed:
        units *= cap_aed / pos_aed
        pos_aed = cap_aed
    if args.market == "uae":
        units = np.floor(units)
        pos_aed = units * args.entry * unit
    print(f"{args.market} long @ {args.entry} stop {args.stop} (−{100*R/args.entry:.2f}%)  regime {regime}: risk {risk_pct}% = {risk_aed:,.0f} AED")
    print(f"  size {units:,.4f} units = {units*args.entry:,.2f} {MARKET_META[args.market]['currency']} ≈ {pos_aed:,.0f} AED "
          f"({100*pos_aed/cap:.1f}% of capital); actual risk {units*R*unit:,.0f} AED")
    print(f"  T1 (+{rk['target1_r']}R) {args.entry + rk['target1_r']*R:.4g}   T2 (+{rk['target2_r']}R) {args.entry + rk['target2_r']*R:.4g}")


def main(argv=None):
    p = argparse.ArgumentParser(description="SwingDesk — swing-trade research & risk engine")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["scan", "backtest", "validate", "regime"]:
        s = sub.add_parser(name)
        s.add_argument("--markets", default=None, help="comma list: us,uae,crypto")
        s.add_argument("--demo", action="store_true", help="synthetic offline data")
        s.add_argument("--ml", action="store_true", help="add the walk-forward ML layer")
        if name == "scan":
            s.add_argument("--top", type=int, default=10)
            s.add_argument("--no-fundamentals", action="store_true")
            s.add_argument("--no-news", action="store_true")
            s.add_argument("--with-backtest", action="store_true")
            s.add_argument("--outdir", default="output")
    s = sub.add_parser("size")
    s.add_argument("--market", default="us", choices=list(MARKET_META))
    s.add_argument("--entry", type=float, required=True)
    s.add_argument("--stop", type=float, required=True)
    s.add_argument("--regime", default=None, choices=["RISK_ON", "NEUTRAL", "RISK_OFF", "CRISIS"])
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    {"scan": cmd_scan, "backtest": cmd_backtest, "validate": cmd_validate, "regime": cmd_regime, "size": cmd_size}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
