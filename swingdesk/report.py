"""Reporting: terminal summary, CSV/JSON files and a self-contained HTML briefing."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

DISPLAY_COLS = ["setup", "conviction", "close", "entry", "stop", "stop_pct", "t1", "t2", "units", "pos_aed",
                "risk_aed", "atr_pct", "earnings_flag", "selected", "skip_reason"]


# ───────────────────────────────────────── helpers ───────────────────────────────────────────────
def _fmt(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "" if x is None else str(x)
    if not np.isfinite(x):
        return "–"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:.{min(nd, 1)}f}"
    if abs(x) < 0.01 and x != 0:
        return f"{x:.4g}"
    return f"{x:.{nd}f}"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    if isinstance(o, (np.ndarray, pd.Series)):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.reset_index().to_dict(orient="records")
    return str(o)


def regime_sentence(snap: dict) -> str:
    parts = [f"{snap['bench']} {snap['bench_close']:,}"]
    if snap.get("above_200d") is not None:
        parts.append("above its 200-day" if snap["above_200d"] else "below its 200-day")
    if snap.get("breadth_pct_above_50d") is not None:
        parts.append(f"{snap['breadth_pct_above_50d']:.0f}% of names above the 50-day")
    if snap.get("vix") is not None:
        parts.append(f"VIX {snap['vix']}")
    if snap.get("credit_risk_appetite"):
        parts.append(f"credit {snap['credit_risk_appetite']}")
    if snap.get("realized_vol_pct") is not None:
        parts.append(f"realised vol at percentile {snap['realized_vol_pct']:.0f}")
    return ", ".join(parts)


# ──────────────────────────────────────── terminal ───────────────────────────────────────────────
def print_scan(res: dict, cfg: dict, top: int = 10) -> None:
    snap = res["regime"]
    cands = res["candidates"]
    port = res["portfolio"]
    line = "─" * 100
    print(f"\n{line}\n {res['label']}   |   regime {snap['regime']} (score {snap['score']:+.2f}, {snap['regime_age_bars']} bars)   |   data: {res['source']}")
    print(f" {regime_sentence(snap)}")
    for n in res.get("notes", []):
        print(f" • {n}")
    rk = cfg["risk"]
    print(f" Risk per trade {rk['risk_per_trade_pct'].get(snap['regime'], 0)}% of {cfg['capital_aed']:,} AED "
          f"→ max {rk['max_positions'].get(snap['regime'], 0)} positions, heat cap {port['heat_cap_aed']:,.0f} AED")
    print(line)
    if cands.empty:
        print(" No candidates (universe empty or everything filtered). Run `python run.py validate` to check data coverage.")
        return
    show = cands.head(top).copy()
    show.insert(0, "ticker", show.index)
    show["conviction"] = show["conviction"].map(lambda v: f"{v:.0f}%")
    for c in ["close", "entry", "stop", "t1", "t2"]:
        show[c] = show[c].map(_fmt)
    show["stop_pct"] = show["stop_pct"].map(lambda v: f"{v:.1f}%")
    show["atr_pct"] = show["atr_pct"].map(lambda v: f"{v:.1f}%")
    show["units"] = show["units"].map(lambda v: _fmt(v, 4) if v < 10 else _fmt(v, 0))
    show["pos_aed"] = show["pos_aed"].map(lambda v: _fmt(v, 0))
    show["risk_aed"] = show["risk_aed"].map(lambda v: _fmt(v, 0))
    show["pick"] = np.where(show["selected"], "✔", "")
    cols = ["ticker", "setup", "conviction", "close", "entry", "stop", "stop_pct", "t1", "t2", "units", "pos_aed", "risk_aed", "atr_pct", "pick"]
    for c in ["ticker", "setup"]:
        w = max(show[c].astype(str).str.len().max(), len(c))
        show[c] = show[c].astype(str).str.ljust(w)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.colheader_justify", "left"):
        print(show[cols].to_string(index=False))
    print()
    for t, r in show.iterrows():
        extra = " | ".join(x for x in [r.get("earnings_flag", ""), "; ".join(r.get("plan_flags", []) or []), r.get("skip_reason", "")] if x)
        print(f"  {t:<14} {r['why']}" + (f"   [{extra}]" if extra else ""))
    print()
    if port["selected"]:
        print(f" Portfolio: {', '.join(port['selected'])}  — open risk {port['heat_aed']:,.0f} / {port['heat_cap_aed']:,.0f} AED, "
              f"capital deployed {port['exposure_aed']:,.0f} / {port['exposure_cap_aed']:,.0f} AED")
    else:
        print(" Portfolio: nothing selected under current regime/constraints.")
    diag = res.get("diagnostics", {})
    if diag.get("ic"):
        ic = diag["ic"]
        print(f" Signal check (this universe, {diag.get('years', '?')}y): rank IC {ic['mean_ic']:+.3f} (t≈{ic['ic_t_stat']:.1f}, "
              f"{ic['pct_positive']*100:.0f}% of weeks positive)")
    if diag.get("deciles") is not None:
        d = diag["deciles"]
        print(f"   top-decile 15-bar forward return {d.loc[10, 'mean']:+.2f}% (hit {d.loc[10, 'hit_rate']*100:.0f}%) vs bottom decile {d.loc[1, 'mean']:+.2f}%")
    if res.get("ml_metrics"):
        m = res["ml_metrics"]
        if "auc" in m:
            print(f" ML walk-forward: AUC {m['auc']:.3f}, top-decile precision {m['top_decile_precision']*100:.0f}% vs base {m['base_rate']*100:.0f}%, "
                  f"top-decile fwd ret {m['top_decile_fwd_ret_pct']:+.2f}% vs all {m['all_fwd_ret_pct']:+.2f}% ({m['n_refits']} refits)")
    if res.get("backtest"):
        print_backtest(res["backtest"], res["label"])


def print_backtest(bt: dict, label: str = "") -> None:
    m = bt["metrics"]
    print(f"\n Backtest {label}: {m.get('start')} → {m.get('end')}")
    print(f"   CAGR {m.get('cagr_pct', np.nan):+.1f}%  max drawdown {m.get('max_dd_pct', np.nan):.1f}%  Sharpe {m.get('sharpe', np.nan):.2f}"
          + (f"   | benchmark CAGR {m['bench_cagr_pct']:+.1f}%, max DD {m['bench_max_dd_pct']:.1f}%" if "bench_cagr_pct" in m else ""))
    if m.get("n_trades"):
        print(f"   {m['n_trades']} trades ({m.get('trades_per_year', np.nan):.0f}/yr), win rate {m['win_rate']*100:.0f}%, avg {m['avg_r']:+.2f}R "
              f"(wins {m['avg_win_r']:+.2f}R / losses {m['avg_loss_r']:+.2f}R), profit factor {m['profit_factor']:.2f}, avg hold {m['avg_bars']:.1f} bars")
        print(f"   exits: {m['exit_reasons']}")


# ────────────────────────────────────────── files ────────────────────────────────────────────────
def save_outputs(results: list[dict], cfg: dict, outdir: str, stamp: str) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    summary = {"generated": stamp, "capital_aed": cfg["capital_aed"], "markets": []}
    for res in results:
        c = res["candidates"].copy()
        if not c.empty:
            c["plan_flags"] = c["plan_flags"].map(lambda v: "; ".join(v) if isinstance(v, list) else v)
            p = out / f"signals_{res['market']}_{stamp}.csv"
            c.to_csv(p, index_label="ticker")
            paths[f"csv_{res['market']}"] = str(p)
        entry = {k: res[k] for k in ["market", "label", "source", "regime", "notes", "portfolio"]}
        entry["candidates"] = c.head(25).reset_index().to_dict(orient="records") if not c.empty else []
        entry["diagnostics"] = {k: v for k, v in res.get("diagnostics", {}).items()}
        entry["ml_metrics"] = res.get("ml_metrics")
        if res.get("backtest"):
            entry["backtest"] = res["backtest"]["metrics"]
            bt_path = out / f"trades_{res['market']}_{stamp}.csv"
            res["backtest"]["trades"].to_csv(bt_path, index=False)
            paths[f"trades_{res['market']}"] = str(bt_path)
        summary["markets"].append(entry)
    jp = out / f"summary_{stamp}.json"
    jp.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    paths["json"] = str(jp)
    hp = out / f"briefing_{stamp}.html"
    hp.write_text(render_html(results, cfg, stamp), encoding="utf-8")
    paths["html"] = str(hp)
    return paths


# ─────────────────────────────────────────── HTML ────────────────────────────────────────────────
_CSS = """
:root{--bg:#f3f5f8;--paper:#ffffff;--ink:#15202b;--muted:#5b6b7c;--rule:#d9dfe6;--pos:#1f7a5c;--neg:#b3362f;--warn:#a8661a;--neutral:#4a5a6a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Avenir Next","Segoe UI","Helvetica Neue",Arial,sans-serif;font-variant-numeric:tabular-nums}
main{max-width:1180px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:30px;font-weight:600;letter-spacing:-.01em;margin:0 0 4px}
h2{font-size:21px;font-weight:600;margin:0}
.sub{color:var(--muted);margin:0 0 26px}
section{background:var(--paper);border:1px solid var(--rule);border-radius:6px;margin:0 0 22px;overflow:hidden}
.strip{display:flex;gap:18px;align-items:flex-start;padding:16px 20px;border-left:8px solid var(--neutral)}
.strip.RISK_ON{border-left-color:var(--pos)} .strip.RISK_OFF{border-left-color:var(--warn)} .strip.CRISIS{border-left-color:var(--neg)}
.strip .reg{font-size:13px;color:var(--muted)}
.strip p{margin:4px 0 0;color:var(--muted);font-size:14px}
.notes{padding:0 20px 12px;color:var(--muted);font-size:14px}
.notes div::before{content:"— ";color:var(--rule)}
.tablewrap{overflow-x:auto;border-top:1px solid var(--rule)}
table{border-collapse:collapse;width:100%;min-width:980px;font-size:13.5px}
th{font-weight:600;text-align:left;color:var(--muted);padding:9px 10px;border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:9px 10px;border-bottom:1px solid var(--rule);vertical-align:top;white-space:nowrap}
tr.pick td:first-child{box-shadow:inset 4px 0 0 var(--pos)}
td.num,th.num{text-align:right}
.tick{font-weight:600;font-size:15px}
.setup{display:inline-block;padding:2px 8px;border-radius:12px;border:1px solid var(--rule);font-size:12.5px;color:var(--ink);background:#f7f9fb}
.conv{display:flex;align-items:center;gap:8px}
.bar{height:8px;width:90px;background:#e6ebf0;border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--pos)}
.why{white-space:normal;max-width:300px;color:var(--muted);font-size:12.5px;line-height:1.35}
.flag{color:var(--warn)} .neg{color:var(--neg)} .pos{color:var(--pos)}
.foot{padding:14px 20px;color:var(--muted);font-size:13.5px;border-top:1px solid var(--rule)}
.foot b{color:var(--ink);font-weight:600}
.method{color:var(--muted);font-size:13.5px;line-height:1.55}
.method p{margin:0 0 8px}
@media(max-width:640px){main{padding:18px 12px}h1{font-size:24px}.strip{flex-direction:column;gap:8px}}
"""


def _spark(vals, w=96, h=28):
    v = [float(x) for x in vals if x is not None and np.isfinite(x)]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    rng = (hi - lo) or 1.0
    pts = " ".join(f"{i * w / (len(v) - 1):.1f},{h - 1 - (x - lo) / rng * (h - 2):.1f}" for i, x in enumerate(v))
    color = "var(--pos)" if v[-1] >= v[0] else "var(--neg)"
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-hidden="true"><polyline fill="none" stroke="{color}" stroke-width="1.5" points="{pts}"/></svg>'


def _esc(s) -> str:
    return html.escape("" if s is None else str(s))


def render_html(results: list[dict], cfg: dict, stamp: str) -> str:
    rk = cfg["risk"]
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>SwingDesk briefing {stamp}</title><style>{_CSS}</style></head><body><main>",
             f"<h1>Swing briefing</h1><p class='sub'>{_esc(stamp)} — capital {cfg['capital_aed']:,} AED, horizon {cfg['holding_days']} trading days, spot only, long only.</p>"]
    for res in results:
        snap, cands, port = res["regime"], res["candidates"], res["portfolio"]
        parts.append(f"<section><div class='strip {snap['regime']}'><div><h2>{_esc(res['label'])}</h2>"
                     f"<div class='reg'>{_esc(res['source'])}</div></div>"
                     f"<div><div><strong>{snap['regime'].replace('_', ' ').title()}</strong> (score {snap['score']:+.2f}, {snap['regime_age_bars']} bars) — "
                     f"risk {rk['risk_per_trade_pct'].get(snap['regime'], 0)}% per trade, up to {rk['max_positions'].get(snap['regime'], 0)} positions</div>"
                     f"<p>{_esc(regime_sentence(snap))}</p></div></div>")
        if res.get("notes"):
            parts.append("<div class='notes'>" + "".join(f"<div>{_esc(n)}</div>" for n in res["notes"]) + "</div>")
        if cands.empty:
            parts.append("<div class='foot'>No candidates. Check data coverage with <b>python run.py validate</b>.</div></section>")
            continue
        sparks = res.get("sparks", {})
        rows = []
        for t, r in cands.head(15).iterrows():
            flags = [x for x in [r.get("earnings_flag", ""), *(r.get("plan_flags") or [])] if x]
            skip = r.get("skip_reason", "")
            flag_html = "".join(f"<div class='flag'>{_esc(f)}</div>" for f in flags)
            if skip and not r["selected"]:
                flag_html += f"<div class='neg'>skip: {_esc(skip)}</div>"
            conv = float(r["conviction"])
            rows.append(
                f"<tr class='{'pick' if r['selected'] else ''}'><td><div class='tick'>{_esc(t)}</div>{_spark(sparks.get(t, []))}</td>"
                f"<td><span class='setup'>{_esc(r['setup'])}</span></td>"
                f"<td><div class='conv'><div class='bar'><i style='width:{conv:.0f}%'></i></div>{conv:.0f}%</div></td>"
                f"<td class='num'>{_fmt(r['close'])}</td><td class='num'>{_fmt(r['entry'])}</td>"
                f"<td class='num'>{_fmt(r['stop'])}<br><span class='neg'>−{r['stop_pct']:.1f}%</span></td>"
                f"<td class='num'>{_fmt(r['t1'])}<br>{_fmt(r['t2'])}</td>"
                f"<td class='num'>{_fmt(r['units'], 4 if r['units'] < 10 else 0)}<br>{_fmt(r['pos_aed'], 0)} AED</td>"
                f"<td class='num'>{_fmt(r['risk_aed'], 0)}</td><td class='num'>{r['atr_pct']:.1f}%</td>"
                f"<td class='why'>{_esc(r['why'])}{flag_html}</td></tr>")
        parts.append("<div class='tablewrap'><table><thead><tr><th>Name</th><th>Setup</th><th>Conviction</th><th class='num'>Last</th>"
                     "<th class='num'>Entry</th><th class='num'>Stop</th><th class='num'>T1 / T2</th><th class='num'>Size</th>"
                     "<th class='num'>Risk AED</th><th class='num'>ATR</th><th>Why / flags</th></tr></thead><tbody>"
                     + "".join(rows) + "</tbody></table></div>")
        foot = (f"Portfolio: <b>{', '.join(port['selected']) if port['selected'] else 'nothing selected'}</b> — open risk "
                f"{port['heat_aed']:,.0f} / {port['heat_cap_aed']:,.0f} AED, deployed {port['exposure_aed']:,.0f} / {port['exposure_cap_aed']:,.0f} AED.")
        diag = res.get("diagnostics", {})
        if diag.get("ic"):
            ic = diag["ic"]
            foot += f" Signal check: rank IC {ic['mean_ic']:+.3f} (t≈{ic['ic_t_stat']:.1f})"
            if diag.get("deciles") is not None:
                d = diag["deciles"]
                foot += f"; top decile {d.loc[10, 'mean']:+.2f}% vs bottom {d.loc[1, 'mean']:+.2f}% over {cfg['holding_days']} bars."
        if res.get("ml_metrics") and "auc" in res["ml_metrics"]:
            m = res["ml_metrics"]
            foot += f" ML walk-forward AUC {m['auc']:.3f}, top-decile precision {m['top_decile_precision']*100:.0f}% vs base {m['base_rate']*100:.0f}%."
        if res.get("backtest"):
            m = res["backtest"]["metrics"]
            foot += (f" Backtest {m.get('start')}→{m.get('end')}: CAGR {m.get('cagr_pct', float('nan')):+.1f}%, max DD {m.get('max_dd_pct', float('nan')):.1f}%, "
                     f"{m.get('n_trades', 0)} trades, win rate {m.get('win_rate', float('nan'))*100:.0f}%, avg {m.get('avg_r', float('nan')):+.2f}R.")
        parts.append(f"<div class='foot'>{foot}</div></section>")
    parts.append("<section><div class='foot method'>"
                 "<p>How to read this: each name is ranked against its own market on trend, momentum, entry timing, volatility compression, "
                 "volume accumulation and (for equities) fundamentals and news, with weights that shift by regime. Conviction is a ranking, not a probability of profit. "
                 "Stops sit under structure or 2 ATR; sizing risks a fixed fraction of capital per trade; the heat cap limits total open risk.</p>"
                 "<p>Limits: free, delayed data; survivorship bias in a hand-picked universe; backtests ignore borrow, halts and gap-throughs beyond the modelled slippage. "
                 "Nothing here is investment advice — it is a research tool for your own decisions.</p></div></section></main></body></html>")
    return "".join(parts)
