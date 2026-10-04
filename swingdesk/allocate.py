"""Portfolio layer: turn per-market rankings into ONE plan for the whole account.

1. Market sleeves — capital is split between US / UAE / crypto from base weights, scaled by each market's regime
   (RISK_ON 100% … CRISIS 0%) and by whether the ranking has shown real information in that market (IC t-stat).
   Whatever isn't earned by a sleeve stays in cash. Cash is a position, not a failure.
2. Within a sleeve — the best eligible, uncorrelated names get inverse-volatility weights tilted by conviction
   (equal-risk-contribution style), then every position is capped by its own risk budget (AED at risk if the stop
   is hit) and by the single-position cap. Leftover budget is redistributed once, then goes to cash.
3. Holdings — positions you already own (holdings.yaml) consume their sleeve first and get a HOLD / ADD / EXIT call.
4. Risk summary — total heat if every stop is hit, one-day 1σ move from the 60-day covariance, diversification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scoring import SETUP_INFO
from .universe import MARKET_META, aed_per_unit

REGIME_MULT = {"RISK_ON": 1.0, "NEUTRAL": 0.7, "RISK_OFF": 0.35, "CRISIS": 0.0}

ORDER_TEXT = {
    "breakout": "Buy at market on the open",
    "squeeze": "Buy-stop {entry} (triggers only if price breaks out)",
    "pullback": "Limit buy {entry} (or market if it reclaims the 21-EMA)",
    "bounce": "Buy at market; 5–7 day hold",
    "momentum": "Buy at market, half size; add after a 2–3 day pause",
}


def _fin(x, default=np.nan):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(x) else x


def _round_units(units: float, market: str, cfg: dict) -> float:
    rk = cfg["risk"]
    if market == "uae" or (market == "us" and not rk.get("fractional_shares_us", True)):
        return float(np.floor(units))
    if market == "crypto":
        return float(np.floor(units * 1e6) / 1e6)
    return float(np.floor(units * 1e4) / 1e4)


def _apply_live(r: pd.Series, live: float | None) -> dict:
    """Shift a market-entry plan to the live price (levels for limit/stop orders stay where they are)."""
    entry, stop, t1, t2 = _fin(r.get("entry")), _fin(r.get("stop")), _fin(r.get("t1")), _fin(r.get("t2"))
    kind = SETUP_INFO.get(r.get("setup"), SETUP_INFO["No clean setup"])["kind"]
    if live is not None and np.isfinite(live) and live > 0 and kind in ("breakout", "momentum", "bounce") and np.isfinite(entry):
        R = entry - stop
        entry, stop, t1, t2 = live, live - R, live + (t1 - r["entry"]), live + (t2 - r["entry"])
    return {"entry": entry, "stop": stop, "t1": t1, "t2": t2, "kind": kind}


# ─────────────────────────────────────────── sleeves ─────────────────────────────────────────────
def market_sleeves(results: list[dict], cfg: dict, macro_flags: dict | None = None) -> dict:
    al, capital = cfg["allocation"], float(cfg["capital_aed"])
    mf = macro_flags or {}
    sleeves = {}
    for res in results:
        m, reg = res["market"], res["regime"]["regime"]
        ic = (res.get("diagnostics") or {}).get("ic") or {}
        t = _fin(ic.get("ic_t_stat"), 0.0)
        ic_mult = 1.0 if t >= float(al["ic_t_full"]) else (float(al["ic_low_mult"]) if t > 0 else float(al["ic_none_mult"]))
        base = float(al["market_base_weights"].get(m, 0.0))
        rm = REGIME_MULT.get(reg, 0.7)
        macro_mult, macro_why = 1.0, ""
        if m == "uae" and mf.get("oil_bull") is False:
            macro_mult, macro_why = 0.85, "oil downtrend"
        if m == "crypto" and mf.get("dollar_strong"):
            macro_mult, macro_why = 0.9, "strong dollar"
        if m == "us" and mf.get("yields_spiking") and mf.get("risk_appetite") is False:
            macro_mult, macro_why = 0.9, "yields spiking with credit risk-off"
        sleeves[m] = {"base_weight": base, "regime": reg, "regime_mult": rm, "ic_t": round(t, 2), "ic_mult": ic_mult,
                      "macro_mult": macro_mult, "macro_why": macro_why,
                      "budget": capital * base * rm * ic_mult * macro_mult, "label": MARKET_META[m]["label"]}
    total = sum(s["budget"] for s in sleeves.values())
    cap_total = capital * float(al["max_deploy_pct"]) / 100.0
    if total > cap_total > 0:
        for s in sleeves.values():
            s["budget"] *= cap_total / total
    return sleeves


# ─────────────────────────────────────────── holdings ────────────────────────────────────────────
def review_holdings(holdings: list[dict] | None, results: list[dict], cfg: dict, live: dict) -> list[dict]:
    rows = []
    by_m = {r["market"]: r for r in results}
    for h in holdings or []:
        m, t = str(h.get("market", "")).lower(), str(h.get("ticker", "")).strip()
        if m not in by_m or not t:
            continue
        c = by_m[m]["candidates"]
        unit = aed_per_unit(cfg, m)
        units, entry = _fin(h.get("units"), 0.0), _fin(h.get("entry"))
        price = live.get(t) if live.get(t) else (_fin(c.loc[t, "close"]) if t in c.index else np.nan)
        if not np.isfinite(price):
            rows.append({"market": m, "ticker": t, "units": units, "entry": entry, "price": np.nan, "value_aed": 0.0,
                         "pnl_pct": np.nan, "action": "REVIEW", "reason": "no data for this ticker today"})
            continue
        value = units * price * unit
        pnl = (price / entry - 1.0) * 100 if np.isfinite(entry) and entry > 0 else np.nan
        stop = _fin(h.get("stop"))
        atr = _fin(c.loc[t, "atr"]) if t in c.index else np.nan
        if not np.isfinite(stop) and np.isfinite(entry) and np.isfinite(atr):
            stop = entry - float(cfg["risk"]["stop_atr_mult"]) * atr
        action, reason = "HOLD", "trend intact"
        if np.isfinite(stop) and price <= stop:
            action, reason = "EXIT", f"stop {stop:.4g} hit"
        elif t in c.index and _fin(c.loc[t, "uptrend"], 1) != 1:
            action, reason = "EXIT", "trend broken (below 200-day or 21<50 EMA)"
        elif by_m[m]["regime"]["regime"] == "CRISIS":
            action, reason = "EXIT", "market in CRISIS regime"
        elif t in c.index and bool(c.loc[t, "eligible"]) and _fin(c.loc[t, "conviction"], 0) >= 60:
            action, reason = "HOLD / ADD", "still ranks as a top setup"
        # trail suggestion
        trail = None
        if t in c.index and np.isfinite(atr) and np.isfinite(entry):
            R = entry - stop if np.isfinite(stop) else float(cfg["risk"]["stop_atr_mult"]) * atr
            gain_r = (price - entry) / R if R > 0 else 0
            if gain_r >= float(cfg["risk"]["trail_after_r"]):
                trail = max(stop, price - float(cfg["risk"]["trail_atr_mult"]) * atr)
            elif gain_r >= float(cfg["risk"]["breakeven_at_r"]):
                trail = max(stop, entry)
        rows.append({"market": m, "ticker": t, "units": units, "entry": entry, "price": price, "value_aed": value,
                     "pnl_pct": pnl, "stop": stop, "suggested_stop": trail if trail else stop, "action": action, "reason": reason})
    return rows


# ───────────────────────────────────────── allocation ────────────────────────────────────────────
def allocate_sleeve(res: dict, cfg: dict, budget: float, skip: set, live: dict) -> list[dict]:
    m, reg = res["market"], res["regime"]["regime"]
    c, rets = res["candidates"], res.get("rets")
    rk, al, capital = cfg["risk"], cfg["allocation"], float(cfg["capital_aed"])
    unit = aed_per_unit(cfg, m)
    if budget <= 0 or c.empty:
        return []
    elig = c[c["eligible"].astype(bool) & ~c["veto"].astype(bool) & (c["units"] > 0) & ~c.index.isin(skip)]
    elig = elig.sort_values("conviction", ascending=False)
    if elig.empty:
        return []
    maxp = int(rk["max_positions"].get(reg, 0))
    min_pos = capital * float(al["min_position_pct"]) / 100.0
    n_target = int(max(1, min(maxp, np.floor(budget / min_pos)))) if min_pos > 0 else maxp
    corr = rets.corr() if rets is not None and rets.shape[1] > 1 else None
    chosen, sectors = [], {}
    for t, r in elig.iterrows():
        if len(chosen) >= n_target:
            break
        if corr is not None and t in corr.index and any(s in corr.columns and corr.loc[t, s] > float(rk["corr_threshold"]) for s in chosen):
            continue
        sec = r.get("sector") if isinstance(r.get("sector"), str) and r.get("sector") else None
        if sec and sectors.get(sec, 0) >= int(rk["sector_cap"]):
            continue
        chosen.append(t)
        if sec:
            sectors[sec] = sectors.get(sec, 0) + 1
    if not chosen:
        return []
    sub = c.loc[chosen]
    # inverse-volatility weights tilted by conviction
    inv_vol = 1.0 / sub["atr_pct"].astype(float).clip(lower=0.5)
    tilt = (sub["conviction"].astype(float) / 50.0).clip(0.5, 2.0) ** float(al["conviction_tilt"])
    w = (inv_vol * tilt)
    w = w / w.sum()
    aed = (w * budget).to_dict()
    # per-name caps: own risk budget (AED lost if stopped) and single-position cap; redistribute leftovers once
    max_pos = capital * float(rk["max_position_pct"]) / 100.0
    levels = {t: _apply_live(sub.loc[t], live.get(t)) for t in chosen}
    caps = {}
    for t in chosen:
        lv = levels[t]
        stop_frac = (lv["entry"] - lv["stop"]) / lv["entry"] if lv["entry"] > 0 and lv["entry"] > lv["stop"] else np.nan
        risk_cap = capital * _fin(sub.loc[t, "risk_pct"], 0.0) / 100.0
        caps[t] = min(max_pos, risk_cap / stop_frac if np.isfinite(stop_frac) and stop_frac > 0 else max_pos)
    for _ in range(2):
        over = {t: aed[t] - caps[t] for t in chosen if aed[t] > caps[t]}
        if not over:
            break
        spill = sum(over.values())
        for t in over:
            aed[t] = caps[t]
        room = {t: caps[t] - aed[t] for t in chosen if t not in over and caps[t] > aed[t]}
        if room and spill > 0:
            tot_room = sum(room.values())
            for t in room:
                aed[t] += min(room[t], spill * room[t] / tot_room)
    out = []
    for t in chosen:
        lv, r = levels[t], sub.loc[t]
        units = _round_units(aed[t] / unit / lv["entry"], m, cfg) if lv["entry"] > 0 else 0.0
        if units <= 0:
            continue
        pos_aed = units * lv["entry"] * unit
        risk_aed = units * (lv["entry"] - lv["stop"]) * unit
        order = ORDER_TEXT.get(lv["kind"], "Buy at market").format(entry=f"{lv['entry']:.4g}")
        out.append({"market": m, "ticker": t, "name": r.get("shortName") if isinstance(r.get("shortName"), str) else "",
                    "setup": r["setup"], "conviction": float(r["conviction"]), "entry": lv["entry"], "stop": lv["stop"],
                    "stop_pct": 100 * (lv["entry"] - lv["stop"]) / lv["entry"], "t1": lv["t1"], "t2": lv["t2"],
                    "units": units, "pos_native": units * lv["entry"], "currency": MARKET_META[m]["currency"],
                    "aed": pos_aed, "pct_capital": 100 * pos_aed / capital, "risk_aed": risk_aed,
                    "hold_days": int(r.get("hold_days", 15) or 15), "order": order, "why": r.get("why", ""),
                    "flags": [x for x in [r.get("earnings_flag", ""), *(r.get("plan_flags") or [])] if x],
                    "live_price": live.get(t)})
    return out


def portfolio_risk(positions: list[dict], results: list[dict], cfg: dict) -> dict:
    capital = float(cfg["capital_aed"])
    heat = sum(p["risk_aed"] for p in positions)
    deployed = sum(p["aed"] for p in positions)
    out = {"n_positions": len(positions), "deployed_aed": deployed, "deployed_pct": 100 * deployed / capital,
           "heat_aed": heat, "heat_pct": 100 * heat / capital}
    # one-day 1σ move from the 60-day covariance of the chosen names (inner-joined dates across markets)
    frames = []
    for res in results:
        rets = res.get("rets")
        if rets is None:
            continue
        cols = [p["ticker"] for p in positions if p["market"] == res["market"] and p["ticker"] in rets.columns]
        if cols:
            frames.append(rets[cols])
    if frames:
        R = pd.concat(frames, axis=1, join="inner").dropna()
        if len(R) >= 20 and R.shape[1] >= 1:
            w = np.array([next(p["aed"] for p in positions if p["ticker"] == t) for t in R.columns])
            cov = R.cov().to_numpy()
            var = float(w @ cov @ w)
            out["one_day_sigma_aed"] = float(np.sqrt(max(var, 0)))
            out["annual_vol_pct"] = 100 * float(np.sqrt(max(var, 0)) * np.sqrt(252) / capital)
            stand_alone = float(np.sum(w * np.sqrt(np.diag(cov))))
            out["diversification_ratio"] = stand_alone / out["one_day_sigma_aed"] if out["one_day_sigma_aed"] > 0 else np.nan
    return out


def build_plan(results: list[dict], cfg: dict, holdings: list[dict] | None = None, live: dict | None = None,
               macro_flags: dict | None = None) -> dict:
    live = live or {}
    capital = float(cfg["capital_aed"])
    sleeves = market_sleeves(results, cfg, macro_flags)
    held = review_holdings(holdings, results, cfg, live)
    kept = [h for h in held if h["action"] != "EXIT" and h["action"] != "REVIEW"]
    positions: list[dict] = []
    for res in results:
        m = res["market"]
        s = sleeves[m]
        kept_value = sum(h["value_aed"] for h in kept if h["market"] == m)
        s["held_aed"] = kept_value
        remaining = max(0.0, s["budget"] - kept_value)
        skip = {h["ticker"] for h in held if h["market"] == m}
        picks = allocate_sleeve(res, cfg, remaining, skip, live)
        s["new_aed"] = sum(p["aed"] for p in picks)
        positions += picks
    # account-level limits: total position count, then portfolio heat (sum of AED lost if every stop is hit)
    max_total = int(cfg["allocation"].get("max_total_positions", 8))
    positions.sort(key=lambda p: -p["conviction"])
    if len(positions) > max_total:
        positions = positions[:max_total]
    heat_cap = capital * float(cfg["risk"]["max_portfolio_heat_pct"]) / 100.0
    heat = sum(p["risk_aed"] for p in positions)
    if heat > heat_cap > 0:
        f = heat_cap / heat
        unit_by_m = {m: aed_per_unit(cfg, m) for m in {p["market"] for p in positions}}
        for p in positions:
            p["units"] = _round_units(p["units"] * f, p["market"], cfg)
            u = unit_by_m[p["market"]]
            p["pos_native"] = p["units"] * p["entry"]
            p["aed"] = p["pos_native"] * u
            p["pct_capital"] = 100 * p["aed"] / capital
            p["risk_aed"] = p["units"] * (p["entry"] - p["stop"]) * u
        positions = [p for p in positions if p["units"] > 0]
    for m, s in sleeves.items():
        s["new_aed"] = sum(p["aed"] for p in positions if p["market"] == m)
    held_value = sum(h["value_aed"] for h in kept)
    new_value = sum(p["aed"] for p in positions)
    cash = capital - held_value - new_value
    risk = portfolio_risk(positions, results, cfg)
    notes = []
    for m, s in sleeves.items():
        why = []
        if s["regime_mult"] < 1:
            why.append(f"{s['regime'].replace('_', ' ').lower()} regime ×{s['regime_mult']:.2f}")
        if s["ic_mult"] < 1:
            why.append(f"weak signal history (IC t={s['ic_t']:.1f}) ×{s['ic_mult']:.2f}")
        if s.get("macro_mult", 1.0) < 1:
            why.append(f"{s['macro_why']} ×{s['macro_mult']:.2f}")
        if why:
            notes.append(f"{s['label']}: budget trimmed — " + ", ".join(why))
    if cash / capital > 0.4:
        notes.append("Large cash balance is deliberate: it is what the regimes and the signal checks earn right now; it redeploys automatically as conditions improve.")
    return {"capital_aed": capital, "sleeves": sleeves, "positions": sorted(positions, key=lambda p: -p["aed"]),
            "holdings": held, "cash_aed": cash, "cash_pct": 100 * cash / capital, "held_aed": held_value,
            "new_aed": new_value, "risk": risk, "notes": notes, "live_prices_used": bool(live)}
