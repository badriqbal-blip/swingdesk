"""Risk engine: fixed-fractional sizing, structure/ATR stops, R-multiple targets, portfolio constraints, Kelly."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scoring import SETUP_INFO


def _fin(x, default=np.nan):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(x) else x


def plan_trade(r, setup: str, cfg: dict, regime: str, capital_aed: float, aed_unit: float,
               market: str, size_mult: float = 1.0) -> dict | None:
    """Entry / stop / targets / units for one candidate. Returns None if no valid plan."""
    rk = cfg["risk"]
    close, atr = _fin(r.get("close")), _fin(r.get("atr"))
    if not (close > 0 and atr > 0):
        return None
    info = SETUP_INFO.get(setup, SETUP_INFO["No clean setup"])
    k, kmax = float(rk["stop_atr_mult"]), float(rk["max_stop_atr_mult"])

    entry = close
    d20 = _fin(r.get("donch20_hi"))
    e21 = _fin(r.get("ema21"))
    if info["kind"] == "squeeze" and np.isfinite(d20):
        entry = max(close, d20 + 0.1 * atr)                  # buy-stop above the range
    elif info["kind"] == "pullback" and np.isfinite(e21):
        entry = min(close, max(e21, close - 0.5 * atr))      # limit toward the 21-EMA (never a fantasy fill)

    stop = entry - k * atr
    sl = _fin(r.get("swing_low10"))
    if np.isfinite(sl):
        stop = min(stop, sl - 0.25 * atr)                    # below structure if it is lower
    stop = max(stop, entry - kmax * atr)                     # but never absurdly wide
    R = entry - stop
    if R <= 0:
        return None

    risk_pct = float(rk["risk_per_trade_pct"].get(regime, 0.0)) * float(size_mult)
    risk_aed = capital_aed * risk_pct / 100.0
    units = (risk_aed / aed_unit) / R if risk_aed > 0 else 0.0
    cap_aed = capital_aed * float(rk["max_position_pct"]) / 100.0
    pos_aed = units * entry * aed_unit
    capped = False
    if pos_aed > cap_aed > 0:
        units *= cap_aed / pos_aed
        capped = True
    if market == "uae" or (market == "us" and not rk.get("fractional_shares_us", True)):
        units = float(np.floor(units))
    elif market == "crypto":
        units = float(np.floor(units * 1e6) / 1e6)
    else:
        units = float(np.floor(units * 1e4) / 1e4)

    pos_native = units * entry
    pos_aed = pos_native * aed_unit
    risk_aed_actual = units * R * aed_unit
    t1 = entry + float(rk["target1_r"]) * R
    t2 = entry + float(rk["target2_r"]) * R
    cost_aed = pos_aed * 2.0 * (float(rk["commission_pct"]) + float(rk["slippage_pct"])) / 100.0
    flags = []
    if capped:
        flags.append(f"size capped at {rk['max_position_pct']}% of capital")
    if units <= 0:
        flags.append("too small: 1-share minimum exceeds risk budget")
    if risk_aed_actual > 0 and cost_aed / risk_aed_actual > 0.15:
        flags.append("costs > 15% of risk budget")
    d55 = _fin(r.get("donch55_hi"))
    if np.isfinite(d55) and entry < d55 < t1:
        flags.append("55-day high sits before T1 (resistance)")
    return {
        "entry": entry, "stop": stop, "R": R, "stop_pct": 100.0 * R / entry, "t1": t1, "t2": t2,
        "units": units, "pos_native": pos_native, "pos_aed": pos_aed, "risk_aed": risk_aed_actual,
        "risk_pct": risk_pct, "cost_aed": cost_aed, "hold_days": info["hold"], "entry_note": info["entry"],
        "plan_flags": flags,
        "exit_rule": (f"Stop {stop:.4g}; to breakeven at +{rk['breakeven_at_r']}R; trail {rk['trail_atr_mult']}×ATR "
                      f"from +{rk['trail_after_r']}R; scale 1/2 at T1 (+{rk['target1_r']}R); time-stop {rk['time_stop_days']} bars if < +0.5R"),
    }


def select_portfolio(c: pd.DataFrame, rets: pd.DataFrame | None, cfg: dict, regime: str, capital_aed: float):
    """Greedy selection by conviction under correlation / sector / heat / exposure / count constraints."""
    rk = cfg["risk"]
    maxp = int(rk["max_positions"].get(regime, 0))
    heat_cap = capital_aed * float(rk["max_portfolio_heat_pct"]) / 100.0
    expo_cap = capital_aed * float(rk["max_total_exposure_pct"]) / 100.0
    corr = None
    if rets is not None and rets.shape[1] > 1:
        corr = rets.corr()
    sel, reasons, sectors = [], {}, {}
    heat = expo = 0.0
    for t, r in c.sort_values("conviction", ascending=False).iterrows():
        if len(sel) >= maxp:
            reasons[t] = f"max {maxp} positions in {regime}"
            continue
        if not bool(r.get("eligible", False)):
            reasons[t] = "not actionable today"
            continue
        if bool(r.get("veto", False)):
            reasons[t] = str(r.get("earnings_flag") or "vetoed")
            continue
        if not (_fin(r.get("units"), 0.0) > 0):
            reasons[t] = "position too small for minimum lot"
            continue
        if corr is not None and t in corr.index:
            hi = [(s, corr.loc[t, s]) for s in sel if s in corr.columns and corr.loc[t, s] > float(rk["corr_threshold"])]
            if hi:
                reasons[t] = f"correlated with {hi[0][0]} ({hi[0][1]:.2f})"
                continue
        sec = r.get("sector") if isinstance(r.get("sector"), str) and r.get("sector") else None
        if sec and sectors.get(sec, 0) >= int(rk["sector_cap"]):
            reasons[t] = f"sector cap ({sec})"
            continue
        if heat + _fin(r.get("risk_aed"), 0.0) > heat_cap + 1e-9:
            reasons[t] = "portfolio heat cap"
            continue
        if expo + _fin(r.get("pos_aed"), 0.0) > expo_cap + 1e-9:
            reasons[t] = "capital fully deployed"
            continue
        sel.append(t)
        heat += _fin(r.get("risk_aed"), 0.0)
        expo += _fin(r.get("pos_aed"), 0.0)
        if sec:
            sectors[sec] = sectors.get(sec, 0) + 1
    c = c.copy()
    c["selected"] = c.index.isin(sel)
    c["skip_reason"] = pd.Series(reasons, dtype=object).reindex(c.index).fillna("")
    summary = {"n": len(sel), "max_positions": maxp, "heat_aed": heat, "heat_cap_aed": heat_cap,
               "exposure_aed": expo, "exposure_cap_aed": expo_cap, "selected": sel}
    return c, summary


def kelly_fraction(win_rate: float, avg_win_r: float, avg_loss_r: float) -> float:
    """Full Kelly fraction of capital per trade given R-multiple stats (use ¼–½ of this in practice)."""
    if not (0 < win_rate < 1) or not (avg_win_r > 0) or not (avg_loss_r < 0):
        return float("nan")
    b = avg_win_r / abs(avg_loss_r)
    return max(0.0, win_rate - (1.0 - win_rate) / b)


def returns_matrix(prices: dict[str, pd.DataFrame], tickers: list[str], lookback: int = 60) -> pd.DataFrame:
    cols = {t: prices[t]["Close"].pct_change().tail(lookback) for t in tickers if t in prices}
    return pd.DataFrame(cols)
