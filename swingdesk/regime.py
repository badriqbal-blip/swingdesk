"""Market regime detection: trend, volatility, breadth, VIX and credit → RISK_ON / NEUTRAL / RISK_OFF / CRISIS."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_regime(bench_close: pd.Series, breadth: pd.Series | None = None, vix: pd.Series | None = None,
                   credit_ratio: pd.Series | None = None, periods_per_year: int = 252) -> pd.DataFrame:
    c = bench_close.dropna().astype(float)
    idx = c.index
    sma200 = c.rolling(200, min_periods=120).mean()
    sma50 = c.rolling(50, min_periods=30).mean()
    rv = np.log(c / c.shift(1)).rolling(20, min_periods=15).std() * np.sqrt(periods_per_year)
    rv_pct = rv.rolling(252, min_periods=60).rank(pct=True)

    comp = pd.DataFrame(index=idx)
    comp["trend200"] = np.where(c > sma200, 1.0, -1.0)
    comp["slope50"] = np.where(sma50 > sma50.shift(10), 0.5, -0.5)
    comp["above50"] = np.where(c > sma50, 0.5, -0.5)
    comp["vol"] = np.where(rv_pct > 0.85, -1.0, np.where(rv_pct < 0.5, 0.5, 0.0))
    b = v = cr = None
    if breadth is not None:
        b = breadth.reindex(idx).ffill()
        comp["breadth"] = np.where(b > 0.6, 1.0, np.where(b < 0.4, -1.0, 0.0))
    if vix is not None:
        v = vix.reindex(idx).ffill()
        comp["vix"] = np.where(v > 30, -1.5, np.where(v > 22, -0.5, np.where(v < 16, 0.5, 0.0)))
    if credit_ratio is not None:
        cr = credit_ratio.reindex(idx).ffill()
        comp["credit"] = np.where(cr / cr.shift(20) - 1.0 > 0, 0.5, -0.5)

    score = comp.sum(axis=1).rolling(3, min_periods=1).mean()
    score = score.where(sma200.notna(), 0.0)
    label = pd.Series("NEUTRAL", index=idx, dtype=object)
    label[score >= 2.0] = "RISK_ON"
    label[score <= -1.0] = "RISK_OFF"
    crisis = score <= -3.5
    if v is not None:
        crisis = crisis | (v > 35)
    label[crisis] = "CRISIS"

    out = comp.copy()
    out["score"] = score
    out["regime"] = label
    out["bench"] = c
    out["sma200"], out["sma50"] = sma200, sma50
    out["rv"], out["rv_pct"] = rv, rv_pct
    if b is not None:
        out["breadth_val"] = b
    if v is not None:
        out["vix_val"] = v
    if cr is not None:
        out["credit_mom"] = cr / cr.shift(20) - 1.0
    return out


def regime_snapshot(reg: pd.DataFrame, bench_name: str) -> dict:
    r = reg.iloc[-1]
    snap = {
        "date": str(reg.index[-1].date()),
        "regime": str(r["regime"]),
        "score": round(float(r["score"]), 2),
        "bench": bench_name,
        "bench_close": round(float(r["bench"]), 2),
        "above_200d": bool(r["bench"] > r["sma200"]) if pd.notna(r["sma200"]) else None,
        "above_50d": bool(r["bench"] > r["sma50"]) if pd.notna(r["sma50"]) else None,
        "sma50_rising": bool(r["slope50"] > 0),
        "realized_vol_pct": round(float(r["rv_pct"]) * 100, 0) if pd.notna(r["rv_pct"]) else None,
        "realized_vol_ann": round(float(r["rv"]) * 100, 1) if pd.notna(r["rv"]) else None,
    }
    if "breadth_val" in reg.columns and pd.notna(r["breadth_val"]):
        snap["breadth_pct_above_50d"] = round(float(r["breadth_val"]) * 100, 0)
    if "vix_val" in reg.columns and pd.notna(r["vix_val"]):
        snap["vix"] = round(float(r["vix_val"]), 1)
    if "credit_mom" in reg.columns and pd.notna(r["credit_mom"]):
        snap["credit_risk_appetite"] = "improving" if r["credit_mom"] > 0 else "deteriorating"
    # how long the current regime has persisted
    lab = reg["regime"].to_numpy()
    n = 1
    while n < len(lab) and lab[-n - 1] == lab[-1]:
        n += 1
    snap["regime_age_bars"] = int(n)
    return snap
