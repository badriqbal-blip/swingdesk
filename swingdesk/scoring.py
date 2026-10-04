"""Cross-sectional scoring: z-scores → factor-group scores → regime-weighted composite; setup classification."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import FACTOR_GROUPS

REGIME_ORDER = {"CRISIS": 0, "RISK_OFF": 1, "NEUTRAL": 2, "RISK_ON": 3}


# ───────────────────────────────────────── z-scores ──────────────────────────────────────────────
def zscore_cs(df: pd.DataFrame, cols: list[str], level: int = 0) -> pd.DataFrame:
    """Cross-sectional z-score within each date (MultiIndex level 0), clipped to ±3."""
    g = df.groupby(level=level)
    mu = g[cols].transform("mean")
    sd = g[cols].transform("std")
    return ((df[cols] - mu) / sd.replace(0.0, np.nan)).clip(-3, 3)


def zscore_flat(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    mu, sd = df[cols].mean(), df[cols].std()
    return ((df[cols] - mu) / sd.replace(0.0, np.nan)).clip(-3, 3)


def _weighted_nanmean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(values)
    w = np.where(mask, weights, 0.0)
    den = w.sum(axis=1)
    num = (np.where(mask, values, 0.0) * w).sum(axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def group_scores(z: pd.DataFrame, groups: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = pd.DataFrame(index=z.index)
    for g, fw in groups.items():
        cols = [c for c in fw if c in z.columns]
        if not cols:
            continue
        w = np.array([fw[c] for c in cols], dtype=float)
        out[f"g_{g}"] = _weighted_nanmean(z[cols].to_numpy(dtype=float), np.tile(w, (len(z), 1)))
    return out


def composite(gs: pd.DataFrame, regime, weights_cfg: dict) -> pd.Series:
    """Regime-weighted composite of g_* columns; weights renormalised over the groups a row actually has."""
    groups = [c[2:] for c in gs.columns if c.startswith("g_")]
    if isinstance(regime, str):
        regime = pd.Series(regime, index=gs.index)
    regime = regime.fillna("NEUTRAL")
    W = np.column_stack([
        regime.map(lambda r, g=g: float(weights_cfg.get(r, weights_cfg["NEUTRAL"]).get(g, 0.0))).to_numpy(dtype=float)
        for g in groups
    ]) if groups else np.zeros((len(gs), 0))
    vals = gs[[f"g_{g}" for g in groups]].to_numpy(dtype=float)
    return pd.Series(_weighted_nanmean(vals, W), index=gs.index)


def score_panel(panel: pd.DataFrame, regime_labels: pd.Series, weights_cfg: dict) -> pd.DataFrame:
    """Score the full history panel (date, ticker): z-scores, group scores, composite and percentile rank."""
    cols = [c for c in {c for g in FACTOR_GROUPS.values() for c in g} if c in panel.columns]
    z = zscore_cs(panel, cols)
    gs = group_scores(z, FACTOR_GROUPS)
    dates = panel.index.get_level_values(0)
    reg_by_date = regime_labels.reindex(dates.unique()).ffill().bfill().fillna("NEUTRAL")
    reg = pd.Series(reg_by_date.reindex(dates).to_numpy(), index=panel.index)
    comp = composite(gs, reg, weights_cfg)
    out = panel.join(z.add_prefix("z_")).join(gs)
    out["regime"] = reg
    out["composite"] = comp
    out["comp_rank"] = out.groupby(level=0)["composite"].rank(pct=True)
    return out


# ────────────────────────────────────── setup playbook ───────────────────────────────────────────
SETUP_INFO = {
    "Breakout":               dict(kind="breakout", hold=15, eligible=True,
                                   entry="Buy at market / next open. Invalid if it closes back below the breakout level."),
    "Squeeze (pre-breakout)": dict(kind="squeeze", hold=15, eligible=True,
                                   entry="Buy-stop just above the 20-day high (+0.1 ATR). Let price trigger it; don't anticipate."),
    "Pullback in uptrend":    dict(kind="pullback", hold=15, eligible=True,
                                   entry="Limit near the 21-EMA, or at market on the first up-close / reclaim day."),
    "Mean-reversion bounce":  dict(kind="bounce", hold=7, eligible=True,
                                   entry="Buy at market; exit on a close above the 8-EMA or after 5–7 days."),
    "Momentum continuation":  dict(kind="momentum", hold=15, eligible=True,
                                   entry="Half-size starter at market; add on a 2–3 day tight consolidation."),
    "Extended (wait)":        dict(kind="extended", hold=15, eligible=False,
                                   entry="Too far above the 21-EMA. Set an alert for a pullback toward it."),
    "Counter-trend bounce":   dict(kind="counter", hold=5, eligible=False,
                                   entry="Speculative: below the 200-day. Small size only, if at all."),
    "No clean setup":         dict(kind="none", hold=0, eligible=False, entry="No actionable pattern today."),
}


def _f(r, k, default=np.nan):
    v = r.get(k, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(v) else v


def classify_setup(r) -> str:
    up = _f(r, "uptrend", 0) == 1
    above200 = _f(r, "above200", 0) == 1
    d21 = _f(r, "dist21_atr")
    rsi14, rsi2 = _f(r, "rsi14"), _f(r, "rsi2")
    if up and _f(r, "breakout20", 0) == 1 and _f(r, "rvol", 1) >= 1.2 and _f(r, "close_range", 0) >= 0.5:
        return "Breakout"
    if up and _f(r, "squeeze_on", 0) == 1 and abs(d21) <= 1.0:
        return "Squeeze (pre-breakout)"
    if up and -1.5 <= d21 <= 0.75 and 35 <= rsi14 <= 60:
        return "Pullback in uptrend"
    if above200 and rsi2 <= 10 and _f(r, "close") >= 0.97 * _f(r, "sma50", np.inf):
        return "Mean-reversion bounce"
    if up and _f(r, "pct_52w_high", 0) >= 0.93 and _f(r, "mom_6m", 0) > 0.10:
        return "Momentum continuation" if d21 <= 2.0 else "Extended (wait)"
    if up and d21 > 2.0:
        return "Extended (wait)"
    if not above200 and rsi14 <= 30 and _f(r, "close_range3", 0) >= 0.6:
        return "Counter-trend bounce"
    return "No clean setup"


def conviction_from_composite(comp) -> np.ndarray:
    comp = np.asarray(comp, dtype=float)
    return 100.0 / (1.0 + np.exp(-1.6 * comp))


def explain(r) -> str:
    bits = []
    if _f(r, "g_momentum", 0) > 0.8:
        bits.append("momentum leader")
    elif _f(r, "g_momentum", 0) < -0.8:
        bits.append("weak momentum")
    if _f(r, "g_trend", 0) > 0.8:
        bits.append("strong trend structure")
    if _f(r, "g_timing", 0) > 0.5:
        bits.append("favourable entry timing")
    if _f(r, "g_volatility", 0) > 0.5:
        bits.append("volatility compression")
    if _f(r, "g_volume", 0) > 0.5:
        bits.append("volume accumulation")
    elif _f(r, "g_volume", 0) < -0.8:
        bits.append("distribution in volume")
    if _f(r, "g_fundamentals", 0) > 0.5:
        bits.append("quality/growth fundamentals")
    elif _f(r, "g_fundamentals", 0) < -0.8:
        bits.append("weak fundamentals")
    ns = _f(r, "news_sent")
    if not np.isnan(ns):
        bits.append("positive news flow" if ns > 0.15 else "negative news flow" if ns < -0.15 else "neutral news")
    if _f(r, "squeeze_on", 0) == 1:
        bits.append("Bollinger/Keltner squeeze on")
    if _f(r, "breakout55", 0) == 1:
        bits.append("new 55-day high")
    if _f(r, "ml_prob", np.nan) >= 0.6:
        bits.append(f"ML p={_f(r, 'ml_prob'):.2f}")
    return "; ".join(bits) if bits else "no dominant factor"
