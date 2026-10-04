"""Feature engineering: technical features per ticker, factor-group definitions, fundamental mapping."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind

# Factor groups → (column, weight). Each column is z-scored cross-sectionally (higher = better).
FACTOR_GROUPS = {
    "trend":      {"ema_align": 1.0, "clenow": 1.0, "adx_dir": 1.0, "dist200": 0.5},
    "momentum":   {"mom_12_1": 1.0, "mom_6m": 1.0, "rs_3m": 1.0, "pct_52w_high": 1.0},
    "timing":     {"rsi_zone": 1.0, "ema21_fit": 1.0, "rsi2_inv": 0.75, "close_range3": 0.5},
    "volatility": {"bw_compression": 1.0, "atr_fit": 1.0, "vol_contraction": 0.75},
    "volume":     {"obv_slope": 1.0, "cmf20": 1.0, "updown_vol": 1.0},
}
FUNDAMENTAL_FACTORS = {
    "roe": 1.0, "gross_margin": 0.75, "low_leverage": 0.5, "fcf_yield": 1.0,          # quality
    "rev_growth": 1.0, "eps_growth": 0.75,                                            # growth
    "earnings_yield": 0.75, "ev_ebitda_inv": 0.5,                                     # value
    "analyst_score": 0.5, "target_upside": 0.5,                                       # street
}
SENTIMENT_FACTORS = {"news_sent": 1.0, "news_attention": 0.25, "funding_contrarian": 1.0}

TECH_FACTOR_COLS = sorted({c for g in FACTOR_GROUPS.values() for c in g})


def compute_features(df: pd.DataFrame, bench_close: pd.Series | None = None, atr_ideal_pct: float = 2.5,
                     horizon: int = 15, periods_per_year: int = 252) -> pd.DataFrame:
    """Full-history feature frame for one instrument (all rolling → causal; fwd_ret is research-only)."""
    c, h, l = df["Close"], df["High"], df["Low"]
    v = df["Volume"].fillna(0.0)
    f = pd.DataFrame(index=df.index)
    f["open"], f["high"], f["low"], f["close"], f["volume"] = df["Open"], h, l, c, v
    f["ret1"] = c.pct_change()

    a14 = ind.atr(df, 14)
    a5, a20 = ind.atr(df, 5), ind.atr(df, 20)
    f["atr"] = a14
    f["atr_pct"] = 100.0 * a14 / c
    f["ema8"], f["ema21"], f["ema50"], f["ema200"] = ind.ema(c, 8), ind.ema(c, 21), ind.ema(c, 50), ind.ema(c, 200)
    f["sma50"], f["sma200"] = ind.sma(c, 50), ind.sma(c, 200)
    f["rsi14"], f["rsi2"] = ind.rsi(c, 14), ind.rsi(c, 2)
    f["adx"], f["pdi"], f["mdi"] = ind.adx(df, 14)
    _, _, f["macd_hist"] = ind.macd(c)

    _, bb_up, bb_lo, f["bb_bw"], f["bb_pctb"] = ind.bollinger(c, 20, 2.0)
    f["bb_bw_pct"] = ind.rolling_rank_pct(f["bb_bw"], 252)
    _, kc_up, kc_lo = ind.keltner(df, 20, 1.5)
    f["squeeze_on"] = ((bb_up < kc_up) & (bb_lo > kc_lo)).astype(float)

    f["donch20_hi"], _ = ind.donchian(df, 20)
    f["donch55_hi"], _ = ind.donchian(df, 55)
    f["swing_low10"] = l.rolling(10, min_periods=5).min()

    obv = ind.obv(df)
    f["obv_slope"] = ind.rolling_slope(obv, 20) / v.rolling(20, min_periods=20).mean().replace(0.0, np.nan)
    f["cmf20"] = ind.cmf(df, 20)
    f["updown_vol"] = ind.updown_volume_ratio(df, 20)
    f["rvol"] = v.rolling(5, min_periods=5).mean() / v.rolling(50, min_periods=20).mean().replace(0.0, np.nan)

    f["clenow"], f["clenow_slope"], f["clenow_r2"] = ind.clenow_momentum(c, 90, periods_per_year)
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1.0
    f["mom_6m"] = c / c.shift(126) - 1.0
    f["mom_3m"] = c / c.shift(63) - 1.0
    f["mom_1m"] = c / c.shift(21) - 1.0
    if bench_close is not None and len(bench_close.dropna()):
        b = bench_close.reindex(f.index).ffill()
        f["rs_3m"] = f["mom_3m"] - (b / b.shift(63) - 1.0)
    else:
        f["rs_3m"] = f["mom_3m"]
    hi52 = h.rolling(252, min_periods=120).max()
    f["pct_52w_high"] = c / hi52
    f["adv20"] = (c * v).rolling(20, min_periods=10).mean()
    f["rv20"] = ind.realized_vol(c, 20, periods_per_year)

    rng = (h - l).replace(0.0, np.nan)
    f["close_range"] = (c - l) / rng
    f["close_range3"] = f["close_range"].rolling(3, min_periods=1).mean()
    f["dist21_atr"] = (c - f["ema21"]) / a14.replace(0.0, np.nan)
    f["dist200"] = c / f["sma200"] - 1.0
    align = (c > f["ema21"]).astype(int) + (f["ema21"] > f["ema50"]).astype(int) + (f["ema50"] > f["ema200"]).astype(int)
    f["ema_align"] = align.astype(float).where(f["ema200"].notna())
    f["above200"] = (c > f["sma200"]).astype(float).where(f["sma200"].notna())
    f["uptrend"] = ((c > f["sma200"]) & (f["ema21"] > f["ema50"])).astype(float).where(f["sma200"].notna())
    f["breakout20"] = (c > f["donch20_hi"]).astype(float)
    f["breakout55"] = (c > f["donch55_hi"]).astype(float)

    # scoring transforms (higher = better)
    f["adx_dir"] = (f["adx"] * np.sign(f["pdi"] - f["mdi"])).clip(-60, 60)
    f["rsi_zone"] = -(f["rsi14"] - 48.0).abs()
    f["ema21_fit"] = -(f["dist21_atr"] - 0.3).abs()
    f["rsi2_inv"] = 100.0 - f["rsi2"]
    f["bw_compression"] = -f["bb_bw_pct"]
    f["atr_fit"] = -np.log((f["atr_pct"] / atr_ideal_pct).replace(0.0, np.nan)).abs()
    f["vol_contraction"] = -(a5 / a20.replace(0.0, np.nan))

    f["fwd_ret"] = c.shift(-horizon) / c - 1.0   # research/backtest label only — never used for today's score
    return f


def build_panel(prices: dict[str, pd.DataFrame], bench_close: pd.Series | None, atr_ideal_pct: float,
                horizon: int, periods_per_year: int = 252) -> pd.DataFrame:
    frames = []
    for t, df in prices.items():
        f = compute_features(df, bench_close, atr_ideal_pct, horizon, periods_per_year)
        f["ticker"] = t
        frames.append(f)
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.set_index("ticker", append=True).sort_index()
    return panel


def latest_cross_section(panel: pd.DataFrame, max_stale_days: int = 7) -> pd.DataFrame:
    """Last row per ticker (index = ticker, 'date' column), dropping stale instruments."""
    last_date = panel.index.get_level_values(0).max()
    latest = panel.groupby(level="ticker").tail(1).reset_index(level="date")
    return latest[latest["date"] >= last_date - pd.Timedelta(days=max_stale_days)].copy()


def breadth_series(panel: pd.DataFrame) -> pd.Series:
    above = (panel["close"] > panel["sma50"]).where(panel["sma50"].notna())
    return above.groupby(level=0).mean()


def fundamental_factors(fund: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Map raw Yahoo fields to quality / growth / value / street factors (higher = better)."""
    def num(col):
        if col in fund.columns:
            return pd.to_numeric(fund[col], errors="coerce").astype(float)
        return pd.Series(np.nan, index=fund.index, dtype=float)

    g = pd.DataFrame(index=fund.index)
    g["roe"] = num("returnOnEquity").clip(-1, 1)
    g["gross_margin"] = num("grossMargins").clip(-1, 1)
    g["low_leverage"] = -np.log1p(num("debtToEquity").clip(lower=0) / 100.0)
    g["fcf_yield"] = (num("freeCashflow") / num("marketCap")).clip(-0.3, 0.3)
    g["rev_growth"] = num("revenueGrowth").clip(-1, 2)
    g["eps_growth"] = num("earningsGrowth").clip(-1, 3)
    fpe = num("forwardPE")
    g["earnings_yield"] = (1.0 / fpe.where(fpe > 0)).clip(upper=0.5)
    ev = num("enterpriseToEbitda")
    g["ev_ebitda_inv"] = (1.0 / ev.where(ev > 0)).clip(upper=1.0)
    g["analyst_score"] = -num("recommendationMean")
    g["target_upside"] = (num("targetMeanPrice") / close.reindex(fund.index) - 1.0).clip(-0.5, 1.0)
    return g
