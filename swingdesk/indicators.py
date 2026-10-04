"""Technical indicators implemented in pandas/numpy (no TA-Lib dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["Close"].shift(1)
    return pd.concat(
        [df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = ru / rd.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.mask((rd == 0) & ru.notna(), 100.0)
    return out


def adx(df: pd.DataFrame, n: int = 14):
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr_ = true_range(df).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().replace(0.0, np.nan)
    pdi = 100.0 * plus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_
    mdi = 100.0 * minus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean(), pdi, mdi


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    width = (upper - lower) / mid.replace(0.0, np.nan)
    pctb = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return mid, upper, lower, width, pctb


def keltner(df: pd.DataFrame, n: int = 20, mult: float = 1.5):
    mid = ema(df["Close"], n)
    a = atr(df, n)
    return mid, mid + mult * a, mid - mult * a


def donchian(df: pd.DataFrame, n: int, shift: int = 1):
    """Prior n-bar high/low (shifted so today's bar can be tested for a breakout)."""
    hi = df["High"].rolling(n, min_periods=n).max().shift(shift)
    lo = df["Low"].rolling(n, min_periods=n).min().shift(shift)
    return hi, lo


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["Close"].diff()).fillna(0.0)
    return (sign * df["Volume"].fillna(0.0)).cumsum()


def cmf(df: pd.DataFrame, n: int = 20) -> pd.Series:
    hl = (df["High"] - df["Low"]).replace(0.0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl
    mfv = (mfm * df["Volume"]).fillna(0.0)
    return mfv.rolling(n, min_periods=n).sum() / df["Volume"].rolling(n, min_periods=n).sum().replace(0.0, np.nan)


def rolling_slope(y: pd.Series, n: int) -> pd.Series:
    """Rolling OLS slope of y against bar index (vectorised: cov(y,t)/var(t))."""
    t = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    return y.rolling(n, min_periods=n).cov(t) / t.rolling(n, min_periods=n).var()


def rolling_r2(y: pd.Series, n: int) -> pd.Series:
    t = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    r = y.rolling(n, min_periods=n).corr(t)
    return r * r


def clenow_momentum(close: pd.Series, n: int = 90, periods_per_year: int = 252):
    """Clenow momentum: annualised exponential-regression slope × R² (trend strength × smoothness)."""
    ly = np.log(close.where(close > 0))
    slope = rolling_slope(ly, n)
    r2 = rolling_r2(ly, n)
    ann = np.exp(slope * periods_per_year) - 1.0
    return ann * r2, ann, r2


def realized_vol(close: pd.Series, n: int = 20, periods_per_year: int = 252) -> pd.Series:
    return np.log(close / close.shift(1)).rolling(n, min_periods=n).std() * np.sqrt(periods_per_year)


def updown_volume_ratio(df: pd.DataFrame, n: int = 20) -> pd.Series:
    ret = df["Close"].diff()
    v = df["Volume"].fillna(0.0)
    upv = v.where(ret > 0, 0.0).rolling(n, min_periods=n).sum()
    dnv = v.where(ret < 0, 0.0).rolling(n, min_periods=n).sum()
    return np.log((upv + 1.0) / (dnv + 1.0))


def rolling_rank_pct(s: pd.Series, n: int = 252) -> pd.Series:
    return s.rolling(n, min_periods=max(60, n // 4)).rank(pct=True)
