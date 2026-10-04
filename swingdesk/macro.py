"""Macro & commodities: a cross-asset dashboard (oil, gold, silver, copper, gas, yields, dollar, credit, EM, UAE ETF),
plain-English cross-asset reads, CFTC speculative positioning (the supply/demand of futures money) and US sector
rotation. Everything is read-only context that feeds the plan as notes and as two transparent budget nudges
(oil → UAE sleeve, dollar → crypto sleeve).
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import requests

from .data import UA, cache_get, cache_put

MACRO_ASSETS = {
    "WTI crude": "CL=F", "Brent crude": "BZ=F", "Gold": "GC=F", "Silver": "SI=F", "Copper": "HG=F",
    "Natural gas": "NG=F", "US 10y yield": "^TNX", "Dollar index": "DX-Y.NYB", "VIX": "^VIX",
    "S&P 500": "SPY", "Nasdaq 100": "QQQ", "US 20y Treasuries": "TLT", "High-yield credit": "HYG",
    "Emerging markets": "EEM", "UAE equities ETF": "UAE", "Bitcoin": "BTC-USD",
}
SECTOR_ETFS = {"XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health care", "XLI": "Industrials",
               "XLY": "Consumer discretionary", "XLP": "Consumer staples", "XLU": "Utilities", "XLB": "Materials",
               "SMH": "Semiconductors", "XBI": "Biotech", "GLD": "Gold", "SLV": "Silver", "TLT": "Long bonds",
               "IWM": "Small caps", "EEM": "Emerging markets", "KWEB": "China internet"}

COT_MARKETS = {  # CFTC legacy futures-only report names (prefix match)
    "WTI crude": "CRUDE OIL, LIGHT SWEET", "Gold": "GOLD - COMMODITY EXCHANGE", "Silver": "SILVER - COMMODITY EXCHANGE",
    "Copper": "COPPER-GRADE #1", "Natural gas": "NATURAL GAS - NEW YORK MERCANTILE", "Bitcoin": "BITCOIN - CHICAGO MERCANTILE",
    "S&P 500 (e-mini)": "E-MINI S&P 500", "US dollar index": "USD INDEX - ICE", "10y Treasury": "UST 10Y NOTE",
}


def _chg(s: pd.Series, n: int) -> float:
    if len(s) <= n or s.iloc[-n - 1] == 0 or pd.isna(s.iloc[-n - 1]):
        return np.nan
    return 100.0 * (s.iloc[-1] / s.iloc[-n - 1] - 1.0)


def macro_dashboard(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, tk in MACRO_ASSETS.items():
        df = prices.get(tk)
        if df is None or len(df) < 30:
            continue
        c = df["Close"].dropna()
        sma200 = c.rolling(200, min_periods=120).mean().iloc[-1]
        sma20 = c.rolling(20).mean()
        hi, lo = c.tail(252).max(), c.tail(252).min()
        yield_like = tk == "^TNX"
        rows.append({
            "asset": name, "ticker": tk, "last": float(c.iloc[-1]),
            "chg_1d": float(c.iloc[-1] - c.iloc[-2]) if yield_like else _chg(c, 1),
            "chg_1w": float(c.iloc[-1] - c.iloc[-6]) if yield_like and len(c) > 6 else _chg(c, 5),
            "chg_1m": float(c.iloc[-1] - c.iloc[-22]) if yield_like and len(c) > 22 else _chg(c, 21),
            "chg_3m": float(c.iloc[-1] - c.iloc[-64]) if yield_like and len(c) > 64 else _chg(c, 63),
            "vs_200d": 100.0 * (c.iloc[-1] / sma200 - 1.0) if pd.notna(sma200) and sma200 else np.nan,
            "trend_20d": "up" if sma20.iloc[-1] > sma20.iloc[-6] else "down",
            "pct_52w_range": 100.0 * (c.iloc[-1] - lo) / (hi - lo) if hi > lo else np.nan,
            "unit": "pts" if yield_like else "%",
        })
    return pd.DataFrame(rows)


def cross_asset_read(dash: pd.DataFrame) -> tuple[list[str], dict]:
    """Plain-English reads plus flags the allocator uses (oil_bull, dollar_strong, yields_spiking, risk_appetite)."""
    g = dash.set_index("asset") if not dash.empty else pd.DataFrame()
    notes: list[str] = []
    flags = {"oil_bull": None, "dollar_strong": None, "yields_spiking": None, "risk_appetite": None}

    def val(a, col):
        return float(g.loc[a, col]) if a in g.index and pd.notna(g.loc[a, col]) else np.nan

    oil_200, oil_1m, oil_tr = val("WTI crude", "vs_200d"), val("WTI crude", "chg_1m"), g.loc["WTI crude", "trend_20d"] if "WTI crude" in g.index else None
    if np.isfinite(oil_200):
        if oil_200 > 0 and oil_tr == "up":
            flags["oil_bull"] = True
            notes.append(f"Oil is in an uptrend (WTI {oil_200:+.0f}% vs its 200-day, {oil_1m:+.1f}% on the month): supportive for Gulf equities, banks and energy; an inflation headwind for long-duration growth and bonds.")
        elif oil_200 < -5 and oil_tr == "down":
            flags["oil_bull"] = False
            notes.append(f"Oil is weak (WTI {oil_200:+.0f}% vs its 200-day, {oil_1m:+.1f}% on the month): a drag on Gulf sentiment and energy names, a relief for consumer and transport sectors.")
        else:
            flags["oil_bull"] = None
            notes.append(f"Oil is range-bound (WTI {oil_200:+.0f}% vs 200-day): neutral for the UAE sleeve.")
    gold_1m, silver_1m, y_1m = val("Gold", "chg_1m"), val("Silver", "chg_1m"), val("US 10y yield", "chg_1m")
    if np.isfinite(gold_1m):
        if gold_1m > 3 and (not np.isfinite(y_1m) or y_1m <= 0):
            notes.append(f"Gold +{gold_1m:.1f}% in a month with yields not rising: haven/defensive bid — the market is hedging; favour quality and keep stops honest.")
        elif gold_1m > 3 and y_1m > 0:
            notes.append(f"Gold +{gold_1m:.1f}% alongside rising yields: an inflation trade rather than a fear trade; commodities and miners lead.")
        elif gold_1m < -3:
            notes.append(f"Gold {gold_1m:+.1f}% on the month: risk appetite is pulling money out of havens.")
        if np.isfinite(silver_1m) and silver_1m - gold_1m > 4:
            notes.append(f"Silver outrunning gold ({silver_1m:+.1f}% vs {gold_1m:+.1f}%): the industrial/high-beta side of metals is bid — a risk-on tell.")
    cu_1m = val("Copper", "chg_1m")
    if np.isfinite(cu_1m) and np.isfinite(gold_1m):
        if cu_1m - gold_1m > 4:
            notes.append(f"Copper/gold ratio rising (copper {cu_1m:+.1f}% vs gold {gold_1m:+.1f}%): growth expectations improving — supports cyclicals, EM and crypto.")
        elif gold_1m - cu_1m > 4:
            notes.append(f"Copper/gold ratio falling (copper {cu_1m:+.1f}% vs gold {gold_1m:+.1f}%): growth worries — favour defensives, trim cyclicals.")
    dxy_1m, dxy_200 = val("Dollar index", "chg_1m"), val("Dollar index", "vs_200d")
    if np.isfinite(dxy_1m):
        flags["dollar_strong"] = dxy_1m > 2 or (dxy_200 > 2 and dxy_1m > 0)
        if flags["dollar_strong"]:
            notes.append(f"Dollar strengthening ({dxy_1m:+.1f}% on the month, {dxy_200:+.1f}% vs 200-day): headwind for commodities, EM and crypto; AED is pegged so no direct FX effect for you.")
        elif dxy_1m < -2:
            notes.append(f"Dollar weakening ({dxy_1m:+.1f}% on the month): tailwind for gold, EM and crypto.")
    if np.isfinite(y_1m):
        flags["yields_spiking"] = y_1m > 0.3
        if y_1m > 0.3:
            notes.append(f"10-year yield up {y_1m:.2f} pts in a month: pressure on growth/tech multiples and on REITs; banks benefit.")
        elif y_1m < -0.3:
            notes.append(f"10-year yield down {abs(y_1m):.2f} pts in a month: relief for growth, housing and gold.")
    hy_1m, tlt_1m, eem_1m, spy_1m = val("High-yield credit", "chg_1m"), val("US 20y Treasuries", "chg_1m"), val("Emerging markets", "chg_1m"), val("S&P 500", "chg_1m")
    if np.isfinite(hy_1m) and np.isfinite(tlt_1m):
        flags["risk_appetite"] = hy_1m > tlt_1m
        notes.append("Credit risk appetite: high-yield is beating Treasuries this month — risk-on." if hy_1m > tlt_1m
                     else "Credit risk appetite: Treasuries are beating high-yield this month — risk-off undertone.")
    if np.isfinite(eem_1m) and np.isfinite(spy_1m):
        notes.append(f"Emerging markets {'leading' if eem_1m > spy_1m else 'lagging'} the S&P this month ({eem_1m:+.1f}% vs {spy_1m:+.1f}%).")
    uae_1m, uae_200 = val("UAE equities ETF", "chg_1m"), val("UAE equities ETF", "vs_200d")
    if np.isfinite(uae_1m):
        notes.append(f"UAE equities ETF {uae_1m:+.1f}% on the month, {uae_200:+.0f}% vs its 200-day — the foreign-investor read on Dubai/Abu Dhabi.")
    btc_1m = val("Bitcoin", "chg_1m")
    if np.isfinite(btc_1m):
        notes.append(f"Bitcoin {btc_1m:+.1f}% on the month, {val('Bitcoin', 'vs_200d'):+.0f}% vs 200-day — the liquidity barometer for the crypto sleeve.")
    return notes, flags


def sector_rotation(prices: dict[str, pd.DataFrame], bench: str = "SPY") -> pd.DataFrame:
    b = prices.get(bench)
    if b is None:
        return pd.DataFrame()
    bc = b["Close"].dropna()
    rows = []
    for tk, name in SECTOR_ETFS.items():
        df = prices.get(tk)
        if df is None or len(df) < 70:
            continue
        c = df["Close"].dropna()
        rows.append({"sector": name, "ticker": tk, "rel_1w": _chg(c, 5) - _chg(bc, 5), "rel_1m": _chg(c, 21) - _chg(bc, 21),
                     "rel_3m": _chg(c, 63) - _chg(bc, 63), "abs_1m": _chg(c, 21),
                     "above_50d": bool(c.iloc[-1] > c.rolling(50).mean().iloc[-1])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["score"] = df["rel_1w"].rank() + df["rel_1m"].rank() * 2 + df["rel_3m"].rank()
    return df.sort_values("score", ascending=False).drop(columns="score").reset_index(drop=True)


# ───────────────────────────────── CFTC commitments of traders ───────────────────────────────────
_COT_COLS = ["market", "yymmdd", "date", "code", "mkt_code", "region", "commodity", "oi", "nc_long", "nc_short", "nc_spread",
             "c_long", "c_short", "rep_long", "rep_short", "nr_long", "nr_short"]


def _parse_cot(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), header=None, usecols=range(17), names=_COT_COLS, dtype=str, on_bad_lines="skip")
    if df["market"].iloc[0].strip().lower().startswith("market"):
        df = df.iloc[1:]
    for c in ["oi", "nc_long", "nc_short", "c_long", "c_short"]:
        df[c] = pd.to_numeric(df[c].str.replace(",", "").str.strip(), errors="coerce")
    df["date"] = pd.to_datetime(df["date"].str.strip(), errors="coerce")
    df["market"] = df["market"].str.strip()
    return df.dropna(subset=["date", "oi"])


def commodity_positioning(cache_dir: str = ".cache", ttl_hours: float = 24) -> pd.DataFrame:
    """Net speculative (non-commercial) positioning as % of open interest, with its 3-year percentile.
    Crowded longs are a supply of future selling; crowded shorts are fuel for squeezes."""
    cached = cache_get(cache_dir, "cot_positioning", ttl_hours)
    if cached is not None:
        return cached
    frames = []
    year = pd.Timestamp.today().year
    for y in range(year - 3, year + 1):
        try:
            r = requests.get(f"https://www.cftc.gov/files/dea/history/deacot{y}.zip", timeout=30, headers=UA)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                name = next(n for n in z.namelist() if n.lower().endswith(".txt"))
                frames.append(_parse_cot(z.read(name).decode("latin-1")))
        except Exception:  # noqa: BLE001
            continue
    try:
        r = requests.get("https://www.cftc.gov/dea/newcot/deafut.txt", timeout=30, headers=UA)
        r.raise_for_status()
        frames.append(_parse_cot(r.text))
    except Exception:  # noqa: BLE001
        pass
    if not frames:
        return pd.DataFrame()
    hist = pd.concat(frames).drop_duplicates(subset=["market", "date"]).sort_values("date")
    rows = []
    for label, prefix in COT_MARKETS.items():
        sub = hist[hist["market"].str.upper().str.startswith(prefix.upper())]
        if sub.empty:
            continue
        # the same commodity can have several contract codes; keep the one with the largest open interest
        code = sub.groupby("code")["oi"].max().idxmax()
        sub = sub[sub["code"] == code].sort_values("date")
        net = (sub["nc_long"] - sub["nc_short"]) / sub["oi"].replace(0, np.nan) * 100
        if net.dropna().empty:
            continue
        cur = float(net.iloc[-1])
        pct = float((net.dropna() <= cur).mean() * 100)
        wk = float(net.iloc[-1] - net.iloc[-2]) if len(net) > 1 else np.nan
        read = "crowded long" if pct >= 85 else "crowded short" if pct <= 15 else "neutral"
        rows.append({"market": label, "as_of": sub["date"].iloc[-1].date().isoformat(), "net_spec_pct_oi": cur,
                     "pct_3y": pct, "wk_change": wk, "read": read})
    out = pd.DataFrame(rows)
    cache_put(cache_dir, "cot_positioning", out)
    return out
