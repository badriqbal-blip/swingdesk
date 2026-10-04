"""Data layer: OHLCV (yfinance / ccxt), fundamentals, earnings dates, news, sentiment gauges, demo data.

Everything network-facing is wrapped so a failing source degrades gracefully instead of killing a scan.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

MIN_BARS = 120
UA = {"User-Agent": "Mozilla/5.0 (SwingDesk research tool)"}


# ───────────────────────────────────────────── cache ─────────────────────────────────────────────
def _ckey(key: str) -> str:
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def cache_get(cache_dir: str, key: str, ttl_hours: float | None):
    p = Path(cache_dir) / f"{_ckey(key)}.pkl"
    if not p.exists():
        return None
    if ttl_hours is not None and time.time() - p.stat().st_mtime > ttl_hours * 3600:
        return None
    try:
        return pickle.loads(p.read_bytes())
    except Exception:
        return None


def cache_put(cache_dir: str, key: str, obj) -> None:
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / f"{_ckey(key)}.pkl").write_bytes(pickle.dumps(obj))
    except Exception:
        pass


# ───────────────────────────────────────── OHLCV helpers ─────────────────────────────────────────
def _norm_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).title() for c in df.columns]
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep]
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df = df.dropna(subset=["Close"])
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    for c in ["Open", "High", "Low"]:
        if c not in df.columns:
            df[c] = df["Close"]
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _split_download(raw: pd.DataFrame, chunk: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if raw is None or len(raw) == 0:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = set(raw.columns.get_level_values(0))
        lv1 = set(raw.columns.get_level_values(1))
        for t in chunk:
            try:
                if t in lv0:
                    sub = raw[t]
                elif t in lv1:
                    sub = raw.xs(t, axis=1, level=1)
                else:
                    continue
            except Exception:
                continue
            sub = sub.dropna(how="all")
            if len(sub):
                out[t] = _norm_ohlcv(sub)
    elif len(chunk) == 1:
        out[chunk[0]] = _norm_ohlcv(raw.dropna(how="all"))
    return out


def fetch_equity_ohlcv(tickers: list[str], years: int = 3, cache_dir: str = ".cache",
                       ttl_hours: float = 12, chunk_size: int = 40, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Daily adjusted OHLCV from Yahoo Finance (via yfinance) for a list of tickers."""
    import yfinance as yf

    key = f"eq_{years}_{dt.date.today()}_{'_'.join(sorted(tickers))}"
    cached = cache_get(cache_dir, key, ttl_hours)
    if cached is not None:
        return cached
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            raw = yf.download(chunk, period=f"{years}y", interval="1d", auto_adjust=True,
                              group_by="ticker", threads=True, progress=False)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  ! download error for chunk starting {chunk[0]}: {e}")
            continue
        out.update(_split_download(raw, chunk))
    out = {t: d for t, d in out.items() if len(d) >= MIN_BARS}
    cache_put(cache_dir, key, out)
    return out


def _crypto_via_yf(symbols: list[str], years: int, cache_dir: str, ttl_hours: float) -> dict[str, pd.DataFrame]:
    yf_map = {s: f"{s.split('/')[0]}-USD" for s in symbols}
    data = fetch_equity_ohlcv(list(yf_map.values()), years, cache_dir, ttl_hours, verbose=False)
    return {s: data[y] for s, y in yf_map.items() if y in data}


def fetch_crypto_ohlcv(symbols: list[str], years: int, exchange_id: str, fallbacks: list[str],
                       cache_dir: str = ".cache", ttl_hours: float = 12, verbose: bool = True):
    """Daily spot OHLCV via ccxt public endpoints (no API key). Falls back to Yahoo (BTC-USD style)."""
    key = f"cx_{exchange_id}_{years}_{dt.date.today()}_{'_'.join(sorted(symbols))}"
    cached = cache_get(cache_dir, key, ttl_hours)
    if cached is not None:
        return cached
    try:
        import ccxt  # type: ignore
    except ImportError:
        if verbose:
            print("  ccxt not installed → using Yahoo Finance for crypto (pip install ccxt for exchange data)")
        res = (_crypto_via_yf(symbols, years, cache_dir, ttl_hours), "yfinance")
        cache_put(cache_dir, key, res)
        return res

    since = int((time.time() - years * 365 * 86400) * 1000)
    for exid in [exchange_id] + [f for f in fallbacks if f != exchange_id]:
        try:
            ex = getattr(ccxt, exid)({"enableRateLimit": True})
            ex.load_markets()
            out: dict[str, pd.DataFrame] = {}
            for s in symbols:
                sym = s if s in ex.markets else s.replace("/USDT", "/USD")
                if sym not in ex.markets:
                    continue
                rows, cursor, loops = [], since, 0
                while loops < 25:
                    loops += 1
                    batch = ex.fetch_ohlcv(sym, timeframe="1d", since=cursor, limit=1000)
                    if not batch:
                        break
                    rows.extend(batch)
                    nxt = batch[-1][0] + 86_400_000
                    if nxt <= cursor or len(batch) < 2 or nxt > time.time() * 1000:
                        break
                    cursor = nxt
                if rows:
                    df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
                    df.index = pd.to_datetime(df.pop("ts"), unit="ms")
                    out[s] = _norm_ohlcv(df)
            out = {t: d for t, d in out.items() if len(d) >= MIN_BARS}
            if out:
                res = (out, exid)
                cache_put(cache_dir, key, res)
                return res
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  ! {exid} failed ({type(e).__name__}): trying next source")
            continue
    res = (_crypto_via_yf(symbols, years, cache_dir, ttl_hours), "yfinance")
    cache_put(cache_dir, key, res)
    return res


def equal_weight_index(prices: dict[str, pd.DataFrame]) -> pd.Series:
    rets = pd.DataFrame({t: d["Close"].pct_change() for t, d in prices.items()})
    ew = rets.mean(axis=1, skipna=True).fillna(0.0)
    return 100.0 * (1.0 + ew).cumprod()


# ──────────────────────────────────────── fundamentals ───────────────────────────────────────────
FUND_FIELDS = [
    "marketCap", "trailingPE", "forwardPE", "pegRatio", "priceToBook", "enterpriseToEbitda",
    "returnOnEquity", "grossMargins", "operatingMargins", "profitMargins", "revenueGrowth",
    "earningsGrowth", "debtToEquity", "freeCashflow", "totalRevenue", "shortPercentOfFloat",
    "shortRatio", "recommendationMean", "recommendationKey", "targetMeanPrice",
    "numberOfAnalystOpinions", "heldPercentInstitutions", "sector", "industry", "dividendYield",
    "beta", "quoteType", "shortName",
]


def fetch_fundamentals(tickers: list[str], cache_dir: str, ttl_hours: float = 24, verbose: bool = True) -> pd.DataFrame:
    import yfinance as yf

    rows = {}
    for i, t in enumerate(tickers):
        key = f"fund_{t}"
        info = cache_get(cache_dir, key, ttl_hours)
        if info is None:
            try:
                info = yf.Ticker(t).info or {}
            except Exception:  # noqa: BLE001
                info = {}
            cache_put(cache_dir, key, info)
        row = {k: info.get(k) for k in FUND_FIELDS}
        rev = cache_get(cache_dir, f"epsrev_{t}", ttl_hours)
        if rev is None:
            rev = _eps_revision(t)
            cache_put(cache_dir, f"epsrev_{t}", rev if rev is not None else "none")
        row["epsRevision30d"] = rev if rev != "none" else None
        rows[t] = row
        if verbose and (i + 1) % 10 == 0:
            print(f"    fundamentals {i + 1}/{len(tickers)}")
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df


def _eps_revision(ticker: str):
    """30-day change in the consensus next-year EPS estimate (positive = analysts raising numbers)."""
    import yfinance as yf

    try:
        tr = yf.Ticker(ticker).eps_trend
        if tr is None or len(tr) == 0:
            return None
        cols = {str(c).lower(): c for c in tr.columns}
        cur, old = cols.get("current"), cols.get("30daysago")
        if cur is None or old is None:
            return None
        for period in ["+1y", "0y", "+1q"]:
            if period in tr.index:
                c, o = tr.loc[period, cur], tr.loc[period, old]
                if pd.notna(c) and pd.notna(o) and float(o) != 0:
                    return float(c) / float(o) - 1.0
    except Exception:  # noqa: BLE001
        return None
    return None


def fetch_exdividend(ticker: str, cache_dir: str, ttl_hours: float = 24):
    import yfinance as yf

    key = f"exdiv_{ticker}"
    cached = cache_get(cache_dir, key, ttl_hours)
    if cached is not None:
        return cached if cached != "none" else None
    result = None
    try:
        cal = yf.Ticker(ticker).calendar
        if isinstance(cal, dict):
            ed = cal.get("Ex-Dividend Date")
            if ed:
                result = pd.Timestamp(ed).date()
    except Exception:  # noqa: BLE001
        pass
    cache_put(cache_dir, key, result if result is not None else "none")
    return result


def fetch_next_earnings(ticker: str, cache_dir: str, ttl_hours: float = 24):
    """Next earnings date (date) or None."""
    import yfinance as yf

    key = f"earn_{ticker}"
    cached = cache_get(cache_dir, key, ttl_hours)
    if cached is not None:
        return cached if cached != "none" else None
    today = pd.Timestamp.today().normalize()
    result = None
    tk = yf.Ticker(ticker)
    try:
        cal = tk.calendar
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed:
                ed = ed if isinstance(ed, (list, tuple)) else [ed]
                fut = [pd.Timestamp(d) for d in ed if pd.Timestamp(d) >= today]
                if fut:
                    result = min(fut).date()
        elif isinstance(cal, pd.DataFrame) and not cal.empty and "Earnings Date" in cal.index:
            fut = [pd.Timestamp(d) for d in cal.loc["Earnings Date"].dropna().tolist() if pd.Timestamp(d) >= today]
            if fut:
                result = min(fut).date()
    except Exception:  # noqa: BLE001
        pass
    if result is None:
        try:
            ed = tk.get_earnings_dates(limit=8)
            if ed is not None and len(ed):
                idx = pd.to_datetime(ed.index)
                if getattr(idx, "tz", None) is not None:
                    idx = idx.tz_localize(None)
                fut = idx[idx >= today]
                if len(fut):
                    result = fut.min().date()
        except Exception:  # noqa: BLE001
            pass
    cache_put(cache_dir, key, result if result is not None else "none")
    return result


# ─────────────────────────────────────────── news ────────────────────────────────────────────────
def fetch_news(ticker: str, max_items: int = 25) -> list[dict]:
    import yfinance as yf

    items: list[dict] = []
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:  # noqa: BLE001
        return items
    for n in raw[:max_items]:
        if not isinstance(n, dict):
            continue
        c = n.get("content")
        if isinstance(c, dict):
            title = c.get("title")
            pub = c.get("pubDate") or c.get("displayTime")
            ts = pd.to_datetime(pub, utc=True, errors="coerce") if pub else pd.NaT
            prov = (c.get("provider") or {}).get("displayName")
            summary = c.get("summary")
        else:
            title = n.get("title")
            ts = pd.to_datetime(n.get("providerPublishTime"), unit="s", utc=True, errors="coerce")
            prov = n.get("publisher")
            summary = None
        if title:
            items.append({"title": str(title), "published": ts, "provider": prov, "summary": summary})
    return items


# ──────────────────────────────────── market sentiment gauges ────────────────────────────────────
def fetch_crypto_fear_greed() -> dict | None:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=8", timeout=10, headers=UA)
        r.raise_for_status()
        data = r.json()["data"]
        latest = data[0]
        return {"value": int(latest["value"]), "label": latest["value_classification"],
                "history": [int(d["value"]) for d in data], "source": "alternative.me"}
    except Exception:  # noqa: BLE001
        return None


def fetch_cnn_fear_greed() -> dict | None:
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        fg = r.json()["fear_and_greed"]
        return {"value": int(round(float(fg["score"]))), "label": str(fg.get("rating", "")).title(),
                "prev_1w": fg.get("previous_1_week"), "source": "CNN"}
    except Exception:  # noqa: BLE001
        return None


def fetch_funding_rates(symbols: list[str]) -> dict[str, float]:
    """Perpetual funding rates (sentiment/crowding gauge) from Binance USDⓈ-M; spot trading unaffected."""
    out: dict[str, float] = {}
    try:
        import ccxt  # type: ignore

        ex = ccxt.binanceusdm({"enableRateLimit": True})
        ex.load_markets()
        for s in symbols:
            sym = s if ":" in s else f"{s}:USDT"
            if sym in ex.markets:
                try:
                    fr = ex.fetch_funding_rate(sym)
                    if fr and fr.get("fundingRate") is not None:
                        out[s] = float(fr["fundingRate"])
                except Exception:  # noqa: BLE001
                    continue
    except Exception:  # noqa: BLE001
        pass
    return out


# ─────────────────────────────────────── live prices & bars ──────────────────────────────────────
def fetch_live_prices(market: str, tickers: list[str], exchange_id: str = "binance", fallbacks: list[str] | None = None) -> dict[str, float]:
    """Last traded price per ticker (delayed quotes are fine for sizing). Missing names are simply absent."""
    out: dict[str, float] = {}
    if not tickers:
        return out
    if market == "crypto":
        try:
            import ccxt  # type: ignore

            for exid in [exchange_id] + [f for f in (fallbacks or []) if f != exchange_id]:
                try:
                    ex = getattr(ccxt, exid)({"enableRateLimit": True})
                    ex.load_markets()
                    for s in tickers:
                        sym = s if s in ex.markets else s.replace("/USDT", "/USD")
                        if sym in ex.markets:
                            tk = ex.fetch_ticker(sym)
                            if tk and tk.get("last"):
                                out[s] = float(tk["last"])
                    if out:
                        return out
                except Exception:  # noqa: BLE001
                    continue
        except ImportError:
            pass
        yf_syms = {s: f"{s.split('/')[0]}-USD" for s in tickers}
    else:
        yf_syms = {s: s for s in tickers}
    try:
        import yfinance as yf

        for s, y in yf_syms.items():
            try:
                fi = yf.Ticker(y).fast_info
                px = None
                for key in ("last_price", "lastPrice", "regular_market_price"):
                    try:
                        px = fi[key]
                        if px:
                            break
                    except Exception:  # noqa: BLE001
                        continue
                if px and float(px) > 0:
                    out[s] = float(px)
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return out


def drop_partial_bar(prices: dict[str, pd.DataFrame], market: str, now: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    """Signals must use completed bars. Drop today's bar while its session is still open
    (US 13:30–21:00 UTC, UAE 06:00–11:00 UTC Mon–Fri; crypto's UTC day bar is partial until midnight)."""
    now = now or pd.Timestamp.now(tz="UTC")
    today = now.normalize().tz_localize(None)
    hhmm = now.hour * 60 + now.minute
    if market == "crypto":
        partial = True
    elif market == "us":
        partial = now.weekday() < 5 and 13 * 60 + 30 <= hhmm < 21 * 60
    else:
        partial = now.weekday() < 5 and 6 * 60 <= hhmm < 11 * 60
    if not partial:
        return prices
    out = {}
    for t, df in prices.items():
        out[t] = df.iloc[:-1] if len(df) and df.index[-1] >= today else df
    return out


def load_holdings(path: str) -> list[dict]:
    """holdings.yaml → list of {market, ticker, units, entry, stop?}. Missing file → []."""
    import yaml

    p = Path(path)
    if not p.exists():
        return []
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return []
    rows = data.get("holdings", data) if isinstance(data, dict) else data
    return [r for r in (rows or []) if isinstance(r, dict) and r.get("ticker")]


# ────────────────────────────────────────── demo data ────────────────────────────────────────────
def demo_prices(tickers: list[str], days: int = 800, seed: int = 7, vol: float = 0.02,
                level: float = 100.0, daily: bool = False) -> dict[str, pd.DataFrame]:
    """Synthetic regime-switching GBM prices so the whole pipeline can run offline."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    idx = pd.date_range(end=end, periods=days, freq="D") if daily else pd.bdate_range(end=end, periods=days)
    days = len(idx)
    out = {}
    for t in tickers:
        v = vol * rng.uniform(0.6, 1.6)
        drift = rng.normal(0.0004, 0.0005)
        states = np.cumsum(rng.random(days) < 0.012) % 3
        mu = np.where(states == 0, drift + 0.0018, np.where(states == 1, drift - 0.0012, drift))
        r = rng.normal(mu, v)
        close = level * rng.uniform(0.3, 3.0) * np.exp(np.cumsum(r))
        o = close * (1 + rng.normal(0, v / 3, days))
        hi = np.maximum(o, close) * (1 + np.abs(rng.normal(0, v / 2, days)))
        lo = np.minimum(o, close) * (1 - np.abs(rng.normal(0, v / 2, days)))
        volume = rng.lognormal(14, 0.5, days) * (1 + 4 * np.abs(r) / v)
        out[t] = pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": close, "Volume": volume}, index=idx)
    return out


def demo_news(ticker: str) -> list[dict]:
    now = pd.Timestamp.now(tz="UTC")
    pool = [
        (f"{ticker} beats estimates, raises full-year guidance", 0.5),
        (f"Analyst upgrades {ticker} to outperform on margin expansion", 1.5),
        (f"{ticker} faces regulatory probe over pricing practices", 2.5),
        (f"{ticker} announces $2bn buyback", 4.0),
        (f"Sector rotation: investors trim {ticker} after record run", 6.0),
    ]
    k = (sum(ord(c) for c in ticker) % 3) + 2
    return [{"title": t, "published": now - pd.Timedelta(days=d), "provider": "demo", "summary": None} for t, d in pool[:k]]
