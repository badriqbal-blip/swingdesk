"""Social mentions & crowd sentiment from free, key-less sources.

- Equities: StockTwits public symbol streams → message volume in the last 24h / 7d and the bullish share of
  tagged messages. (X/Twitter and Reddit no longer expose free APIs; StockTwits is the usable proxy.)
- Crypto: CoinGecko → community sentiment votes, 24h volume change and the trending list.

These are *attention* signals. Attention + price confirmation is useful; attention alone is noise, and extreme
unanimity is a crowding risk — so the factor weights are deliberately small and extremes are flagged, not rewarded.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from .data import UA, cache_get, cache_put

ST_URL = "https://api.stocktwits.com/api/2/streams/symbol/{sym}.json"
CG = "https://api.coingecko.com/api/v3"


def _st_symbol(ticker: str) -> str:
    return ticker.replace("-", ".")           # BRK-B → BRK.B


def stocktwits_symbol(ticker: str, cache_dir: str, ttl_hours: float = 2) -> dict | None:
    key = f"st_{ticker}"
    cached = cache_get(cache_dir, key, ttl_hours)
    if cached is not None:
        return cached or None
    try:
        r = requests.get(ST_URL.format(sym=_st_symbol(ticker)), timeout=10, headers=UA)
        if r.status_code != 200:
            cache_put(cache_dir, key, {})
            return None
        js = r.json()
        msgs = js.get("messages", []) or []
        now = pd.Timestamp.now(tz="UTC")
        n24 = n7 = bull = bear = 0
        for m in msgs:
            ts = pd.to_datetime(m.get("created_at"), utc=True, errors="coerce")
            age_h = (now - ts).total_seconds() / 3600 if pd.notna(ts) else 999
            if age_h <= 24:
                n24 += 1
            if age_h <= 24 * 7:
                n7 += 1
            s = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
            if s == "Bullish":
                bull += 1
            elif s == "Bearish":
                bear += 1
        oldest = min((pd.to_datetime(m.get("created_at"), utc=True, errors="coerce") for m in msgs), default=pd.NaT)
        span_h = (now - oldest).total_seconds() / 3600 if pd.notna(oldest) else np.nan
        rate = len(msgs) / span_h * 24 if span_h and span_h > 0 else np.nan     # msgs/day implied by the last 30 messages
        out = {"messages_24h": n24, "messages_7d": n7, "msgs_per_day": rate, "bullish": bull, "bearish": bear,
               "bull_share": bull / (bull + bear) if (bull + bear) >= 3 else np.nan,
               "watchers": (js.get("symbol") or {}).get("watchlist_count")}
        cache_put(cache_dir, key, out)
        return out
    except Exception:  # noqa: BLE001
        return None


def coingecko_lookup(symbols: list[str], cache_dir: str) -> dict[str, str]:
    """BTC/USDT → coingecko id via the coins list (cached 7 days)."""
    lst = cache_get(cache_dir, "cg_coins_list", 24 * 7)
    if lst is None:
        try:
            r = requests.get(f"{CG}/coins/list", timeout=20, headers=UA)
            r.raise_for_status()
            lst = r.json()
            cache_put(cache_dir, "cg_coins_list", lst)
        except Exception:  # noqa: BLE001
            return {}
    prefer = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "BNB": "binancecoin", "XRP": "ripple", "ADA": "cardano",
              "DOGE": "dogecoin", "AVAX": "avalanche-2", "LINK": "chainlink", "DOT": "polkadot", "TRX": "tron", "LTC": "litecoin",
              "BCH": "bitcoin-cash", "NEAR": "near", "SUI": "sui", "APT": "aptos", "ARB": "arbitrum", "OP": "optimism",
              "INJ": "injective-protocol", "TON": "the-open-network", "HBAR": "hedera-hashgraph", "UNI": "uniswap", "AAVE": "aave",
              "ATOM": "cosmos", "FIL": "filecoin", "RENDER": "render-token", "TAO": "bittensor", "FET": "fetch-ai", "ONDO": "ondo-finance",
              "ENA": "ethena"}
    by_sym: dict[str, list[str]] = {}
    for c in lst:
        by_sym.setdefault(str(c.get("symbol", "")).upper(), []).append(c.get("id"))
    out = {}
    for s in symbols:
        base = s.split("/")[0].upper()
        if base in prefer:
            out[s] = prefer[base]
        elif by_sym.get(base):
            out[s] = sorted(by_sym[base], key=len)[0]
    return out


def coingecko_sentiment(symbols: list[str], cache_dir: str, ttl_hours: float = 3) -> dict[str, dict]:
    ids = coingecko_lookup(symbols, cache_dir)
    out: dict[str, dict] = {}
    trending = set()
    try:
        r = requests.get(f"{CG}/search/trending", timeout=15, headers=UA)
        if r.status_code == 200:
            trending = {str(c["item"]["symbol"]).upper() for c in r.json().get("coins", []) if "item" in c}
    except Exception:  # noqa: BLE001
        pass
    for s, cid in ids.items():
        key = f"cg_{cid}"
        d = cache_get(cache_dir, key, ttl_hours)
        if d is None:
            try:
                r = requests.get(f"{CG}/coins/{cid}", params={"localization": "false", "tickers": "false", "market_data": "true",
                                                              "community_data": "true", "developer_data": "false", "sparkline": "false"},
                                 timeout=15, headers=UA)
                if r.status_code == 429:
                    time.sleep(8)
                    r = requests.get(f"{CG}/coins/{cid}", params={"localization": "false", "tickers": "false", "market_data": "true",
                                                                  "community_data": "true", "developer_data": "false"}, timeout=15, headers=UA)
                if r.status_code != 200:
                    continue
                js = r.json()
                md, cd = js.get("market_data") or {}, js.get("community_data") or {}
                d = {"sentiment_up_pct": js.get("sentiment_votes_up_percentage"),
                     "vol_change_24h_pct": None, "price_change_24h_pct": md.get("price_change_percentage_24h"),
                     "reddit_posts_48h": cd.get("reddit_average_posts_48h"), "reddit_comments_48h": cd.get("reddit_average_comments_48h"),
                     "twitter_followers": cd.get("twitter_followers"), "watchlist_users": js.get("watchlist_portfolio_users")}
                cache_put(cache_dir, key, d)
                time.sleep(1.5)
            except Exception:  # noqa: BLE001
                continue
        d = dict(d)
        d["trending"] = s.split("/")[0].upper() in trending
        out[s] = d
    return out


def social_frame(market: str, tickers: list[str], cache_dir: str, demo: bool = False) -> pd.DataFrame:
    """Per-ticker social factors: social_buzz (log messages/day), social_bull (bullish share 0–1), plus raw fields."""
    rows = {}
    if demo:
        rng = np.random.default_rng(len(tickers))
        for t in tickers:
            rows[t] = {"social_buzz": float(rng.uniform(0, 4)), "social_bull": float(rng.uniform(0.3, 0.9)), "source": "demo"}
        return pd.DataFrame.from_dict(rows, orient="index")
    if market == "us":
        for t in tickers:
            d = stocktwits_symbol(t, cache_dir)
            if d:
                rows[t] = {"social_buzz": float(np.log1p(d["msgs_per_day"])) if d.get("msgs_per_day") and np.isfinite(d["msgs_per_day"]) else np.nan,
                           "social_bull": d.get("bull_share", np.nan), "messages_24h": d["messages_24h"], "watchers": d.get("watchers"), "source": "StockTwits"}
    elif market == "crypto":
        for s, d in coingecko_sentiment(tickers, cache_dir).items():
            up = d.get("sentiment_up_pct")
            rows[s] = {"social_buzz": float(np.log1p(d.get("reddit_posts_48h") or 0)) + (1.0 if d.get("trending") else 0.0),
                       "social_bull": float(up) / 100.0 if up is not None else np.nan, "trending": d.get("trending"), "source": "CoinGecko"}
    return pd.DataFrame.from_dict(rows, orient="index") if rows else pd.DataFrame()


def social_flags(row: pd.Series) -> list[str]:
    out = []
    bull, buzz = row.get("social_bull"), row.get("social_buzz")
    try:
        if bull is not None and np.isfinite(bull) and bull >= 0.9 and buzz is not None and np.isfinite(buzz) and buzz >= 3.5:
            out.append("crowd euphoric on social (contrarian caution)")
        elif bull is not None and np.isfinite(bull) and bull <= 0.3:
            out.append("crowd bearish on social")
        if row.get("trending") is True:
            out.append("trending on CoinGecko")
    except TypeError:
        pass
    return out
