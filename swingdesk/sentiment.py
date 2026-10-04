"""News sentiment: VADER compound score boosted by a finance-specific phrase lexicon, recency-weighted."""
from __future__ import annotations

import numpy as np
import pandas as pd

# phrase → boost (added to compound / 5). VADER alone misreads finance jargon ("beat", "miss", "probe").
PHRASES = {
    "beats": 2.0, "beat estimates": 2.5, "tops estimates": 2.5, "raises guidance": 3.0, "raised guidance": 3.0,
    "upgrade": 2.5, "upgraded": 2.5, "outperform": 2.0, "overweight": 1.5, "record revenue": 2.0, "record profit": 2.0,
    "buyback": 1.5, "dividend increase": 1.5, "all-time high": 1.5, "surge": 1.5, "soars": 2.0, "rally": 1.0,
    "partnership": 1.0, "contract win": 2.0, "approval": 1.5, "etf approval": 2.5, "etf inflows": 2.0,
    "downgrade": -2.5, "downgraded": -2.5, "underperform": -2.0, "misses": -2.5, "missed estimates": -2.5,
    "cuts guidance": -3.0, "lowers guidance": -3.0, "lowered guidance": -3.0, "profit warning": -3.0,
    "probe": -2.0, "investigation": -2.0, "lawsuit": -2.0, "recall": -2.0, "plunge": -2.0, "plunges": -2.5,
    "slump": -2.0, "sell-off": -2.0, "selloff": -2.0, "layoffs": -1.0, "bankruptcy": -3.5, "default": -2.5,
    "fraud": -3.5, "short seller": -2.0, "delisting": -3.0, "hack": -3.0, "exploit": -2.5, "liquidation": -2.0,
    "liquidations": -2.0, "outage": -1.5, "sec charges": -3.0, "fine": -1.0, "tariff": -1.0, "sanction": -1.5,
}

try:  # optional dependency
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

    _SIA = SentimentIntensityAnalyzer()
except Exception:  # noqa: BLE001
    _SIA = None


def score_text(text: str) -> float:
    t = (text or "").lower()
    base = _SIA.polarity_scores(text)["compound"] if _SIA is not None else 0.0
    boost = sum(w for p, w in PHRASES.items() if p in t)
    return float(np.clip(base + boost / 5.0, -1.0, 1.0))


def score_headlines(items: list[dict], now: pd.Timestamp | None = None, half_life_days: float = 3.0) -> dict:
    """Recency-weighted average sentiment of headlines (+ attention = count in last 7 days)."""
    now = now or pd.Timestamp.now(tz="UTC")
    rows = []
    for it in items:
        s = score_text(it.get("title", ""))
        ts = it.get("published")
        if isinstance(ts, pd.Timestamp) and not pd.isna(ts):
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            age = max(0.0, (now - ts).total_seconds() / 86400.0)
        else:
            age = 7.0
        rows.append({**it, "score": s, "age_days": age, "weight": 0.5 ** (age / half_life_days)})
    if not rows:
        return {"news_sent": np.nan, "news_attention": 0, "n": 0, "items": []}
    w = np.array([r["weight"] for r in rows])
    s = np.array([r["score"] for r in rows])
    ws = float(np.average(s, weights=w)) if w.sum() > 0 else float(s.mean())
    rows.sort(key=lambda r: r["age_days"])
    n7 = int(sum(1 for r in rows if r["age_days"] <= 7))
    return {"news_sent": ws, "news_attention": n7, "n": len(rows),
            "items": [{"title": r["title"], "score": round(r["score"], 2), "age_days": round(r["age_days"], 1),
                       "provider": r.get("provider")} for r in rows[:5]]}


def fear_greed_adjustment(fg: dict | None, regime: str) -> tuple[float, str]:
    """Market-level sentiment as a position-size multiplier + note (contrarian at extremes)."""
    if not fg or fg.get("value") is None:
        return 1.0, ""
    v = int(fg["value"])
    if v >= 85:
        return 0.75, f"Fear & Greed {v} (extreme greed): size trimmed 25%, favour pullbacks over chases"
    if v >= 70:
        return 0.9, f"Fear & Greed {v} (greed): slightly reduced size"
    if v <= 15:
        note = f"Fear & Greed {v} (extreme fear): capitulation zone — bounce setups historically strong"
        return (1.0 if regime in ("RISK_ON", "NEUTRAL") else 0.75), note
    if v <= 30:
        return 1.0, f"Fear & Greed {v} (fear): contrarian tailwind for mean-reversion entries"
    return 1.0, f"Fear & Greed {v} ({fg.get('label', 'neutral')})"
