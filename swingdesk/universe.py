"""Market metadata and universe helpers."""
from __future__ import annotations

MARKET_META = {
    "us": {
        "label": "US equities & ETFs",
        "currency": "USD",
        "atr_ideal_pct": 2.5,      # healthy daily range for a 2–4 week swing
        "min_adv_key": "min_adv_us_usd",
        "periods_per_year": 252,
        "fractional": True,
    },
    "uae": {
        "label": "UAE equities (DFM / ADX)",
        "currency": "AED",
        "atr_ideal_pct": 1.8,
        "min_adv_key": "min_adv_uae_aed",
        "periods_per_year": 252,
        "fractional": False,
    },
    "crypto": {
        "label": "Crypto spot",
        "currency": "USD",
        "atr_ideal_pct": 5.0,
        "min_adv_key": "min_adv_crypto_usd",
        "periods_per_year": 365,
        "fractional": True,
    },
}


def get_universe(cfg: dict, market: str) -> list[str]:
    raw = cfg.get("universe", {}).get(market, []) or []
    seen, out = set(), []
    for t in raw:
        t = str(t).strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def aed_per_unit(cfg: dict, market: str) -> float:
    """AED value of one unit of the market's quote currency."""
    return 1.0 if MARKET_META[market]["currency"] == "AED" else float(cfg["aed_per_usd"])


def min_adv(cfg: dict, market: str) -> float:
    return float(cfg.get("liquidity", {}).get(MARKET_META[market]["min_adv_key"], 0.0))
