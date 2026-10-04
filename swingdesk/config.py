"""Configuration loading."""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else DEFAULT_PATH
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_path"] = str(p)
    cfg.setdefault("cache_dir", ".cache")
    cfg.setdefault("cache_ttl_hours", 12)
    cfg.setdefault("fundamentals_ttl_hours", 24)
    cfg.setdefault("scan", {})
    cfg["scan"].setdefault("fundamentals_top", 40)
    cfg["scan"].setdefault("news_top", 25)
    Path(cfg["cache_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def capital_usd(cfg: dict) -> float:
    return float(cfg["capital_aed"]) / float(cfg["aed_per_usd"])
