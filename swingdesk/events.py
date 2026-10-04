"""Event risk: earnings dates inside the holding window and scheduled macro events."""
from __future__ import annotations

import pandas as pd


def earnings_assessment(next_date, today: pd.Timestamp, window_days: int, policy: str) -> dict:
    base = {"earnings_date": None, "days_to_earnings": None, "flag": "", "size_mult": 1.0, "veto": False}
    if next_date is None:
        return base
    d = pd.Timestamp(next_date).normalize()
    days = int((d - today).days)
    base["earnings_date"] = str(d.date())
    base["days_to_earnings"] = days
    if days < 0 or days > window_days:
        return base
    flag = f"Earnings in {days}d"
    if policy == "halve":
        base["size_mult"] = 0.5
        flag += " (half size)"
    elif policy == "veto":
        base["veto"] = True
        flag += " (vetoed)"
    else:
        flag += " — binary gap risk"
    base["flag"] = flag
    return base


def macro_warnings(macro_dates: list[dict] | None, today: pd.Timestamp, horizon_days: int) -> list[str]:
    out = []
    for m in macro_dates or []:
        try:
            d = pd.Timestamp(m["date"]).normalize()
        except Exception:  # noqa: BLE001
            continue
        days = int((d - today).days)
        if 0 <= days <= horizon_days:
            out.append(f"{m.get('name', 'Macro event')} in {days}d ({d.date()})")
    return out
