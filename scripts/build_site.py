"""Collect the latest scan outputs into ./site for GitHub Pages (index.html = latest briefing)."""
from __future__ import annotations

import shutil
from pathlib import Path

out, site = Path("output"), Path("site")
site.mkdir(exist_ok=True)
briefings = sorted(out.glob("briefing_*.html"))
if not briefings:
    (site / "index.html").write_text("<p>No briefing was produced. Check the workflow log.</p>", encoding="utf-8")
    raise SystemExit(0)
latest = briefings[-1]
shutil.copy(latest, site / "index.html")
assets = Path("assets")
if assets.exists():
    for p in assets.iterdir():
        shutil.copy(p, site / p.name)          # manifest + icons for "Add to Home Screen"
for p in list(out.glob("signals_*.csv")) + list(out.glob("summary_*.json")) + list(out.glob("trades_*.csv")) + list(out.glob("plan_*.csv")):
    shutil.copy(p, site / p.name)
links = "".join(f'<li><a href="{p.name}">{p.name}</a></li>' for p in sorted(site.iterdir()) if p.name != "index.html")
(site / "files.html").write_text(f"<!doctype html><meta charset='utf-8'><title>SwingDesk files</title>"
                                 f"<h1>Latest run files</h1><ul>{links}</ul><p><a href='index.html'>Back to the briefing</a></p>",
                                 encoding="utf-8")
print(f"site built from {latest.name}")
