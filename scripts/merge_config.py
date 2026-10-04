"""Merge a user's existing config.yaml with a newer default: keep every value the user has, append any new
top-level sections (with their comments) from the default, and list new nested keys as comments.
Usage: python scripts/merge_config.py OLD_CONFIG NEW_DEFAULT  → writes the result to NEW_DEFAULT's path."""
from __future__ import annotations

import re
import sys
from pathlib import Path


def _blocks(text: str) -> dict[str, str]:
    """Top-level key → its text block (including the comment lines directly above it)."""
    lines = text.splitlines(keepends=True)
    starts = [i for i, l in enumerate(lines) if re.match(r"^[A-Za-z_][\w-]*:", l)]
    out = {}
    for n, i in enumerate(starts):
        j = starts[n + 1] if n + 1 < len(starts) else len(lines)
        while j - 1 > i and lines[j - 1].lstrip().startswith("#"):   # comments above the next key belong to it
            j -= 1
        k = i
        while k > 0 and lines[k - 1].lstrip().startswith("#"):
            k -= 1
        key = lines[i].split(":", 1)[0].strip()
        out[key] = "".join(lines[k:j])
    return out


def _missing_nested(old: dict, new: dict, prefix: str = "") -> list[str]:
    out = []
    for k, v in (new or {}).items():
        if not isinstance(old, dict) or k not in old:
            continue
        if isinstance(v, dict) and isinstance(old.get(k), dict):
            for kk in v:
                if kk not in old[k]:
                    out.append(f"{prefix}{k}.{kk}")
            out += _missing_nested(old[k], v, f"{prefix}{k}.")
    return out


def main(old_path: str, new_path: str) -> None:
    old_text = Path(old_path).read_text(encoding="utf-8")
    new_text = Path(new_path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        old = yaml.safe_load(old_text) or {}
        new = yaml.safe_load(new_text) or {}
    except Exception:  # noqa: BLE001
        Path(new_path).write_text(old_text, encoding="utf-8")
        return
    blocks = _blocks(new_text)
    added = [k for k in new if k not in old and k in blocks]
    merged = old_text.rstrip("\n") + "\n"
    if added:
        merged += "\n# ── added by SwingDesk update (new options with their defaults) ──\n" + "".join(blocks[k] for k in added)
    nested = _missing_nested(old, new)
    if nested:
        merged += "\n# New nested options available (defaults apply until you add them): " + ", ".join(nested) + "\n"
    Path(new_path).write_text(merged, encoding="utf-8")
    print(f"config merged: kept your values; added sections {added or 'none'}; new nested keys {nested or 'none'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
