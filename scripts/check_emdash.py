#!/usr/bin/env python3
"""Enforce ADR GLOBAL-003: no em dashes in repository files.

The em dash (U+2014) is treated as a visible AI writing pattern in this
organisation, so it is banned from committed text. The en dash (U+2013) and the
horizontal bar (U+2015) are checked too, since they are the usual substitutions
people reach for once the em dash is blocked.

Usage:
    python scripts/check_emdash.py            # check tracked text files
    python scripts/check_emdash.py path ...   # check specific paths

Exit code 1 means at least one violation was found.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Defined as escapes on purpose: this file must not contain the characters it
# forbids, or it would fail its own check when passed explicitly by the hook.
BANNED = {
    "\u2014": "em dash",
    "\u2013": "en dash",
    "\u2015": "horizontal bar",
}

CHECKED_SUFFIXES = {
    ".md", ".py", ".txt", ".yml", ".yaml", ".json", ".toml", ".cfg", ".ini",
    ".html", ".css", ".js", ".ts", ".tsx", ".jsx", ".sh",
}

SKIP_PARTS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".ruff_cache",
              ".pytest_cache", "dist", "build"}


def tracked_files() -> list[Path]:
    """Return git tracked files, or fall back to walking the tree."""
    try:
        output = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True
        ).stdout
        return [Path(line) for line in output.splitlines() if line]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [p for p in Path(".").rglob("*") if p.is_file()]


def should_check(path: Path) -> bool:
    if any(part in SKIP_PARTS for part in path.parts):
        return False
    return path.suffix.lower() in CHECKED_SUFFIXES


def scan(path: Path) -> list[tuple[int, int, str, str]]:
    """Return (line number, column, character name, line text) for each violation."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    findings = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for character, name in BANNED.items():
            column = line.find(character)
            if column >= 0:
                findings.append((line_number, column + 1, name, line.strip()))
    return findings


def main(argv: list[str]) -> int:
    if argv:
        paths = [Path(arg) for arg in argv]
    else:
        paths = [p for p in tracked_files() if should_check(p)]

    violations = 0
    for path in paths:
        if not path.is_file():
            continue
        for line_number, column, name, line in scan(path):
            violations += 1
            print(f"{path}:{line_number}:{column}: {name} found")
            print(f"    {line[:110]}")

    if violations:
        print(f"\n{violations} violation(s). ADR GLOBAL-003 forbids these characters.")
        print("Replace with a comma, colon, period, or restructure the sentence.")
        return 1

    print(f"No banned dash characters found in {len(paths)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
