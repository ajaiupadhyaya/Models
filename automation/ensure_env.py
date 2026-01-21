#!/usr/bin/env python3
"""
Ensure .env is present and required keys are set.
Non-interactive mode prints missing keys and exits non-zero.
Interactive mode prompts and updates .env in place.
"""

import os
import sys
from pathlib import Path
from typing import Dict

REQUIRED_KEYS = [
    "FRED_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "OPENAI_API_KEY",
]

OPTIONAL_KEYS = [
    "ENABLE_PAPER_TRADING",
    "ALPACA_API_KEY",
    "ALPACA_API_SECRET",
    "ALPACA_API_BASE",
]


def load_env(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    return values


def save_env(path: Path, values: Dict[str, str]) -> None:
    lines = ["# Managed by ensure_env.py", ""]
    for k, v in values.items():
        lines.append(f"{k}={v}")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    interactive = "--interactive" in sys.argv
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"

    values = load_env(env_path)
    changed = False

    missing = [k for k in REQUIRED_KEYS if not values.get(k)]
    if missing and not interactive:
        print("Missing required environment keys:\n  - " + "\n  - ".join(missing))
        print("\nAdd them to .env or run: python automation/ensure_env.py --interactive")
        return 1

    if interactive:
        for k in REQUIRED_KEYS + OPTIONAL_KEYS:
            current = values.get(k, "")
            prompt = f"Enter {k}{' (optional)' if k in OPTIONAL_KEYS else ''} [{current}]: "
            try:
                new = input(prompt).strip()
            except EOFError:
                new = ""
            if new:
                values[k] = new
                changed = True

        if changed:
            save_env(env_path, values)
            print(f"Updated {env_path}")
        else:
            print("No changes made to .env")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
