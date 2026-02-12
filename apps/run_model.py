#!/usr/bin/env python3
"""Stable launcher for model execution scripts."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


SCRIPT_MAP = {
    "amm-only": "apps/model/run_amm_only_from_input.py",
    "hybrid-one-tx": "apps/model/run_hybrid_one_tx_from_windows.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a model script by alias.")
    parser.add_argument("alias", choices=sorted(SCRIPT_MAP.keys()), help="Script alias")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to target script")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / SCRIPT_MAP[args.alias]
    sys.argv = [str(target), *args.script_args]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
