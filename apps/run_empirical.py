#!/usr/bin/env python3
"""Stable launcher for legacy empirical scripts."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


SCRIPT_MAP = {
    "discovery": "empirical/scripts/discovery.py",
    "analyse": "empirical/scripts/analyse.py",
    "check-freshness": "empirical/scripts/check_delta_sharing_freshness.py",
    "export-window": "empirical/scripts/export_rusd_xrp_window_20251216.py",
    "export-fees-window": "empirical/scripts/export_amm_fees_window_20251216.py",
    "build-input": "empirical/scripts/build_input_from_window.py",
    "inspect-window": "empirical/scripts/inspect_window.py",
    "inspect-hybrid-tx": "empirical/scripts/inspect_hybrid_path_one_tx.py",
    "scan-hybrid": "empirical/scripts/scan_hybrid_candidates_20251201.py",
    "trace-ledger": "empirical/scripts/trace_ledger_rusd_xrp_1215.py",
    "compare-vs-model": "empirical/scripts/compare_rusd_xrp_100894647_vs_model.py",
    "test-share-profile": "empirical/scripts/test_delta_sharing_profile.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a legacy empirical script by alias.")
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
