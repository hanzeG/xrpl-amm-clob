#!/usr/bin/env python3
"""Stable launcher for empirical scripts."""

from __future__ import annotations

import argparse
import json
import runpy
import stat
import sys
from datetime import datetime, timezone
import os
from pathlib import Path


SCRIPT_MAP = {
    "delta-sharing-check-freshness": "empirical/scripts/delta_sharing_check_freshness.py",
    "delta-sharing-test-profile": "empirical/scripts/delta_sharing_test_profile.py",
    "empirical-download-clob-offers-range": "empirical/scripts/empirical_download_clob_offers_range.py",
    "empirical-export-window": "empirical/scripts/empirical_export_window.py",
    "empirical-build-model-input": "empirical/scripts/empirical_build_model_input.py",
    "empirical-compare-rolling": "empirical/scripts/empirical_compare_rolling.py",
    "empirical-compare-single": "empirical/scripts/empirical_compare_single.py",
    "empirical-analyze-traces": "empirical/scripts/empirical_analyze_traces.py",
    "empirical-enrich-clob-with-tx-index": "empirical/scripts/empirical_enrich_clob_with_tx_index.py",
    "empirical-check-parquet-bounds": "empirical/scripts/empirical_check_parquet_bounds.py",
    "empirical-reconstruct-real-paths": "empirical/scripts/empirical_reconstruct_real_paths.py",
}

LEGACY_ALIAS_MAP = {
    "check-freshness": "delta-sharing-check-freshness",
    "test-share-profile": "delta-sharing-test-profile",
    "download-clob-offers-range": "empirical-download-clob-offers-range",
    "pipeline-export-window": "empirical-export-window",
    "pipeline-build-model-input": "empirical-build-model-input",
    "research-compare-rolling": "empirical-compare-rolling",
    "research-compare-single": "empirical-compare-single",
    "research-analyse-traces": "empirical-analyze-traces",
    "research-enrich-clob": "empirical-enrich-clob-with-tx-index",
    "research-check-parquet": "empirical-check-parquet-bounds",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an empirical script by alias.")
    parser.add_argument("alias", help="Script alias")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to target script")
    return parser.parse_args()


def _strip_double_dash(xs: list[str]) -> list[str]:
    if xs and xs[0] == "--":
        return xs[1:]
    return xs


def _run_script(root: Path, rel_path: str, script_args: list[str]) -> None:
    target = root / rel_path
    sys.argv = [str(target), *_strip_double_dash(script_args)]
    runpy.run_path(str(target), run_name="__main__")


def _parse_pipeline_args(raw: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-shot empirical pipeline runner.")
    p.add_argument("--config", default="configs/empirical/pipeline_run.default.json")
    p.add_argument("--pair", default=None)
    p.add_argument("--ledger-start", type=int, default=None)
    p.add_argument("--ledger-end", type=int, default=None)
    p.add_argument("--share-profile", default=None)
    p.add_argument("--share", default=None)
    p.add_argument("--schema", default=None)
    p.add_argument("--table-amm", default=None)
    p.add_argument("--table-clob", default=None)
    p.add_argument("--table-fees", default=None)
    p.add_argument("--book-gets-xrp", default=None, help="Required for compare step unless --skip-compare")
    p.add_argument("--book-gets-rusd", default=None, help="Required for compare step unless --skip-compare")
    p.add_argument("--exports-dir", default=None)
    p.add_argument("--model-input-dir", default=None)
    p.add_argument("--compare-dir", default=None)
    p.add_argument("--dry-run", action="store_true", help="Print planned commands without executing them")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--skip-build-input", action="store_true")
    p.add_argument("--skip-compare", action="store_true")
    return p.parse_args(_strip_double_dash(raw))


def _validate_share_profile(profile_path: Path) -> None:
    if not profile_path.exists():
        raise SystemExit(
            f"share profile not found: {profile_path}. "
            "Provide --share-profile or set XRPL_SHARE_PROFILE."
        )

    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"share profile is not valid JSON: {profile_path} ({exc})")

    exp_raw = payload.get("expirationTime")
    if exp_raw:
        try:
            exp = datetime.fromisoformat(exp_raw.replace("Z", "+00:00"))
        except Exception:
            raise SystemExit(f"share profile has unparseable expirationTime: {exp_raw}")
        if exp <= datetime.now(timezone.utc):
            raise SystemExit(
                f"share profile is expired at {exp.isoformat()} (UTC): {profile_path}"
            )

    mode = stat.S_IMODE(profile_path.stat().st_mode)
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        print(
            f"[warn] {profile_path} is group/other-readable "
            f"(mode {oct(mode)}). Recommended: chmod 600 {profile_path}"
        )


def _run_pipeline(root: Path, raw_args: list[str]) -> int:
    args = _parse_pipeline_args(raw_args)
    cfg_path = root / args.config
    cfg_data = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)

    pair = args.pair or cfg_data.get("pair", "rlusd_xrp")
    ledger_start = args.ledger_start if args.ledger_start is not None else cfg_data.get("ledger_start")
    ledger_end = args.ledger_end if args.ledger_end is not None else cfg_data.get("ledger_end")
    if ledger_start is None or ledger_end is None:
        raise SystemExit("pipeline-run requires --ledger-start and --ledger-end (or defaults in --config).")
    share_profile = (
        args.share_profile
        or os.environ.get("XRPL_SHARE_PROFILE")
        or cfg_data.get("share_profile", "data/config.share")
    )
    share = args.share or cfg_data.get("share", "ripple-ubri-share")
    schema = args.schema or cfg_data.get("schema", "ripplex")
    table_amm = args.table_amm or cfg_data.get("table_amm", "fact_amm_swaps")
    table_clob = args.table_clob or cfg_data.get("table_clob", "offers_fact_tx")
    table_fees = args.table_fees or cfg_data.get("table_fees", "fact_amm_fees")

    window = f"ledger_{ledger_start}_{ledger_end}"

    exports_dir = args.exports_dir or f"artifacts/exports/{pair}/{window}"
    model_input_dir = args.model_input_dir or f"artifacts/model_input/{pair}/{window}"
    compare_dir = args.compare_dir or f"artifacts/compare/{pair}/{window}"

    planned: list[tuple[str, list[str]]] = []

    if not args.skip_export:
        _validate_share_profile(root / share_profile)
        planned.append(
            (
                "empirical/scripts/empirical_export_window.py",
                [
                    "--pair",
                    pair,
                    "--ledger-start",
                    str(ledger_start),
                    "--ledger-end",
                    str(ledger_end),
                    "--share-profile",
                    share_profile,
                    "--share",
                    share,
                    "--schema",
                    schema,
                    "--table-amm",
                    table_amm,
                    "--table-clob",
                    table_clob,
                    "--table-fees",
                    table_fees,
                    "--output-dir",
                    exports_dir,
                ],
            )
        )

    if not args.skip_build_input:
        planned.append(
            (
                "empirical/scripts/empirical_build_model_input.py",
                [
                    "--input-amm",
                    f"{exports_dir}/amm_swaps",
                    "--input-clob",
                    f"{exports_dir}/clob_legs",
                    "--input-fees",
                    f"{exports_dir}/amm_fees",
                    "--pair",
                    pair,
                    "--ledger-start",
                    str(ledger_start),
                    "--ledger-end",
                    str(ledger_end),
                    "--output-dir",
                    model_input_dir,
                ],
            )
        )

    if not args.skip_compare:
        if not args.book_gets_xrp or not args.book_gets_rusd:
            raise SystemExit(
                "compare step requires --book-gets-xrp and --book-gets-rusd (or pass --skip-compare)."
            )
        planned.append(
            (
                "empirical/scripts/empirical_compare_rolling.py",
                [
                    "--root",
                    exports_dir,
                    "--pair",
                    pair,
                    "--ledger-start",
                    str(ledger_start),
                    "--ledger-end",
                    str(ledger_end),
                    "--amm-swaps",
                    f"{exports_dir}/amm_swaps",
                    "--amm-fees",
                    f"{exports_dir}/amm_fees",
                    "--clob-legs",
                    f"{exports_dir}/clob_legs",
                    "--book-gets-xrp",
                    args.book_gets_xrp,
                    "--book-gets-rusd",
                    args.book_gets_rusd,
                    "--output-dir",
                    compare_dir,
                ],
            )
        )

    if args.dry_run:
        print("[dry-run] planned pipeline steps:")
        for i, (script, argv) in enumerate(planned, start=1):
            print(f"  {i}. python {script} {' '.join(argv)}")
        return 0

    for script, argv in planned:
        _run_script(root, script, argv)

    return 0


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.alias == "pipeline-run":
        return _run_pipeline(root, args.script_args)
    canonical = LEGACY_ALIAS_MAP.get(args.alias, args.alias)
    if canonical != args.alias:
        print(f"[info] alias '{args.alias}' is legacy; prefer '{canonical}'.")
    if canonical not in SCRIPT_MAP:
        choices = sorted([*SCRIPT_MAP.keys(), *LEGACY_ALIAS_MAP.keys(), "pipeline-run"])
        raise SystemExit(f"Unknown alias: {args.alias}. Available: {', '.join(choices)}")
    _run_script(root, SCRIPT_MAP[canonical], args.script_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
