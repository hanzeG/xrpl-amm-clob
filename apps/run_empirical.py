#!/usr/bin/env python3
"""Stable launcher for empirical scripts."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


SCRIPT_MAP = {
    "discovery": "empirical/scripts/discovery.py",
    "analyse": "empirical/scripts/analyse.py",
    "check-freshness": "empirical/scripts/check_delta_sharing_freshness.py",
    "export-window": "empirical/scripts/pipeline_export_window.py",
    "export-fees-window": "empirical/scripts/pipeline_export_window.py",
    "build-input": "empirical/scripts/pipeline_build_model_input.py",
    "inspect-window": "empirical/scripts/inspect_window.py",
    "inspect-hybrid-tx": "empirical/scripts/inspect_hybrid_path_one_tx.py",
    "scan-hybrid": "empirical/scripts/scan_hybrid_candidates_20251201.py",
    "trace-ledger": "empirical/scripts/trace_ledger_rusd_xrp_1215.py",
    "compare-vs-model": "empirical/scripts/compare_rusd_xrp_100894647_vs_model.py",
    "test-share-profile": "empirical/scripts/test_delta_sharing_profile.py",
    "research-compare-rolling": "empirical/scripts/research_compare_rolling.py",
    "research-compare-single": "empirical/scripts/research_compare_single.py",
    "research-analyse-traces": "empirical/scripts/research_analyse_traces.py",
    "research-enrich-clob": "empirical/scripts/research_enrich_clob_with_tx_index.py",
    "research-check-parquet": "empirical/scripts/research_check_parquet_bounds.py",
    "pipeline-export-window": "empirical/scripts/pipeline_export_window.py",
    "pipeline-build-model-input": "empirical/scripts/pipeline_build_model_input.py",
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
    p.add_argument("--pair", default="rlusd_xrp")
    p.add_argument("--ledger-start", type=int, required=True)
    p.add_argument("--ledger-end", type=int, required=True)
    p.add_argument("--share-profile", default="data/config.share")
    p.add_argument("--share", default="ripple-ubri-share")
    p.add_argument("--schema", default="ripplex")
    p.add_argument("--table-amm", default="fact_amm_swaps")
    p.add_argument("--table-clob", default="offers_fact_tx")
    p.add_argument("--table-fees", default="fact_amm_fees")
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


def _run_pipeline(root: Path, raw_args: list[str]) -> int:
    args = _parse_pipeline_args(raw_args)
    window = f"ledger_{args.ledger_start}_{args.ledger_end}"

    exports_dir = args.exports_dir or f"artifacts/exports/{args.pair}/{window}"
    model_input_dir = args.model_input_dir or f"artifacts/model_input/{args.pair}/{window}"
    compare_dir = args.compare_dir or f"artifacts/compare/{args.pair}/{window}"

    planned: list[tuple[str, list[str]]] = []

    if not args.skip_export:
        planned.append(
            (
                "empirical/scripts/pipeline_export_window.py",
                [
                    "--pair",
                    args.pair,
                    "--ledger-start",
                    str(args.ledger_start),
                    "--ledger-end",
                    str(args.ledger_end),
                    "--share-profile",
                    args.share_profile,
                    "--share",
                    args.share,
                    "--schema",
                    args.schema,
                    "--table-amm",
                    args.table_amm,
                    "--table-clob",
                    args.table_clob,
                    "--table-fees",
                    args.table_fees,
                    "--output-dir",
                    exports_dir,
                ],
            )
        )

    if not args.skip_build_input:
        planned.append(
            (
                "empirical/scripts/pipeline_build_model_input.py",
                [
                    "--input-amm",
                    f"{exports_dir}/amm_swaps",
                    "--input-clob",
                    f"{exports_dir}/clob_legs",
                    "--input-fees",
                    f"{exports_dir}/amm_fees",
                    "--pair",
                    args.pair,
                    "--ledger-start",
                    str(args.ledger_start),
                    "--ledger-end",
                    str(args.ledger_end),
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
                "empirical/scripts/research_compare_rolling.py",
                [
                    "--root",
                    exports_dir,
                    "--pair",
                    args.pair,
                    "--ledger-start",
                    str(args.ledger_start),
                    "--ledger-end",
                    str(args.ledger_end),
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
    if args.alias not in SCRIPT_MAP:
        choices = sorted([*SCRIPT_MAP.keys(), "pipeline-run"])
        raise SystemExit(f"Unknown alias: {args.alias}. Available: {', '.join(choices)}")
    _run_script(root, SCRIPT_MAP[args.alias], args.script_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
