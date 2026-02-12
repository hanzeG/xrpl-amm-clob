# Empirical Zone

Empirical workflows include:

- Delta Sharing discovery/export
- Ledger trace extraction
- Windowed comparison and diagnostics

Scripts are now in `empirical/scripts/`.
Use `apps/run_empirical.py` as the stable launcher.
Local Delta Sharing token should be stored in `data/config.share` (copied from `data/config.share.example`).
Archived one-off scripts are stored in `empirical/scripts_archive/`.

Whitelisted active scripts:
- `empirical/scripts/pipeline_export_window.py`
- `empirical/scripts/pipeline_build_model_input.py`
- `empirical/scripts/research_compare_rolling.py`
- `empirical/scripts/research_compare_single.py`
- `empirical/scripts/research_analyse_traces.py`
- `empirical/scripts/research_enrich_clob_with_tx_index.py`
- `empirical/scripts/research_check_parquet_bounds.py`
- `empirical/scripts/check_delta_sharing_freshness.py`
- `empirical/scripts/test_delta_sharing_profile.py`
- `empirical/scripts/download_clob_offers_range.py`

Examples:
- `python apps/run_empirical.py pipeline-export-window -- --pair rlusd_xrp --ledger-start <n> --ledger-end <n> --output-dir artifacts/exports/rlusd_xrp/ledger_<start>_<end>`
- `python apps/run_empirical.py pipeline-build-model-input -- --input-amm <amm_parquet> --input-clob <clob_parquet> --input-fees <fees_parquet> --pair rlusd_xrp --ledger-start <n> --ledger-end <n>`
- `python apps/run_empirical.py research-compare-rolling -- --root <window_dir> --pair rlusd_xrp --ledger-start <n> --ledger-end <n> --output-dir artifacts/compare/rlusd_xrp/ledger_<start>_<end>`
- `python apps/run_empirical.py research-compare-single -- --root <window_dir> --pair rlusd_xrp --ledger-start <n> --ledger-end <n> --output-dir artifacts/compare/rlusd_xrp/ledger_<start>_<end>`
- `python apps/run_empirical.py pipeline-run -- --config configs/empirical/pipeline_run.default.json --ledger-start <n> --ledger-end <n> --book-gets-xrp <book_gets_xrp.ndjson> --book-gets-rusd <book_gets_rusd.ndjson>`
