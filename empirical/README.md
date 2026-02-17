# Empirical Workflows

## Active Scripts
- `empirical_export_window.py`
- `empirical_build_model_input.py`
- `empirical_compare_rolling.py`
- `empirical_compare_single.py`
- `empirical_analyze_traces.py`
- `empirical_enrich_clob_with_tx_index.py`
- `empirical_check_parquet_bounds.py`
- `delta_sharing_check_freshness.py`
- `delta_sharing_test_profile.py`
- `empirical_download_clob_offers_range.py`

## Recommended Pipeline
```bash
python apps/run_empirical.py pipeline-run -- \
  --config configs/empirical/pipeline_run.default.json \
  --ledger-start <start> --ledger-end <end> \
  --book-gets-xrp <book_gets_xrp.ndjson> \
  --book-gets-rusd <book_gets_rusd.ndjson>
```

Command reference: `docs/COMMANDS.md`

## Credentials
- Local token: `data/config.share`
- Template: `data/config.share.example`
- Optional override: `XRPL_SHARE_PROFILE=/absolute/path/to/config.share`
- Recommended local permission: `chmod 600 data/config.share`
