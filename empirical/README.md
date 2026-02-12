# Empirical Workflows

## Active Scripts
- `pipeline_export_window.py`
- `pipeline_build_model_input.py`
- `research_compare_rolling.py`
- `research_compare_single.py`
- `research_analyse_traces.py`
- `research_enrich_clob_with_tx_index.py`
- `research_check_parquet_bounds.py`
- `check_delta_sharing_freshness.py`
- `test_delta_sharing_profile.py`
- `download_clob_offers_range.py`

## Recommended Pipeline
```bash
python apps/run_empirical.py pipeline-run -- \
  --config configs/empirical/pipeline_run.default.json \
  --ledger-start <start> --ledger-end <end> \
  --book-gets-xrp <book_gets_xrp.ndjson> \
  --book-gets-rusd <book_gets_rusd.ndjson>
```

## Credentials
- Local token: `data/config.share`
- Template: `data/config.share.example`
