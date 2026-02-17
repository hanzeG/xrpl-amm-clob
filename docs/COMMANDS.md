# Commands

## Model
```bash
python apps/run_model.py amm-only
python apps/run_model.py hybrid-one-tx
```

## Empirical Pipeline (Recommended)
```bash
python apps/run_empirical.py pipeline-run -- \
  --config configs/empirical/pipeline_run.default.json \
  --ledger-start <start> --ledger-end <end> \
  --book-gets-xrp <book_gets_xrp.ndjson> \
  --book-gets-rusd <book_gets_rusd.ndjson>
```

Dry-run only:
```bash
python apps/run_empirical.py pipeline-run -- \
  --config configs/empirical/pipeline_run.default.json \
  --ledger-start <start> --ledger-end <end> \
  --book-gets-xrp <book_gets_xrp.ndjson> \
  --book-gets-rusd <book_gets_rusd.ndjson> \
  --dry-run
```

## Empirical Script Aliases
```bash
python apps/run_empirical.py delta-sharing-check-freshness
python apps/run_empirical.py delta-sharing-test-profile
python apps/run_empirical.py empirical-download-clob-offers-range -- --help
python apps/run_empirical.py empirical-export-window -- --help
python apps/run_empirical.py empirical-build-model-input -- --help
python apps/run_empirical.py empirical-compare-rolling -- --help
python apps/run_empirical.py empirical-compare-single -- --help
python apps/run_empirical.py empirical-analyze-traces -- --help
python apps/run_empirical.py empirical-enrich-clob-with-tx-index -- --help
python apps/run_empirical.py empirical-check-parquet-bounds -- --help
python apps/run_empirical.py empirical-reconstruct-real-paths -- --help
```

Use a custom Delta Sharing profile path:
```bash
XRPL_SHARE_PROFILE=/absolute/path/to/config.share python apps/run_empirical.py delta-sharing-test-profile
```

Legacy aliases remain supported with a compatibility info message.
