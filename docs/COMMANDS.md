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
python apps/run_empirical.py check-freshness
python apps/run_empirical.py test-share-profile
python apps/run_empirical.py download-clob-offers-range -- --help
python apps/run_empirical.py pipeline-export-window -- --help
python apps/run_empirical.py pipeline-build-model-input -- --help
python apps/run_empirical.py research-compare-rolling -- --help
python apps/run_empirical.py research-compare-single -- --help
python apps/run_empirical.py research-analyse-traces -- --help
python apps/run_empirical.py research-enrich-clob -- --help
python apps/run_empirical.py research-check-parquet -- --help
```
