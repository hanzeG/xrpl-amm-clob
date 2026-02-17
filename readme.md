# XRPL AMM/CLOB Router

XRPL AMM/CLOB routing and execution-efficiency research toolkit.

## Structure
- `src/xrpl_router/`: routing core and math
- `empirical/`: data export and comparison workflows
- `apps/`: stable CLI entry points
- `tests/`: automated tests
- `artifacts/`: generated outputs (ignored)

## Quick Start
```bash
python -m pip install -e .
pytest -q tests/ci
```

Command reference: `docs/COMMANDS.md`

## Main Commands
```bash
python apps/run_model.py amm-only
python apps/run_model.py hybrid-one-tx
python apps/run_empirical.py pipeline-run -- \
  --config configs/empirical/pipeline_run.default.json \
  --ledger-start <start> --ledger-end <end> \
  --book-gets-xrp <book_gets_xrp.ndjson> \
  --book-gets-rusd <book_gets_rusd.ndjson>
```

## Notes
- Empirical defaults are centralized in `configs/empirical/pipeline_run.default.json`.
- Credentials are local-only via `data/config.share` (template: `data/config.share.example`).
- You can override the profile path with `XRPL_SHARE_PROFILE=/absolute/path/to/config.share`.
- Recommended local permission: `chmod 600 data/config.share`.
