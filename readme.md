# XRPL AMM/CLOB Router

XRPL AMM/CLOB routing and execution-efficiency research toolkit.

## Start Here: Single-Tx Dual Replay Notebook
Primary workflow now: run one tx end-to-end in
`notebooks/single_tx_dual_replay.ipynb`.

### Run
1. Install deps:
```bash
python -m pip install -e .
```
2. Put Delta Sharing profile at `data/config.share`.
3. Open notebook:
```bash
jupyter lab notebooks/single_tx_dual_replay.ipynb
```
or open the same file in VSCode Notebook.
4. In the notebook config cell, set `TARGET_TX_HASH`.
5. Restart kernel, then **Run All**.

### Outputs
- Real trajectory table (metadata order + DS-filled amounts, pair-scoped rows)
- Model trajectory table (single-path replay; AMM slices collapsed to one row)
- Key compare lines (`in` gap %, AMM input share compare, offer-id overlap)

## Structure
- `src/xrpl_router/`: routing core and math
- `empirical/`: data export and comparison workflows
- `apps/`: stable CLI entry points
- `tests/`: automated tests
- `artifacts/`: generated outputs (ignored)

## Quick Start (General)
```bash
python -m pip install -e .
pytest -q tests/ci
```

Command reference: `docs/COMMANDS.md`

## Other Main Commands
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
