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

### Example Output (Current Single-Tx Demo)
Tx:
`6AA281DD3ED17153F5149EE82965563D8B44BAB69226717EAD4E68326CF96A64`

Real trajectory table (excerpt):

| segment_no | segment_kind | node_type | clob_offer_id | out_take | in_take | avg_quality |
|---:|---|---|---|---:|---:|---:|
| 1 | CLOB | DeletedNode | 27CC | 1910.999520 | 983.818990 | 1.942429999 |
| 2 | CLOB | DeletedNode | AE48 | 38.828862 | 20.000000 | 1.941443123 |
| 3 | CLOB | DeletedNode | 655A | 9.883000 | 5.091257 | 1.941170913 |
| 12 | AMM | ModifiedNode | None | 1020.051024 | 526.257185 | 1.938312773 |

Model trajectory table (excerpt):

| segment_no | segment_kind | node_type | clob_offer_id | out_take | in_take | avg_quality |
|---:|---|---|---|---:|---:|---:|
| 1 | CLOB | DeletedNode(model) | 27CC | 1910.999520 | 983.818991 | 1.942429997 |
| 2 | CLOB | DeletedNode(model) | AE48 | 38.828862 | 20.000000 | 1.941443123 |
| 3 | CLOB | DeletedNode(model) | 655A | 9.883000 | 5.091257 | 1.941170913 |
| 12 | AMM | ModifiedNode(model-AMM) | None | 1020.051025 | 526.257186 | 1.938312772 |

The notebook prints the full two tables for the selected tx.
Use them directly to compare real execution trajectory vs single-path model trajectory.

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
