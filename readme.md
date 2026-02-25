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

Real trajectory table:

| segment_no | segment_kind | node_type    | clob_offer_id | out_take     | in_take     | avg_quality |
| ---------: | ------------ | ------------ | ------------- | ------------ | ----------- | ----------- |
|          1 | CLOB         | DeletedNode  | 27CC          | 1910.999520  | 983.818990  | 1.942429999 |
|          2 | CLOB         | DeletedNode  | AE48          | 38.828862    | 20.000000   | 1.941443123 |
|          3 | CLOB         | DeletedNode  | 655A          | 9.883000     | 5.091257    | 1.941170913 |
|          4 | CLOB         | DeletedNode  | C576          | 106.000000   | 54.611306   | 1.940990021 |
|          5 | CLOB         | DeletedNode  | FABB          | 1909.081070  | 983.818990  | 1.940479996 |
|          6 | CLOB         | DeletedNode  | 9075          | 8.412263     | 4.336217    | 1.940000363 |
|          7 | CLOB         | DeletedNode  | 4D56          | 5.424320     | 2.796041    | 1.940000128 |
|          8 | CLOB         | DeletedNode  | 3695          | 11640.000000 | 6000.000000 | 1.940000000 |
|          9 | CLOB         | DeletedNode  | EBC1          | 3571.946277  | 1841.209421 | 1.940000000 |
|         10 | CLOB         | DeletedNode  | 1626          | 2860.758690  | 1475.728480 | 1.938540002 |
|         11 | CLOB         | DeletedNode  | F6B8          | 67.000000    | 34.562955   | 1.938491660 |
|         12 | AMM          | ModifiedNode | None          | 1020.051024  | 526.257185  | 1.938312773 |
|         13 | CLOB         | DeletedNode  | 4A8B          | 9.883000     | 5.099133    | 1.938172627 |
|         14 | CLOB         | DeletedNode  | 2D60          | 2238.938272  | 1155.760000 | 1.937200000 |

Model trajectory table (before AMM merge):

| segment_no | segment_kind | node_type               | clob_offer_id | out_take     | in_take     | avg_quality |
| ---------: | ------------ | ----------------------- | ------------- | ------------ | ----------- | ----------- |
|          1 | CLOB         | DeletedNode(model)      | 27CC          | 1910.999520  | 983.818991  | 1.942429997 |
|          2 | CLOB         | DeletedNode(model)      | AE48          | 38.828862    | 20.000000   | 1.941443123 |
|          3 | CLOB         | DeletedNode(model)      | 655A          | 9.883000     | 5.091257    | 1.941170913 |
|          4 | CLOB         | DeletedNode(model)      | C576          | 106.000000   | 54.611306   | 1.940990021 |
|          5 | CLOB         | DeletedNode(model)      | FABB          | 1909.081070  | 983.818990  | 1.940479996 |
|          6 | CLOB         | DeletedNode(model)      | 4D56          | 5.424320     | 2.796041    | 1.940000128 |
|          7 | CLOB         | DeletedNode(model)      | 3695          | 11640.000000 | 6000.000000 | 1.940000000 |
|          8 | CLOB         | DeletedNode(model)      | EBC1          | 3571.946277  | 1841.209421 | 1.940000000 |
|          9 | CLOB         | DeletedNode(model)      | 9075          | 8.412263     | 4.336218    | 1.939999916 |
|         10 | CLOB         | DeletedNode(model)      | 1626          | 2860.758690  | 1475.728480 | 1.938540002 |
|         11 | CLOB         | DeletedNode(model)      | F6B8          | 67.000000    | 34.562955   | 1.938491660 |
|         12 | AMM          | ModifiedNode(model-AMM) | None          | 1020.051025  | 526.257186  | 1.938312772 |
|         13 | CLOB         | DeletedNode(model)      | 4A8B          | 9.883000     | 5.099133    | 1.938172627 |
|         14 | CLOB         | DeletedNode(model)      | 2D60          | 2238.938272  | 1155.760000 | 1.937200000 |

Model trajectory table (after AMM merge):

| segment_no | segment_kind | node_type               | clob_offer_id | out_take     | in_take     | avg_quality |
| ---------: | ------------ | ----------------------- | ------------- | ------------ | ----------- | ----------- |
|          1 | CLOB         | DeletedNode(model)      | 27CC          | 1910.999520  | 983.818991  | 1.942429997 |
|          2 | CLOB         | DeletedNode(model)      | AE48          | 38.828862    | 20.000000   | 1.941443123 |
|          3 | CLOB         | DeletedNode(model)      | 655A          | 9.883000     | 5.091257    | 1.941170913 |
|          4 | CLOB         | DeletedNode(model)      | C576          | 106.000000   | 54.611306   | 1.940990021 |
|          5 | CLOB         | DeletedNode(model)      | FABB          | 1909.081070  | 983.818990  | 1.940479996 |
|          6 | CLOB         | DeletedNode(model)      | 4D56          | 5.424320     | 2.796041    | 1.940000128 |
|          7 | CLOB         | DeletedNode(model)      | 3695          | 11640.000000 | 6000.000000 | 1.940000000 |
|          8 | CLOB         | DeletedNode(model)      | EBC1          | 3571.946277  | 1841.209421 | 1.940000000 |
|          9 | CLOB         | DeletedNode(model)      | 9075          | 8.412263     | 4.336218    | 1.939999916 |
|         10 | CLOB         | DeletedNode(model)      | 1626          | 2860.758690  | 1475.728480 | 1.938540002 |
|         11 | CLOB         | DeletedNode(model)      | F6B8          | 67.000000    | 34.562955   | 1.938491660 |
|         12 | AMM          | ModifiedNode(model-AMM) | None          | 1020.051025  | 526.257186  | 1.938312772 |
|         13 | CLOB         | DeletedNode(model)      | 4A8B          | 9.883000     | 5.099133    | 1.938172627 |
|         14 | CLOB         | DeletedNode(model)      | 2D60          | 2238.938272  | 1155.760000 | 1.937200000 |

The notebook prints the full three tables for the selected tx.
Use them directly to compare real execution trajectory, model pre-merge AMM slices, and merged model trajectory.

## BookStep Priority (Current)
- Routing is processed by quality from best to worse.
- If no CLOB exists at the current top quality, the step uses AMM-only synthetic liquidity for that level.
- If CLOB exists, AMM is allowed to anchor first only when `AMM SPQ > CLOB top quality` (strictly better).
- If `AMM SPQ == CLOB top quality` (or worse), CLOB is preferred first.
- Within a level, matching stays on that level's liquidity (AMM synthetic and/or same-quality CLOB) and does not jump to worse quality until the current level is exhausted or target is met.

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
