# Architecture

This repository is now organized into clear zones with single-path script layout.

## Zones

- `src/xrpl_router/`: model core (routing logic, AMM/CLOB math, quality/ordering)
- `empirical/`: empirical workflows (data discovery/export/trace/analysis)
- `apps/`: stable CLI entry points for model and empirical jobs
- `tests/`: unit/integration/example tests
- `artifacts/`: generated outputs (parquet/csv/ndjson/log), not tracked by git

## Current Status

Repository cleanup status:

- No model behavior changes
- No algorithm changes in empirical scripts
- Empirical scripts live in `empirical/scripts/`
- No compatibility shims are kept under `data/`
- Legacy root-level outputs were migrated to `artifacts/snapshots/legacy_root/`

## Migration Plan (Next Phases)

1. Extract shared empirical utilities (Delta Sharing load/filter/write)
2. Move stable configs to `configs/`
3. Normalize defaults from `data/output*` toward `artifacts/` over time
