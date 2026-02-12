# Architecture

This repository is organized into clear zones with stable script entry points.

## Zones

- `src/xrpl_router/`: model core (routing logic, AMM/CLOB math, quality/ordering)
- `empirical/`: empirical workflows (data discovery/export/trace/analysis)
- `apps/`: stable CLI entry points for model and empirical jobs
- `tests/`: unit/integration/example tests
- `artifacts/`: generated outputs (parquet/csv/ndjson/log), not tracked by git

## Operational Notes

- Model and empirical code paths are separated.
- Empirical active scripts live in `empirical/scripts/`; archived one-off scripts live in `empirical/scripts_archive/`.
- Empirical scripts are invoked via `apps/run_empirical.py`.
- Generated outputs are stored under `artifacts/`.
