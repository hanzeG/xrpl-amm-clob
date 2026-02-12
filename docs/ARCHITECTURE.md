# Architecture

## Zones
- `src/xrpl_router/`: model core
- `apps/model/`: model execution scripts
- `empirical/scripts/`: active empirical workflows
- `empirical/scripts_archive/`: historical one-off scripts
- `apps/`: stable CLI launchers
- `tests/`: validation
- `artifacts/`: output datasets

## Execution Entry Points
- Model: `python apps/run_model.py <alias>`
- Empirical: `python apps/run_empirical.py <alias>`
