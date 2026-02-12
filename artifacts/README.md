# Artifacts

Generated datasets and experiment outputs are stored under this directory.

## Layout

- `exports/<pair>/ledger_<start>_<end>/`: exported AMM/CLOB/fees windows
- `compare/<pair>/ledger_<start>_<end>/`: comparison outputs
- `model_input/<pair>/ledger_<start>_<end>/`: generated model input files

## Policy

- Do not commit files under `artifacts/`
- Keep source code and configs outside `artifacts/`
