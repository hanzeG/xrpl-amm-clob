

# XRPL AMM/CLOB Routing Simulation

This repository is under active development. It aims to simulate routing algorithms between **AMM (Automated Market Maker)** and **CLOB (Central Limit Order Book)** on the XRP Ledger.

## Environment Setup

We recommend using **conda**:

```bash
git clone https://github.com/hanzeG/xrpl-amm-clob.git
cd xrpl-amm-clob

conda create -n xrpl-amm-clob python=3.11 -y
conda activate xrpl-amm-clob
```

Install dependencies using `pyproject.toml`:

```bash
pip install -U pip poetry
poetry install
poetry shell
```

## Quick Start

Run the demo file to see results across different routing scenarios:

```bash
python demo.py
```