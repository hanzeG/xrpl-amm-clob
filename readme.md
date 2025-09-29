"""
Numeric primitives for XRPL AMM/CLOB routing experiments.
Centralises Decimal policy (precision, quantum, rounding) so AMM, CLOB, and the router share identical arithmetic.

Whitepaper alignment (for cross‑reference):
- §1.3.2  Reverse→Forward execution and limiting‑step replay — these helpers ensure price/amount rounding is consistent across reverse pass estimates and forward replays.
- §1.2.7.3 Multi‑path AMM behaviour (Fibonacci slicing & bounded iterations) — configuration keys in `RoutingConfig` are used by higher layers to enforce the AMM OUT caps each iteration.

Quality is always quantised on QUALITY_QUANTUM (down); IOU/XRP amounts are quantised on their respective grids.
"""

from dataclasses import dataclass

@dataclass
class RoutingConfig:
    """
    Centralised knobs used by routing and analysis utilities.

    Keeping them here ensures experiments are reproducible without changing code
    across modules. Callers can import and override `ROUTING_CFG` at runtime if needed.

    Whitepaper cross‑references:
    - `fib_base_factor` → §1.2.7.3 (Fibonacci slicing in multi‑path AMM). Controls the base OUT cap for the first two AMM slices each time AMM participates; caps then grow fibonaccially per iteration.
    - `alpha_step_default` → α‑scan granularity when searching for the optimal split ratio α* (task spec: Execution Efficiency).
    - `qscan_coarse_steps` / `qscan_refine_iters` → scanning and bisection depth for the crossover size q* between AMM‑only and CLOB‑only costs.
    - `amm_max_iters` → documented upper bound for AMM participation count (whitepaper: capped iterations); enforcement lives in AMMContext.

    Recommended ranges (typical studies):
    - `fib_base_factor`: 1e‑5 … 1e‑3  (larger ⇒ fewer but coarser AMM slices per iteration)
    - `alpha_step_default`: 0.01 … 0.10 (smaller ⇒ finer α curve, more runs)
    - `qscan_coarse_steps`: 8 … 24; `qscan_refine_iters`: 12 … 24
    """
    fib_base_factor: float = 1e-4
    alpha_step_default: float = 0.05
    qscan_coarse_steps: int = 12
    qscan_refine_iters: int = 18
    amm_max_iters: int = 30

## The rest of the file remains unchanged.


# Append to /Users/guohanze/Documents/Codebase/xrpl-amm-clob/readme.md

# After the existing content in readme.md, append:

"""

## Configuration for Reproducible Experiments
The simulation reads central parameters from `xrpl_router.core.ROUTING_CFG`. Adjusting these avoids editing source in multiple places.

| Key                  | Default | Whitepaper ref. | Effect (short)                                                                                                                           | Typical range   |
| -------------------- | ------: | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `fib_base_factor`    |  `1e-4` | §1.2.7.3        | Base OUT cap for AMM in multi‑path; per‑iteration caps then follow Fibonacci (base, base, 2·base, …). Larger ⇒ fewer but coarser slices. | `1e-5` … `1e-3` |
| `alpha_step_default` |  `0.05` | α* (task spec)  | Step for α‑scan when searching the optimal split ratio α*. Smaller ⇒ finer curve, more runs.                                             | `0.01` … `0.10` |
| `qscan_coarse_steps` |    `12` | q* search       | Number of coarse grid points for crossover search.                                                                                       | `8` … `24`      |
| `qscan_refine_iters` |    `18` | q* search       | Bisection refinement iterations after a bracket is found.                                                                                | `12` … `24`     |
| `amm_max_iters`      |    `30` | §1.2.7.3        | Documented cap on AMM participation count; enforced in `AMMContext`.                                                                     | `20` … `50`     |

> Note: quality is `OUT/IN` and is always quantised down to `QUALITY_QUANTUM`; IOU/XRP amounts are quantised on their respective grids. Keeping arithmetic consistent across reverse and forward passes is essential for fair AMM vs CLOB comparisons (§1.3.2).

## Whitepaper Alignment (Quick Cross‑walk)
- **Reverse→Forward with limiting‑step replay** (§1.3.2): implemented in `flow.py` and `steps.py` with reverse dry‑run (no write‑backs) and forward replay applying state.
- **Multi‑path selection**: per‑iteration path chosen by upper‑bound quality. Hybrid execution exposed via `efficiency_scan.hybrid_flow`.
- **AMM Fibonacci slicing & capped participation** (§1.2.7.3): enforced via per‑iteration AMM OUT caps (router) and participation counting (AMMContext).
- **Mode isolation for apples‑to‑apples**: `exec_modes.run_trade_mode` exposes `CLOB_ONLY`, `AMM_ONLY`, `HYBRID`.
- **Efficiency metrics**: per‑iteration and aggregate metrics available in `RouteResult.report` (`IterationMetrics`, `ExecutionReport`). Fees (pool / issuer) are metered via `apply_sink` with an AMM preview helper.

## Reproducing AMM vs CLOB Comparisons
- **AMM‑only / CLOB‑only / Hybrid** totals and reports: use `exec_modes.run_trade_mode` and `efficiency_scan.hybrid_flow`.
- **Optimal split α\***: use `efficiency_scan.analyze_alpha_scan` (defaults read from `ROUTING_CFG`).
- **Crossover size q\***: use `efficiency_scan.find_crossover_q`.
- **Batch runs / CSV**: use `efficiency_scan.batch_analyze` and `batch_rows_to_csv`.

"""