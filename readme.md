# XRPL AMM/CLOB Routing Experiments

This repository provides a modelling and simulation toolkit to study **execution efficiency** on XRPL across AMM and CLOB venues. It centralises Decimal policy (precision, quantum, rounding) so AMM, CLOB and the router share identical arithmetic.

## Whitepaper alignment (cross‑reference)
- **Reverse → Forward with limiting‑step replay** — §1.3.2. The router performs a reverse dry‑run (no write‑backs) to find required inputs and the limiting step, then forward replays with state application.
- **Multi‑path AMM behaviour (Fibonacci slicing & bounded iterations)** — §1.2.7.3. Per‑iteration AMM OUT is capped (Fibonacci growth from a base), and AMM participation is bounded in count.
- **Mode isolation for apples‑to‑apples** — `CLOB_ONLY`, `AMM_ONLY`, `HYBRID` execution modes are exposed for fair comparisons.
- **Efficiency metrics** — Per‑iteration and aggregate metrics (`IterationMetrics`, `ExecutionReport`) are returned with each run. AMM pool/issuer fees can be metered via an `apply_sink` wrapper that previews fees for each `(dx, dy)` write‑back.

> **Quantisation note.** Quality is defined as `OUT/IN` and quantised **down** to `QUALITY_QUANTUM`. IOU/XRP amounts are quantised on their respective grids. Keeping arithmetic consistent across reverse and forward passes is essential for fair AMM vs CLOB comparisons (whitepaper §1.3.2).

## Configuration for reproducible experiments
Central parameters live in `xrpl_router.core.ROUTING_CFG`. Adjusting them avoids editing source in multiple places.

| Key                  | Default | Whitepaper ref. | Effect (short)                                                                                                                           | Typical range   |
| -------------------- | ------: | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `fib_base_factor`    |  `1e-4` | §1.2.7.3        | Base OUT cap for AMM in multi‑path; per‑iteration caps then follow Fibonacci (base, base, 2·base, …). Larger ⇒ fewer but coarser slices. | `1e-5` … `1e-3` |
| `alpha_step_default` |  `0.05` | α* search       | Step for the α‑scan when searching the optimal split ratio α*. Smaller ⇒ finer curve, more runs.                                         | `0.01` … `0.10` |
| `qscan_coarse_steps` |    `12` | q* search       | Number of coarse grid points for crossover search.                                                                                       | `8` … `24`      |
| `qscan_refine_iters` |    `18` | q* search       | Bisection refinement iterations after a bracket is found.                                                                                | `12` … `24`     |
| `amm_max_iters`      |    `30` | §1.2.7.3        | Documented cap on AMM participation count; enforcement lives in `AMMContext`.                                                            | `20` … `50`     |

## Reproducing AMM vs CLOB comparisons
- **AMM‑only / CLOB‑only / Hybrid totals and reports** — `exec_modes.run_trade_mode` (modes: `AMM_ONLY`, `CLOB_ONLY`, `HYBRID`).
- **Hybrid (multi‑path) execution with observed α** — `efficiency_scan.hybrid_flow`.
- **Optimal split α\*** — `efficiency_scan.analyze_alpha_scan` (defaults read from `ROUTING_CFG`).
- **Crossover size q\*** — `efficiency_scan.find_crossover_q`.
- **Batch runs / CSV export** — `efficiency_scan.batch_analyze`, `efficiency_scan.batch_rows_to_csv`.

## Minimal usage sketch
```python
from decimal import Decimal
from xrpl_router.exec_modes import run_trade_mode, ExecutionMode
from xrpl_router.efficiency_scan import hybrid_flow, analyze_alpha_scan, find_crossover_q
from xrpl_router.core import ROUTING_CFG

# (1) Single run, apples‑to‑apples modes
res_clob  = run_trade_mode(ExecutionMode.CLOB_ONLY, target_out=Decimal("100"), segments=clob_segments)
res_amm   = run_trade_mode(ExecutionMode.AMM_ONLY,  target_out=Decimal("100"), segments=clob_segments,
                           amm_anchor=amm_anchor, amm_curve=amm_curve, amm_context=amm_ctx)
res_hyb   = hybrid_flow(target_out=Decimal("100"), clob_segments=clob_segments,
                        amm_anchor=amm_anchor, amm_curve=amm_curve, amm_context=amm_ctx)

# (2) Optimal split alpha*
scan = analyze_alpha_scan(target_out=Decimal("100"), segments=clob_segments,
                          amm_anchor=amm_anchor, amm_curve=amm_curve, amm_context=amm_ctx,
                          step=ROUTING_CFG.alpha_step_default)

# (3) Crossover size q*
qstar = find_crossover_q(segments=clob_segments, amm_anchor=amm_anchor,
                         amm_curve=amm_curve, amm_context=amm_ctx)
```

## Notes
- The toolkit focuses on **execution efficiency** (effective prices, slippage, fee breakdowns) rather than empirical data collection. Empirical pipelines can be integrated upstream using the same interfaces.
- All defaults aim to reflect the whitepaper behaviours (§1.2.7.3, §1.3.2) while remaining tunable via `ROUTING_CFG` for sensitivity analyses.