"""
Numeric primitives for XRPL AMM/CLOB routing experiments.
Centralises Decimal policy (precision, quantum, rounding) so AMM, CLOB, and the router share identical arithmetic.

Whitepaper alignment (for cross‑reference):
- §1.3.2  Reverse→Forward execution and limiting‑step replay — these helpers ensure price/amount rounding is consistent across reverse pass estimates and forward replays.
- §1.2.7.3 Multi‑path AMM behaviour (Fibonacci slicing & bounded iterations) — configuration keys in `RoutingConfig` are used by higher layers to enforce the AMM OUT caps each iteration.

Quality is always quantised on QUALITY_QUANTUM (down); IOU/XRP amounts are quantised on their respective grids.
"""
from __future__ import annotations

from decimal import Decimal, getcontext, ROUND_DOWN, ROUND_UP
from typing import Union
from dataclasses import dataclass
from typing import Literal, Dict, Any, List, Optional

# -------------------------------
# Shared data structures
# -------------------------------

@dataclass(frozen=True)
class Segment:
    """A homogeneous quote slice used by the router."""
    src: Literal["AMM", "CLOB"]
    quality: Decimal            # OUT / IN (higher is better)
    out_max: Decimal            # Max OUT available on this slice
    in_at_out_max: Decimal      # IN required to consume out_max
    in_is_xrp: bool             # Input asset uses XRP grid (drops)
    out_is_xrp: bool            # Output asset uses XRP grid (drops)


# --- New: Per-iteration and aggregate execution metrics ---
@dataclass(frozen=True)
class IterationMetrics:
    """Per-iteration execution metrics for efficiency analysis."""
    iter_index: int
    tier_quality: Decimal
    out_filled: Decimal
    in_spent: Decimal
    price_effective: Decimal          # in_spent / out_filled (0 if out_filled == 0)
    amm_used: bool
    budget_limited: bool
    limit_quality_floor: Optional[Decimal]
    fee_pool: Decimal
    fee_tr_in: Decimal
    fee_tr_out: Decimal
    slippage_price: Decimal


@dataclass(frozen=True)
class ExecutionReport:
    """Aggregated execution report across all iterations."""
    iterations: List[IterationMetrics]
    total_out: Decimal
    total_in: Decimal
    avg_price: Decimal               # total_in / total_out (0 if total_out == 0)
    avg_quality: Decimal
    filled_ratio: Decimal            # total_out / target_out
    in_budget_ratio: Optional[Decimal]  # total_in / send_max if send_max provided
    fee_pool_total: Decimal
    fee_tr_in_total: Decimal
    fee_tr_out_total: Decimal
    slippage_price_avg: Decimal


@dataclass(frozen=True)
class RouteResult:
    """Router outcome with totals and step-by-step trace."""
    filled_out: Decimal
    spent_in: Decimal
    avg_quality: Decimal
    usage: Dict[str, Decimal]          # OUT consumed per source
    trace: List[Dict[str, Any]]        # Iteration records
    report: ExecutionReport | None = None

# -------------------------------
# Decimal context & constants
# -------------------------------
DEFAULT_DECIMAL_PRECISION: int = 28
getcontext().prec = DEFAULT_DECIMAL_PRECISION

# Amount/quality quanta; callers pick based on asset/metric.
XRP_QUANTUM: Decimal = Decimal("1")         # drops (integer)
IOU_QUANTUM: Decimal = Decimal("1e-15")     # IOU amounts
QUALITY_QUANTUM: Decimal = Decimal("1e-15") # quality grid
DEFAULT_QUANTUM: Decimal = IOU_QUANTUM

# -------------------------------
# Helpers (decimal rounding, quantization, and quality bucketing)
# -------------------------------

def clamp_nonneg(x: Decimal) -> Decimal:
    """Clamp negative to 0; pass through NaN/Inf unchanged."""
    if x.is_nan() or x.is_infinite():
        return x
    return x if x >= 0 else Decimal(0)

def to_decimal(x: Union[str, int, float, Decimal]) -> Decimal:
    """Convert to Decimal; prefer str/Decimal to avoid float artefacts."""
    if isinstance(x, Decimal):
        return x
    if isinstance(x, float):
        return Decimal(str(x))
    return Decimal(x)

def is_finite(x: Decimal) -> bool:
    """Return True iff x is a finite Decimal number."""
    return not (x.is_nan() or x.is_infinite())

def quantize_down(x: Decimal, quantum: Decimal = DEFAULT_QUANTUM) -> Decimal:
    """Quantise x downward to the quantum grid (ledger-favourable)."""
    if x.is_nan() or x.is_infinite():
        return x
    return x.quantize(quantum, rounding=ROUND_DOWN)

def quantize_up(x: Decimal, quantum: Decimal = DEFAULT_QUANTUM) -> Decimal:
    """Quantise x upward to the quantum grid (taker-favourable)."""
    if x.is_nan() or x.is_infinite():
        return x
    return x.quantize(quantum, rounding=ROUND_UP)

# Rounding helpers for amounts and quality
def round_in_min(x: Decimal, *, is_xrp: bool = False) -> Decimal:
    """Minimum IN for a target OUT (round up to the ledger amount grid)."""
    q = XRP_QUANTUM if is_xrp else IOU_QUANTUM
    return quantize_up(clamp_nonneg(x), q)

def round_out_max(x: Decimal, *, is_xrp: bool = False) -> Decimal:
    """Maximum OUT given a budget (round down to the ledger amount grid)."""
    q = XRP_QUANTUM if is_xrp else IOU_QUANTUM
    return quantize_down(clamp_nonneg(x), q)

def quantize_quality(x: Decimal) -> Decimal:
    """Quantise a quality value to the QUALITY_QUANTUM grid (round down)."""
    return quantize_down(x, QUALITY_QUANTUM)

def calc_quality(out_amt: Decimal, in_amt: Decimal) -> Decimal:
    """Return quality = OUT/IN, quantised down to QUALITY_QUANTUM; 0 if IN is 0 or negative."""
    if in_amt <= 0:
        return Decimal(0)
    q = out_amt / in_amt
    return quantize_quality(q)


def quality_bucket(q: Decimal) -> Decimal:
    """Alias: bucket quality by quantising down to QUALITY_QUANTUM."""
    return quantize_quality(q)

# -------------------------------
# Centralised routing/analysis configuration (for reproducibility)
# -------------------------------

from dataclasses import dataclass

@dataclass(frozen=True)
class RoutingConfig:
    """Centralised knobs used by routing and analysis utilities.

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
    # Fibonacci base factor for multi-path AMM OUT cap (see router._amm_out_cap_for_iter)
    fib_base_factor: Decimal = Decimal("1e-4")
    # Default step for alpha scan
    alpha_step_default: Decimal = Decimal("0.05")
    # Defaults for q* search
    qscan_coarse_steps: int = 12
    qscan_refine_iters: int = 18
    # Optional: AMM usage iteration cap (kept here for documentation; enforced in AMMContext)
    amm_max_iters: int = 30


# Module-level default configuration (mutable by advanced users if desired)
ROUTING_CFG = RoutingConfig()

__all__ = [
    "DEFAULT_DECIMAL_PRECISION",
    "DEFAULT_QUANTUM",
    "to_decimal",
    "quantize_down",
    "quantize_up",
    "calc_quality",
    "Segment",
    "IterationMetrics",
    "ExecutionReport",
    "RouteResult",
    "XRP_QUANTUM",
    "IOU_QUANTUM",
    "QUALITY_QUANTUM",
    "clamp_nonneg",
    "round_in_min",
    "round_out_max",
    "quantize_quality",
    "quality_bucket",
    "RoutingConfig", "ROUTING_CFG",
]