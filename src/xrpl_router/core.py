"""
Numeric primitives for XRPL AMM/CLOB routing experiments.
Centralises Decimal policy (precision, quantum, rounding) so AMM, CLOB, and the router share identical arithmetic.

Whitepaper alignment (for cross‑reference):
- §1.3.2  Reverse→Forward execution and limiting‑step replay — these helpers ensure price/amount rounding is consistent across reverse pass estimates and forward replays.
- §1.2.7.3 Multi‑path AMM behaviour (Fibonacci slicing & bounded iterations) — enforcement lives in AMMContext/AMMLiquidity; router no longer enforces caps.

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
    """A homogeneous quote slice used by the router.
    
    Notes:
    - `quality` is the **bucketed** quality (quantised to QUALITY_QUANTUM) for tiering/sorting.
    - `raw_quality` preserves the **unbucketed** quality for precise equality/priority checks
      (e.g., “AMM == CLOB” → prefer CLOB per whitepaper §1.2.7.2).
    - Fees on the OUT side are *not* deducted here (taker-view OUT).
      OUT-side issuer fee is accounted during execution (ownerGives) per §1.3.2.4.
    """
    src: Literal["AMM", "CLOB"]
    quality: Decimal            # OUT / IN (higher is better) — bucketed
    out_max: Decimal            # Max OUT available on this slice
    in_at_out_max: Decimal      # IN required to consume out_max
    in_is_xrp: bool             # Input asset uses XRP grid (drops)
    out_is_xrp: bool            # Output asset uses XRP grid (drops)
    raw_quality: Optional[Decimal] = None  # Unbucketed OUT/IN for precise comparisons
    source_id: Optional[str] = None        # Optional external identifier (e.g., offer id / pool tag)


# --- Per-iteration and aggregate execution metrics ---
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
XRP_QUANTUM: Decimal = Decimal("1e-6")      # drops (1e-6 XRP)
IOU_QUANTUM: Decimal = Decimal("1e-15")     # IOU amounts
QUALITY_QUANTUM: Decimal = Decimal("1e-15") # quality grid
DEFAULT_QUANTUM: Decimal = IOU_QUANTUM

# Epsilon constants (kept at 0 to preserve existing comparisons; may be raised by callers)
INST_EPS: Decimal = Decimal("0")     # Instantaneous guards / step filters
PRICE_EPS: Decimal = Decimal("0")    # Price equality / root-finding tolerances

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


def compose_path_quality(step_qualities: List[Decimal]) -> Decimal:
    """Compose a path/strand quality as the product of step qualities.
    
    Returns 0 if any step quality is non-positive. The result is quantised
    down to QUALITY_QUANTUM, matching whitepaper §1.3.2 path quality usage.
    """
    q = Decimal(1)
    for s in step_qualities:
        if s <= 0:
            return Decimal(0)
        q *= s
    return quantize_quality(q)


def quality_bucket(q: Decimal) -> Decimal:
    """Alias: bucket quality by quantising down to QUALITY_QUANTUM."""
    return quantize_quality(q)

# -------------------------------
# Internal helpers shared by router (no behaviour change; extracted to deduplicate)
# -------------------------------

def _guard_inst_quality(
    out_take: Decimal,
    in_ceiled: Decimal,
    slice_quality: Decimal,
    *,
    in_is_xrp: bool,
) -> Decimal:
    """Instantaneous quality guard used when taking a slice.

    Ensures that after rounding the IN amount up to ledger grid, the effective
    quality (out_take / in_ceiled) does not exceed the advertised slice quality.

    Behaviour is intentionally identical to the repeated inline patterns in the
    router: if the rounded IN would cause an *improved* quality, we bump IN by
    one quantum and round again. No epsilon is applied (INST_EPS=0) to preserve
    current semantics.
    """
    if in_ceiled <= 0:
        return Decimal(0)
    eff_q = out_take / in_ceiled
    if eff_q <= slice_quality:
        return in_ceiled
    bump = XRP_QUANTUM if in_is_xrp else IOU_QUANTUM
    return round_in_min(in_ceiled + bump, is_xrp=in_is_xrp)


def _ensure_stable_order_ids(
    segs: List[Segment], order_map_id: Dict[int, int]
) -> None:
    """Ensure each Segment object has a stable insertion index keyed by its id().

    This avoids collisions when two distinct Segment instances have identical
    field values (dataclass equality), which can otherwise merge entries when
    using the Segment itself as a dict key.
    """
    for s in segs:
        sid = id(s)
        if sid not in order_map_id:
            order_map_id[sid] = len(order_map_id)


def _sort_by_bucket_stable(
    segs: List[Segment], order_map_id: Dict[int, int]
) -> None:
    """Sort segments by (quality bucket, stable insertion order), best first.

    Quality bucket uses quantised-down QUALITY_QUANTUM. Stable order is the
    negative of insertion index so earlier entries win ties when reverse=True.
    """
    # Make sure all present segments have an order id
    _ensure_stable_order_ids(segs, order_map_id)
    segs.sort(
        key=lambda s: (
            quality_bucket(s.quality),
            -order_map_id.get(id(s), 0),
        ),
        reverse=True,
    )


def _apply_quality_floor(
    segs: List[Segment], qmin_bucket: Optional[Decimal]
) -> List[Segment]:
    """Filter segments by a quantised quality floor.

    The floor is specified in bucket space (already quantised). If None, the
    list is returned unchanged. The returned list is a new list; the input list
    is not modified.
    """
    if qmin_bucket is None:
        return segs
    return [s for s in segs if quality_bucket(s.quality) >= qmin_bucket]

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
    - `fib_base_factor` → §1.2.7.3 (Fibonacci slicing in multi‑path AMM). Used by AMMLiquidity to size AMM slices when multi‑path is active; AMMContext enforces the ≤30 iteration cap.
    - `alpha_step_default` → α‑scan granularity when searching for the optimal split ratio α* (task spec: Execution Efficiency).
    - `qscan_coarse_steps` / `qscan_refine_iters` → scanning and bisection depth for the crossover size q* between AMM‑only and CLOB‑only costs.
    - `amm_max_iters` → documented upper bound for AMM participation count (whitepaper: capped iterations); enforcement lives in AMMContext/AMMLiquidity.

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
    "compose_path_quality",
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