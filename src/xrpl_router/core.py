"""
Numeric primitives for XRPL AMM/CLOB routing experiments.
Centralises Decimal policy (precision, quantum, rounding) so AMM, CLOB, and the router share identical arithmetic.
Conservative emulation; this is not a bit-exact reproduction of rippled/STAmount.
"""
from __future__ import annotations

from decimal import Decimal, getcontext, ROUND_DOWN, ROUND_UP
from typing import Union
from dataclasses import dataclass
from typing import Literal, Dict, Any, List

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


@dataclass(frozen=True)
class RouteResult:
    """Router outcome with totals and step-by-step trace."""
    filled_out: Decimal
    spent_in: Decimal
    avg_quality: Decimal
    usage: Dict[str, Decimal]          # OUT consumed per source
    trace: List[Dict[str, Any]]        # Iteration records

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
# Helpers
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
    """Minimum IN for a target OUT (round up to amount grid)."""
    q = XRP_QUANTUM if is_xrp else IOU_QUANTUM
    return quantize_up(clamp_nonneg(x), q)

def round_out_max(x: Decimal, *, is_xrp: bool = False) -> Decimal:
    """Maximum OUT given a budget (round down to amount grid)."""
    q = XRP_QUANTUM if is_xrp else IOU_QUANTUM
    return quantize_down(clamp_nonneg(x), q)

def quantize_quality(x: Decimal) -> Decimal:
    """Quantise quality to the quality grid (round down)."""
    return quantize_down(x, QUALITY_QUANTUM)

def calc_quality(out_amt: Decimal, in_amt: Decimal,
                 *, quantum: Decimal = DEFAULT_QUANTUM) -> Decimal:
    """Return quality = OUT/IN quantised down; 0 if IN is 0."""
    if in_amt == 0:
        return Decimal(0)
    q = out_amt / in_amt
    return quantize_down(q, QUALITY_QUANTUM)

__all__ = [
    "DEFAULT_DECIMAL_PRECISION",
    "DEFAULT_QUANTUM",
    "to_decimal",
    "quantize_down",
    "quantize_up",
    "calc_quality",
    "Segment",
    "RouteResult",
    "XRP_QUANTUM",
    "IOU_QUANTUM",
    "QUALITY_QUANTUM",
    "clamp_nonneg",
    "round_in_min",
    "round_out_max",
    "quantize_quality",
]