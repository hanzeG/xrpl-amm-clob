"""
Ordering utilities (quality-based ranking, guardrails, and floors) aligned
with XRPL semantics (non-negative domain, explicit guard behaviour).

Key behaviours:
- Sort by bucketed quality (higher is better), with stable tie-breaking.
- Apply a quality floor: drop segments strictly worse than a given floor.
- Instantaneous quality guardrail: if an (out_take / in_ceiled) ratio would
  be strictly better (higher) than the segment's quoted slice quality, bump IN by
  the minimal step on the current grid so the implied ratio is ≤ quoted.

Notes:
- Uses integer fixed-point IOUAmount and Quality; no Decimal.
- XRP guard operates on the drops grid (integers) and ceilings to whole drops.
- For XRP-side IN amounts (drops), a dedicated guard is provided: 
    guard_instant_quality_xrp(out_take, in_drops, slice_quality).
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, TypeVar

from .amounts import IOUAmount, XRPAmount, _ceil_div
from .exc import AmountDomainError, InvariantViolation
from .quality import Quality

# Debug printing control
DEBUG_ORDERING = False

def _dbg(msg: str) -> None:
    if DEBUG_ORDERING:
        print(msg)

def guard_instant_quality_xrp(
    out_take: IOUAmount,
    in_drops: XRPAmount,
    slice_quality: Quality,
) -> XRPAmount:
    """Closed-form XRP-side guard: return minimal drops so implied ≤ quoted.

    Preconditions:
    - in_drops is an XRPAmount with value >= 0
    - out_take > 0
    - slice_quality.rate > 0
    """
    if in_drops.value < 0:
        raise AmountDomainError("in_drops must be >= 0")
    if out_take.is_zero():
        return in_drops

    start = in_drops.value

    implied = Quality.from_amounts(offer_out=out_take, offer_in=in_drops)
    if not (implied.rate > slice_quality.rate):
        return in_drops

    # Compute minimal IN on the drops grid (exp = 0) so that out/in <= quoted rate.
    # Let: out = m_o * 10^{e_o}, rate = m_q * 10^{e_q}, in = D * 10^0 (drops)
    # Then D >= ceil( m_o * 10^{e_o - e_q} / m_q ).
    m_o, e_o = out_take.mantissa, out_take.exponent
    m_q, e_q = slice_quality.rate.mantissa, slice_quality.rate.exponent
    shift = e_o - e_q
    if shift >= 0:
        num = m_o * (10 ** shift)
        den = m_q
    else:
        num = m_o
        den = m_q * (10 ** (-shift))

    target_drops = _ceil_div(num, den)

    # Never reduce caller-provided ceiling
    if target_drops < start:
        target_drops = start

    # Post-check: implied(candidate) must be ≤ quoted
    cand_amt = XRPAmount(target_drops)
    implied_cand = Quality.from_amounts(offer_out=out_take, offer_in=cand_amt)
    if implied_cand.rate > slice_quality.rate:
        raise InvariantViolation(f"guard_instant_quality_xrp: implied quality still better than quoted after guard; out=({m_o},{e_o}), q=({m_q},{e_q}), in_drops={target_drops}")

    return XRPAmount(target_drops)

T = TypeVar("T")


# ----------------------------
# Internal helpers
# ----------------------------

def _implied_quality(out_take: IOUAmount, in_amt: IOUAmount) -> Quality:
    """Compute implied quality as out/in (higher is better)."""
    return Quality.from_amounts(offer_out=out_take, offer_in=in_amt)


# ----------------------------
# Closed-form helpers
# ----------------------------

def _compute_min_in_on_grid(out_take: IOUAmount, rate: Quality, exp: int) -> IOUAmount:
    """Closed-form minimal IN on a given exponent grid so that out/in <= rate.

    Using canonical values (no extra offsets):
      rate = m_q * 10^{e_q}
      out  = m_o * 10^{e_o}
      IN'  = m_i * 10^{exp}
    Choose IN' exponent = `exp`; then:
      m_i >= ceil( m_o * 10^{e_o - e_q - exp} / m_q )
    Finally normalise via IOUAmount.from_components.
    """
    if out_take.is_zero():
        # Minimal IN to satisfy inequality is zero; caller typically handles zero earlier.
        return IOUAmount.zero()

    m_o, e_o = out_take.mantissa, out_take.exponent
    m_q, e_q = rate.rate.mantissa, rate.rate.exponent
    shift = e_o - e_q - exp
    if shift >= 0:
        num = m_o * (10 ** shift)
        den = m_q
    else:
        num = m_o
        den = m_q * (10 ** (-shift))

    m_needed = _ceil_div(num, den)
    return IOUAmount.from_components(m_needed, exp)


# ----------------------------
# Public guardrail
# ----------------------------

def guard_instant_quality(
    out_take: IOUAmount,
    in_ceiled: IOUAmount,
    slice_quality: Quality,
) -> IOUAmount:
    """Closed-form: ensure implied instant quality does not beat the quoted slice quality."""
    if in_ceiled.is_zero():
        # No input budget -> nothing to guard; caller may handle this earlier.
        return in_ceiled
    if out_take.is_zero():
        return in_ceiled

    implied = _implied_quality(out_take, in_ceiled)
    if not (implied.rate > slice_quality.rate):
        return in_ceiled

    # Compute minimal IN on the current grid that enforces implied <= quoted
    m_o, e_o = out_take.mantissa, out_take.exponent
    m_q, e_q = slice_quality.rate.mantissa, slice_quality.rate.exponent
    candidate = _compute_min_in_on_grid(out_take, slice_quality, in_ceiled.exponent)

    # Respect caller's ceiling (never reduce IN)
    if candidate < in_ceiled:
        candidate = in_ceiled

    # Post-check: implied(candidate) must be ≤ quoted
    implied_cand = _implied_quality(out_take, candidate)
    if implied_cand.rate > slice_quality.rate:
        raise InvariantViolation(f"guard_instant_quality: implied quality still better than quoted after guard; out=({m_o},{e_o}), q=({m_q},{e_q}), in_candidate=({candidate.mantissa},{candidate.exponent})")

    return candidate


# ----------------------------
# Ordering: floor + stable sort
# ----------------------------

def apply_quality_floor(
    items: Iterable[T],
    *,
    qmin: Optional[Quality],
    get_quality: Callable[[T], Quality],
) -> List[T]:
    """Filter out items strictly worse than floor (lower quality value).

    Higher is better. Keep `item` iff `get_quality(item) >= qmin`.
    If `qmin` is None, return all items as list.
    """
    if qmin is None:
        return list(items)
    return [x for x in items if get_quality(x).rate >= qmin.rate]


def stable_sort_by_quality(
    items: Iterable[T],
    *,
    get_quality: Callable[[T], Quality],
) -> List[T]:
    """Stable sort by quality (descending, higher is better).

    Python's built-in sort is stable, so items with equal quality retain their
    original insertion order.
    """
    lst = list(items)
    return sorted(lst, key=lambda x: get_quality(x), reverse=True)


def prepare_and_order(
    items: Iterable[T],
    *,
    qmin: Optional[Quality],
    get_quality: Callable[[T], Quality],
) -> List[T]:
    """Apply quality floor, then stable sort by quality.

    Guardrails (instant quality bump) are expected to be applied by the caller
    per-fill since they depend on `out_take` and provisional `in_ceiled` for
    each slice.

    Python's built-in sort is stable, so items with equal quality retain their
    original insertion order.
    """
    filtered = apply_quality_floor(items, qmin=qmin, get_quality=get_quality)
    return stable_sort_by_quality(filtered, get_quality=get_quality)


__all__ = [
    "guard_instant_quality",
    "guard_instant_quality_xrp",
    "apply_quality_floor",
    "stable_sort_by_quality",
    "prepare_and_order",
]