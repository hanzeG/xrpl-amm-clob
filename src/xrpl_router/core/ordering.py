"""
Ordering utilities (quality-based ranking, guardrails, and floors) aligned
with XRPL semantics.

Key behaviours:
- Sort by bucketed quality (higher is better), with stable tie-breaking.
- Apply a quality floor: drop segments strictly worse than a given floor.
- Instantaneous quality guardrail: if an (out_take / in_ceiled) ratio would
  be strictly better (higher) than the segment's quoted slice quality, bump IN by
  one native step until the implied ratio is not better (i.e., ≤ quoted). This mirrors XRPL's
  "no better than quoted" behaviour.

Notes:
- Uses only integer fixed-point `STAmount` and `Quality`; no Decimal.
- Instantaneous guard uses a closed-form minimal-IN computation (no iterative bumping).
- XRP guard operates on the drops grid (exp = -15) and ceilings to whole drops.
- For XRP-side IN amounts (drops), a dedicated guard is provided: 
  	guard_instant_quality_xrp(out_take, in_drops, slice_quality).
"""

from __future__ import annotations

from functools import cmp_to_key
from typing import Callable, Iterable, List, Optional, TypeVar

from .amounts import STAmount, st_from_drops
from .constants import ST_MANTISSA_MIN
from .quality import Quality
def guard_instant_quality_xrp(
    out_take: STAmount,
    in_drops: int,
    slice_quality: Quality,
) -> int:
    """Closed-form XRP-side guard: return minimal drops so implied ≤ quoted."""
    # Normalise negatives to zero drops at the boundary.
    if out_take.is_zero():
        return in_drops if in_drops > 0 else 0

    start = in_drops if in_drops > 0 else 0
    in_amt = st_from_drops(start)

    implied = Quality.from_amounts(offer_out=out_take, offer_in=in_amt)
    if not (implied.rate > slice_quality.rate):
        return start

    # Compute minimal IN amount on the XRP grid (exp = -15)
    target_amt = _compute_min_in_on_grid(out_take, slice_quality, exp=-15)
    # exp=-15 is the drops grid (1 drop = 10^-6 XRP); conversion to whole drops is handled below.

    # Convert canonical IOU fixed-point back to whole-drop integers with ceiling.
    # '+15' aligns the canonical mantissa scale (~1e15) to the integer drops grid (1 drop = 1e-6 XRP).
    shift = target_amt.exponent + 15  # scale to the drops grid
    if shift >= 0:
        num = target_amt.mantissa * (10 ** shift)
        den = ST_MANTISSA_MIN
    else:
        num = target_amt.mantissa
        den = ST_MANTISSA_MIN * (10 ** (-shift))

    target_drops = _ceil_div(num, den)

    # Never reduce below the caller-provided ceiling
    return target_drops if target_drops > start else start

T = TypeVar("T")


# ----------------------------
# Internal helpers
# ----------------------------





def _implied_quality(out_take: STAmount, in_amt: STAmount) -> Quality:
    """Compute implied quality as out/in (higher is better)."""
    return Quality.from_amounts(offer_out=out_take, offer_in=in_amt)


# ----------------------------
# Closed-form helpers
# ----------------------------

def _ceil_div(a: int, b: int) -> int:
    """Ceiling integer division for non-negative integers."""
    if b <= 0:
        raise ValueError("divisor must be positive")
    if a <= 0:
        return 0
    return -(-a // b)


def _compute_min_in_on_grid(out_take: STAmount, rate: Quality, exp: int) -> STAmount:
    """Closed-form minimal IN on a given exponent grid so that out/in <= rate.

    Using canonical values (no extra offsets):
      rate = m_q * 10^{e_q}
      out  = m_o * 10^{e_o}
      IN'  = m_i * 10^{exp}
    Choose IN' exponent = `exp`; then:
      m_i >= ceil( m_o * 10^{e_o - e_q - exp} / m_q )
    Finally normalise via STAmount.from_components.
    """
    if out_take.is_zero():
        return STAmount.from_components(ST_MANTISSA_MIN, exp, 1)

    m_o, e_o = out_take.mantissa, out_take.exponent
    m_q, e_q = rate.rate.mantissa, rate.rate.exponent
    # STAmount numeric value is mantissa * 10^exponent; rearrangement gives shift = e_o - e_q - exp.
    shift = e_o - e_q - exp
    if shift >= 0:
        num = m_o * (10 ** shift)
        den = m_q
    else:
        num = m_o
        den = m_q * (10 ** (-shift))

    m_needed = _ceil_div(num, den)
    # Positive IN by construction. Normalisation handled by constructor.
    return STAmount.from_components(m_needed, exp, 1)


# ----------------------------
# Public guardrail
# ----------------------------

def guard_instant_quality(
    out_take: STAmount,
    in_ceiled: STAmount,
    slice_quality: Quality,
) -> STAmount:
    """Closed-form: ensure implied instant quality does not beat the quoted slice quality."""
    if out_take.is_zero():
        return in_ceiled

    implied = _implied_quality(out_take, in_ceiled)
    if not (implied.rate > slice_quality.rate):
        # Already not better than quoted (i.e. <= quoted)
        return in_ceiled

    # Compute minimal IN on the current grid that enforces implied <= quoted
    candidate = _compute_min_in_on_grid(out_take, slice_quality, in_ceiled.exponent)

    # Respect caller's ceiling (never reduce IN)
    if candidate < in_ceiled:
        return in_ceiled
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


def _cmp_quality_then_index(
    a: T,
    b: T,
    get_quality: Callable[[T], Quality],
    get_index: Callable[[T], int],
) -> int:
    qa = get_quality(a)
    qb = get_quality(b)
    if qa.rate > qb.rate:
        return -1
    if qa.rate < qb.rate:
        return 1
    ia = get_index(a)
    ib = get_index(b)
    if ia < ib:
        return -1
    if ia > ib:
        return 1
    return 0


def stable_sort_by_quality(
    items: Iterable[T],
    *,
    get_quality: Callable[[T], Quality],
    get_index: Callable[[T], int],
) -> List[T]:
    """Stable sort by quality (descending, higher is better),
    with insertion-index tie-breaker.
    """
    lst = list(items)
    key = cmp_to_key(lambda a, b: _cmp_quality_then_index(a, b, get_quality, get_index))
    return sorted(lst, key=key)


def prepare_and_order(
    items: Iterable[T],
    *,
    qmin: Optional[Quality],
    get_quality: Callable[[T], Quality],
    get_index: Callable[[T], int],
) -> List[T]:
    """Apply quality floor, then stable sort by quality.

    Guardrails (instant quality bump) are expected to be applied by the caller
    per-fill since they depend on `out_take` and provisional `in_ceiled` for
    each slice.
    """
    filtered = apply_quality_floor(items, qmin=qmin, get_quality=get_quality)
    return stable_sort_by_quality(filtered, get_quality=get_quality, get_index=get_index)


__all__ = [
    "guard_instant_quality",
    "guard_instant_quality_xrp",
    "apply_quality_floor",
    "stable_sort_by_quality",
    "prepare_and_order",
]