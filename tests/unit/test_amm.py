import pytest
from decimal import Decimal

from xrpl_router.amm import AMM
from xrpl_router.core import Quality, XRPAmount, IOUAmount

# Debug helpers for readable output during pytest -s
from fractions import Fraction

from xrpl_router.core.fmt import fmt_dec, amount_to_decimal, quality_rate_to_decimal, quality_price_to_decimal
LOG_PLACES = 6

def _qd(q) -> str:
    return fmt_dec(quality_rate_to_decimal(q), places=LOG_PLACES)

def _pd(q) -> str:
    return fmt_dec(quality_price_to_decimal(q), places=LOG_PLACES)

def frac_str(fr: Fraction) -> str:
    return f"{fr.numerator}/{fr.denominator} (~{fr.numerator/fr.denominator:.12g})"

# Pretty checks for human-friendly test output


def _amm_default():
    # Symmetric IOU/IOU pool, no issuer transfer fees
    # x=1000, y=2000, fee=0.003 (input-side)
    return AMM(
        Decimal("1000"), Decimal("2000"), Decimal("0.003"),
        x_is_xrp=False, y_is_xrp=False,
        tr_in=Decimal("0"), tr_out=Decimal("0"),
    )


def _units(amm: AMM):
    return amm._reserve_units()


def test_spq_non_increasing_after_apply_fill_st():
    amm = _amm_default()
    prev_spq = amm.spq_quality_int().as_fraction()
    print("\n===== AMM_APPLY_FILL =====")

    # Ask for a modest OUT and compute the corresponding IN with the AMM itself
    target_out = IOUAmount.from_components(40, -0)  # 40 units on IOU grid (10^-15 scaled inside)
    dx_needed = amm.swap_in_given_out_st(target_out)
    print(f"    swap_in_given_out_st: dx_needed={fmt_dec(amount_to_decimal(dx_needed), places=LOG_PLACES)}")
    assert not dx_needed.is_zero(), "swap_in_given_out_st returned zero IN"

    # Apply the fill and verify SPQ did not improve
    amm.apply_fill_st(dx_needed, target_out)
    next_spq = amm.spq_quality_int().as_fraction()
    print(f"    prev SPQ={float(prev_spq):.6f}, next SPQ={float(next_spq):.6f}, Δ={float(prev_spq - next_spq):.6f}, monotone={next_spq <= prev_spq}")

    assert next_spq <= prev_spq, (
        f"SPQ improved after fill: prev={prev_spq}, next={next_spq}")


def test_synthetic_segment_anchors_below_lob_top():
    amm = _amm_default()
    spq_now = amm.spq_quality_int().as_fraction()
    print("\n===== AMM_SYNTHETIC_SEGMENT =====")
    q_lob_top = Quality.from_amounts(IOUAmount.from_components(1, 0), IOUAmount.from_components(1, 0))
    q_lob_top_frac = q_lob_top.as_fraction()
    seg = amm.synthetic_segment_for_quality(q_lob_top)
    q_slice_frac = seg.quality.as_fraction() if seg is not None else None
    outd = fmt_dec(amount_to_decimal(seg.out_max), places=LOG_PLACES) if seg is not None else "N/A"
    ind = fmt_dec(amount_to_decimal(seg.in_at_out_max), places=LOG_PLACES) if seg is not None else "N/A"
    print(f"    spq_now={float(spq_now):.6f}, q_lob_top={float(q_lob_top_frac):.6f}")
    print(f"    seg.q={float(q_slice_frac):.6f}, out={outd}, in={ind}")
    print(f"    range_check: lob≤slice≤spq? {q_slice_frac >= q_lob_top_frac and q_slice_frac <= spq_now}")
    assert seg is not None, "Expected a synthetic AMM segment when AMM outruns LOB"
    assert seg.quality.as_fraction() >= q_lob_top_frac
    assert seg.quality.as_fraction() <= spq_now
    amm.apply_fill_st(seg.in_at_out_max, seg.out_max)
    spq_next = amm.spq_quality_int().as_fraction()
    x_u, y_u = amm._reserve_units()
    keep_fee_num = amm._keep_fee_num
    fee_den = amm._fee_den
    from fractions import Fraction
    if x_u > 0:
        tol_y = Fraction(keep_fee_num, x_u * fee_den)
        tol_x = Fraction(y_u * keep_fee_num, (x_u * x_u) * fee_den) if y_u > 0 else Fraction(0, 1)
        tol = tol_y if tol_y >= tol_x else tol_x
    else:
        tol = Fraction(0, 1)
    # Explicitly show SPQ_next ≤ q_lob_top and any overshoot vs tolerance
    over = (spq_next - q_lob_top_frac) if spq_next > q_lob_top_frac else Fraction(0, 1)
    print(f"    check SPQ_next ≤ q_lob_top: {spq_next <= q_lob_top_frac} | next={float(spq_next):.6f}, lob={float(q_lob_top_frac):.6f}")
    print(f"    Δ_over=max(next-lob,0): {float(over):.6g}")
    print(f"    tol≈{float(tol):.3e}, within_tolerance={over <= tol}")
    assert (spq_next <= q_lob_top_frac) or (over <= tol), (
        f"Post-trade SPQ not anchored: next={spq_next}, lob={q_lob_top_frac}, over={over}, tol={tol}")


def test_synthetic_segment_is_not_dust():
    amm = _amm_default()
    q_lob_top = Quality.from_amounts(IOUAmount.from_components(1, 0), IOUAmount.from_components(1, 0))

    seg = amm.synthetic_segment_for_quality(q_lob_top)
    print("\n===== AMM_SYNTHETIC_SEGMENT_NON_DUST =====")
    out_units = amm._amt_to_units_floor(seg.out_max) if seg is not None else 0
    in_units = amm._amt_to_units_floor(seg.in_at_out_max) if seg is not None else 0
    print(f"    out_units={out_units}, in_units={in_units}, non_dust={(out_units>0 and in_units>0)}")
    assert seg is not None
    assert (not seg.out_max.is_zero()) and (not seg.in_at_out_max.is_zero()), "Synthetic segment produced dust"


def test_slice_quality_never_exceeds_current_spq():
    amm = _amm_default()
    spq_now = amm.spq_quality_int().as_fraction()
    print("\n===== AMM_SLICE_QUALITY_VS_SPQ =====")
    q_lob_top = Quality.from_amounts(IOUAmount.from_components(1, 0), IOUAmount.from_components(1, 0))
    q_lob_top_frac = q_lob_top.as_fraction()
    seg = amm.synthetic_segment_for_quality(q_lob_top)
    q_slice_frac = seg.quality.as_fraction() if seg is not None else None
    print(f"    spq_now={float(spq_now):.6f}, slice_q={float(q_slice_frac):.6f}, within_limit={q_slice_frac <= spq_now}")
    assert seg.quality.as_fraction() <= spq_now, (
        f"Slice quality exceeds current SPQ: slice={seg.quality.as_fraction()}, spq={spq_now}")