# tests/unit/test_clob.py
from decimal import Decimal

from xrpl_router.core import STAmount, Quality, ST_MANTISSA_MIN
from xrpl_router.clob import Clob, ClobLevel, normalise_segments
from xrpl_router.book_step import BookStep
from xrpl_router.flow import flow, PaymentSandbox


def iou_units(n: int) -> STAmount:
    # IOU 1 unit = 1e15 mantissa @ exp=-15
    return STAmount.from_components(ST_MANTISSA_MIN * n, -15, 1)


def q_from_ratio(num: int, den: int) -> Quality:
    # Quality = OUT/IN = num/den (IOU units)
    out_u = STAmount.from_components(ST_MANTISSA_MIN * num, -15, 1)
    in_u = STAmount.from_components(ST_MANTISSA_MIN * den, -15, 1)
    return Quality.from_amounts(out_u, in_u)


def test_clob_single_level_basic_flow():
    """
    Single CLOB level: quality=2/1 with max OUT=1000.
    Request OUT=500. Expect positive IN/OUT and OUT ≤ requested.
    """
    # price = IN/OUT = 1/2
    lvl = ClobLevel.from_numbers(price_in_per_out=Decimal("0.5"), out_liquidity=Decimal("1000"))
    clob = Clob([lvl], in_is_xrp=False, out_is_xrp=False)

    # Build integer-domain segments from CLOB levels
    segs = clob.segments()
    segs = normalise_segments(segs)  # Run normalisation once more to ensure clean grids

    # segments_provider for BookStep
    def provider():
        return segs

    # No AMM; no quality floor
    step = BookStep(
        segments_provider=provider,
        limit_quality=None,
        amm_anchor=None,
        amm_curve=None,
        amm_context=None,
        fee_hook=None,
    )

    sb = PaymentSandbox()
    out_req = iou_units(500)

    total_in, total_out = flow(sb, [[step]], out_req)
    assert not total_in.is_zero()
    assert not total_out.is_zero()
    # Must not exceed requested OUT
    assert total_out.to_decimal() <= out_req.to_decimal() + Decimal("1e-15")


def test_clob_multi_levels_with_quality_floor():
    """
    Multiple CLOB levels:
      - Level1: quality=3/2=1.5, OUT=600
      - Level2: quality=4/3≈1.333, OUT=600
    Quality floor = 1.4 → only Level1 should participate.
    """
    lvl1 = ClobLevel.from_numbers(price_in_per_out=Decimal("2") / Decimal("3"), out_liquidity=Decimal("600"))
    lvl2 = ClobLevel.from_numbers(price_in_per_out=Decimal("3") / Decimal("4"), out_liquidity=Decimal("600"))
    clob = Clob([lvl1, lvl2], in_is_xrp=False, out_is_xrp=False)

    segs = normalise_segments(clob.segments())

    def provider():
        return segs

    q_floor = q_from_ratio(3, 2)  # 1.5
    step = BookStep(
        segments_provider=provider,
        limit_quality=q_floor,     # Integer-domain Quality floor
        amm_anchor=None,
        amm_curve=None,
        amm_context=None,
        fee_hook=None,
    )

    sb = PaymentSandbox()
    out_req = iou_units(500)

    total_in, total_out = flow(sb, [[step]], out_req)
    assert not total_in.is_zero()
    assert not total_out.is_zero()
    # OUT must not exceed single level capacity (600) nor requested (500)
    assert total_out.to_decimal() <= out_req.to_decimal() + Decimal("1e-15")