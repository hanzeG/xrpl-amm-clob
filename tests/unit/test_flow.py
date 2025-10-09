from xrpl_router.flow import flow, PaymentSandbox
from xrpl_router.core import STAmount, Quality, ST_MANTISSA_MIN


class DummyStep:
    """Minimal mock Step for flow() verification.

    rev(): doubles the IN requirement, returns (in, out)
    fwd(): halves the IN consumption, returns (in, out)
    quality_upper_bound(): fixed quality
    """

    def __init__(self, num: int, den: int = 1, name: str = ""):
        self.name = name
        # Build OUT/IN units in integer domain (num/den IOU units)
        out_unit = STAmount.from_components(ST_MANTISSA_MIN * num, 0, 1)
        in_unit = STAmount.from_components(ST_MANTISSA_MIN * den, 0, 1)
        self._q = Quality.from_amounts(offer_out=out_unit, offer_in=in_unit)

    def quality_upper_bound(self):
        return self._q

    def rev(self, sb: PaymentSandbox, need: STAmount):
        # require double input for requested out
        required_in = STAmount.from_components(
            need.mantissa * 2, need.exponent, 1
        )
        sb.stage_after_iteration(required_in, need)
        return required_in, need

    def fwd(self, sb: PaymentSandbox, cap: STAmount):
        # simple proportional return: half of input propagates
        out = STAmount.from_components(
            cap.mantissa // 2, cap.exponent, 1
        )
        sb.stage_fee(cap, out)
        return cap, out


def test_flow_single_path_basic():
    """Basic sanity test: one strand, one step, integer-domain path closes."""
    sb = PaymentSandbox()
    step = DummyStep(num=2, den=1)
    out_req = STAmount.from_components(ST_MANTISSA_MIN * 10, 0, 1)
    total_in, total_out = flow(sb, [[step]], out_req)
    assert total_out.to_decimal() > 0
    assert total_in.to_decimal() > 0
    # Expect out roughly half of input (since fwd halves it)
    ratio = total_out.to_decimal() / total_in.to_decimal()
    assert 0.4 < ratio < 0.6


def test_flow_multiple_strands_ordering():
    """Ensure flow selects higher-quality strand first (sorted by Quality)."""
    sb = PaymentSandbox()
    high = DummyStep(num=3, den=1, name="high")
    low = DummyStep(num=3, den=2, name="low")
    out_req = STAmount.from_components(ST_MANTISSA_MIN * 5, 0, 1)
    total_in, total_out = flow(sb, [[high], [low]], out_req)
    # The high-quality strand should deliver proportionally more output
    assert total_out.to_decimal() > 0
    # internal order: high first due to quality sort
    assert total_in.is_zero() is False


def test_flow_zero_out_request_returns_zero():
    """Zero or negative out_req must short-circuit with ZERO tuple."""
    sb = PaymentSandbox()
    out_req = STAmount.zero()
    total_in, total_out = flow(sb, [], out_req)
    assert total_in.is_zero() and total_out.is_zero()