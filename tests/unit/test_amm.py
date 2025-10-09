# tests/integration/test_flow_with_amm.py
from decimal import Decimal
from typing import Tuple

from xrpl_router.flow import flow, PaymentSandbox
from xrpl_router.core import STAmount, Quality, ST_MANTISSA_MIN
from xrpl_router.steps import Step
from xrpl_router.amm import AMM


class MinimalAMMStep(Step):
    """
    Minimal AMM-backed Step to validate integer-domain flow():
    - Orientation: AMM inputs are X, outputs are Y (same as AMM class).
    - rev(): compute min IN for requested OUT using AMM.swap_in_given_out, stage to sandbox.
    - fwd(): compute OUT for IN capacity using AMM.swap_out_given_in, stage to sandbox.
    - quality_upper_bound(): construct Quality from AMM SPQ.
    Note: AMM keeps Decimal maths internally (by design for now); Step bridges at the boundary.
    """

    def __init__(self, amm: AMM, name: str = "amm-step"):
        self.amm = amm
        self.name = name
        self._cached_in = STAmount.zero()
        self._cached_out = STAmount.zero()

    def _st_iou_units(self, units: int) -> STAmount:
        # Helper: units in IOU grid (mantissa=k*1e15, exponent=-15)
        return STAmount.from_components(ST_MANTISSA_MIN * units, -15, 1)

    def quality_upper_bound(self) -> Quality:
        q_dec = self.amm.spq()  # Decimal
        if q_dec <= 0:
            # Return zero quality
            return Quality.from_amounts(STAmount.zero(), self._st_iou_units(1))
        out_1 = STAmount.from_decimal(q_dec)  # OUT per 1 IN
        in_1 = self._st_iou_units(1)
        return Quality.from_amounts(out_1, in_1)

    def rev(self, sandbox: PaymentSandbox, out_req: STAmount) -> Tuple[STAmount, STAmount]:
        # Requested OUT is on Y side
        if out_req.is_zero():
            self._cached_in = STAmount.zero()
            self._cached_out = STAmount.zero()
            return self._cached_in, self._cached_out

        # Convert to Decimal for AMM
        dy_req_dec = out_req.to_decimal()
        try:
            dx_need_dec = self.amm.swap_in_given_out(dy_req_dec)
        except Exception:
            dx_need_dec = Decimal(0)

        if dx_need_dec <= 0:
            self._cached_in = STAmount.zero()
            self._cached_out = STAmount.zero()
            return self._cached_in, self._cached_out

        # Recompute achievable OUT for safety (fees/grids)
        dy_got_dec = self.amm.swap_out_given_in(dx_need_dec)
        if dy_got_dec <= 0:
            self._cached_in = STAmount.zero()
            self._cached_out = STAmount.zero()
            return self._cached_in, self._cached_out

        # Cap to out_req
        if dy_got_dec > dy_req_dec:
            dy_got_dec = dy_req_dec

        dx_need = STAmount.from_decimal(dx_need_dec)
        dy_got = STAmount.from_decimal(dy_got_dec)

        # Stage to sandbox (router convention: stage_after_iteration records (dx, dy))
        sandbox.stage_after_iteration(dx_need, dy_got)

        self._cached_in = dx_need
        self._cached_out = dy_got
        return self._cached_in, self._cached_out

    def fwd(self, sandbox: PaymentSandbox, in_cap: STAmount) -> Tuple[STAmount, STAmount]:
        # Forward must consume the reverse cache when possible to stay consistent with flow()
        ZERO = STAmount.zero()
        if in_cap.is_zero():
            self._cached_in = ZERO
            self._cached_out = ZERO
            return self._cached_in, self._cached_out

        # If reverse computed a feasible pair (cached_in, cached_out), prefer consuming it.
        if (not self._cached_in.is_zero()) and (not self._cached_out.is_zero()):
            # Case A: caller allows at least the cached IN → execute cached pair
            if (in_cap.mantissa == self._cached_in.mantissa and in_cap.exponent == self._cached_in.exponent and in_cap.sign == self._cached_in.sign) or (
                (in_cap.exponent == self._cached_in.exponent and in_cap.mantissa > self._cached_in.mantissa) or
                (in_cap.exponent > self._cached_in.exponent)
            ):
                in_spent = self._cached_in
                out_got = self._cached_out
                sandbox.stage_after_iteration(in_spent, out_got)
                return in_spent, out_got
            # Case B: smaller cap → scale proportionally (best-effort) to avoid zeroing out
            try:
                cap_dec = in_cap.to_decimal()
                cin_dec = self._cached_in.to_decimal()
                cout_dec = self._cached_out.to_decimal()
                if cin_dec > 0:
                    dy_dec = (cap_dec * cout_dec) / cin_dec
                else:
                    dy_dec = Decimal(0)
            except Exception:
                dy_dec = Decimal(0)
            if dy_dec <= 0:
                self._cached_in = ZERO
                self._cached_out = ZERO
                return self._cached_in, self._cached_out
            in_spent = in_cap
            out_got = STAmount.from_decimal(dy_dec)
            sandbox.stage_after_iteration(in_spent, out_got)
            self._cached_in = in_spent
            self._cached_out = out_got
            return in_spent, out_got

        # Fallback: no reverse cache available → compute directly via AMM
        dx_cap_dec = in_cap.to_decimal()
        dy_dec = self.amm.swap_out_given_in(dx_cap_dec)
        if dy_dec <= 0:
            self._cached_in = ZERO
            self._cached_out = ZERO
            return self._cached_in, self._cached_out
        dy = STAmount.from_decimal(dy_dec)
        sandbox.stage_after_iteration(in_cap, dy)
        self._cached_in = in_cap
        self._cached_out = dy
        return self._cached_in, self._cached_out


# ---------------------------
# Helpers to build STAmount
# ---------------------------
def iou_units(n: int) -> STAmount:
    return STAmount.from_components(ST_MANTISSA_MIN * n, -15, 1)


# ===========================
# Tests
# ===========================

def test_flow_with_amm_iou_iou():
    """
    IOU→IOU AMM:
    - Pool: x=1,000,000 IOU, y=500,000 IOU, fee=0.3%
    - Request: out_req = 1,000 IOU on Y
    Expect: positive IN/OUT, and OUT ≤ requested.
    """
    amm = AMM(
        x_reserve=Decimal("1000000"),
        y_reserve=Decimal("500000"),
        fee=Decimal("0.003"),
        x_is_xrp=False,
        y_is_xrp=False,
    )
    step = MinimalAMMStep(amm)
    sb = PaymentSandbox()

    out_req = iou_units(1000)
    total_in, total_out = flow(sb, [[step]], out_req)

    assert not total_in.is_zero()
    assert not total_out.is_zero()
    # OUT cannot exceed requested in our step design
    assert (total_out.to_decimal() - out_req.to_decimal()) <= Decimal("1e-15")


def test_flow_with_amm_xrp_iou():
    """
    XRP→IOU AMM:
    - Pool: x=2,000,000 XRP, y=1,000,000 IOU, fee=0.2%
    - Request: out_req = 2,000 IOU on Y
    Expect: positive IN/OUT, OUT ≤ requested.
    """
    amm = AMM(
        x_reserve=Decimal("2000000"),
        y_reserve=Decimal("1000000"),
        fee=Decimal("0.002"),
        x_is_xrp=True,   # X is XRP
        y_is_xrp=False,  # Y is IOU
    )
    step = MinimalAMMStep(amm)
    sb = PaymentSandbox()

    out_req = iou_units(2000)
    total_in, total_out = flow(sb, [[step]], out_req)

    assert not total_in.is_zero()
    assert not total_out.is_zero()
    assert (total_out.to_decimal() - out_req.to_decimal()) <= Decimal("1e-15")