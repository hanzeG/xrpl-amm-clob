"""AMM liquidity routing helpers matching the whitepaper (1.2.7).

Provides:
 - AMMOffer: lightweight AMM offer descriptor (for router diagnostics).
 - AMMLiquidity: helper with get_offer supporting three modes:
    * "single" (1.2.7.1): exact swap for requested in/out.
    * "anchored" (1.2.7.2): anchored to CLOB quality, only if SPQ > lob_quality.
    * "multi" (1.2.7.3): multi-path Fibonacci slicing, respecting MAX_AMM_ITERS.

IMPORTANT: In multi-path mode (§1.2.7.3), Fibonacci sizing is applied on the taker-pays base. Quality anchoring and CLOB-priority tie-break are enforced at the BookStep layer; this helper only computes candidate slices.
This helper enforces the iteration cap but leaves incrementing of AMM-used iterations to the caller once a slice is actually consumed.
"""

from __future__ import annotations
from decimal import Decimal
from typing import Optional, List, Literal

from .amm import AMM
# NOTE: AMM exposes integer-domain swap/apply interfaces; Decimal may still appear for sizing heuristics.
from .core.datatypes import Segment
from .core.amounts import STAmount
from .core.quality import Quality
from .amm_context import MAX_AMM_ITERS

def fibonacci(n: int) -> int:
    """Returns the nth Fibonacci number (1-indexed: F(1)=1, F(2)=1, F(3)=2, ...)."""
    if n <= 0:
        return 0
    a, b = 1, 1
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b

class AMMOffer:
    """Lightweight AMM offer descriptor (for router diagnostics)."""
    def __init__(self, amm: AMM):
        self.x = amm.x
        self.y = amm.y
        self.fee = amm.fee
        self.x_is_xrp = amm.x_is_xrp
        self.y_is_xrp = amm.y_is_xrp
        self.tr_in = amm.tr_in
        self.tr_out = amm.tr_out
        # spot price quality (SPQ)
        self.spq = amm.spq()

class AMMLiquidity:
    """AMM liquidity helper for router (whitepaper 1.2.7)."""
    def __init__(self, amm: AMM, amm_context):
        """
        amm: AMM pool object.
        amm_context: context with .ammUsedIters (int) and .maxItersReached() method.
        This helper **does not** call setAMMUsed(); callers should increment the
        counter only when AMM liquidity is actually consumed in a router iteration.
        The iteration cap is enforced using MAX_AMM_ITERS from amm_context.
        """
        self.amm = amm
        self.amm_context = amm_context

    def get_offer(
        self,
        mode: Literal["single", "anchored", "multi"],
        *,
        in_amount: Optional[STAmount] = None,
        out_amount: Optional[STAmount] = None,
        lob_quality: Optional[Quality] = None,
        max_out_cap: Optional[STAmount] = None,
    ) -> Optional[Segment]:
        """
        Returns a Segment representing an AMM liquidity slice according to routing mode.
        Modes:
        - "single": exact swap for in_amount or out_amount (1.2.7.1).
        - "anchored": anchored to lob_quality (1.2.7.2); returns None if SPQ <= lob_quality.
        - "multi": Fibonacci multi-path slicing (1.2.7.3), using taker-pays base.
        Returns None if not feasible or not allowed.
        Both `quality` and `raw_quality` fields are filled.
        """
        if mode == "single":
            # Single mode: exact swap, using in_amount or out_amount (integer domain at boundaries, Decimal internally)
            if in_amount is not None:
                out_st = self.amm.swap_out_given_in_st(in_amount)
                if out_st.is_zero():
                    return None
                q = Quality.from_amounts(out_st, in_amount)
                return Segment(
                    src="AMM",
                    quality=q,
                    out_max=out_st,
                    in_at_out_max=in_amount,
                    in_is_xrp=self.amm.x_is_xrp,
                    out_is_xrp=self.amm.y_is_xrp,
                    raw_quality=q,
                )
            elif out_amount is not None:
                in_st = self.amm.swap_in_given_out_st(out_amount)
                if in_st.is_zero():
                    return None
                out_got_st = self.amm.swap_out_given_in_st(in_st)
                if out_got_st.is_zero():
                    return None
                # Cap to requested out (integer-domain comparison)
                out_st = out_amount if out_got_st > out_amount else out_got_st
                q = Quality.from_amounts(out_st, in_st)
                return Segment(
                    src="AMM",
                    quality=q,
                    out_max=out_st,
                    in_at_out_max=in_st,
                    in_is_xrp=self.amm.x_is_xrp,
                    out_is_xrp=self.amm.y_is_xrp,
                    raw_quality=q,
                )
            else:
                return None

        elif mode == "anchored":
            # Anchored mode: anchored to lob_quality (Quality), only if SPQ > lob_quality
            if lob_quality is None or lob_quality.rate.sign <= 0:
                return None
            # Integer-domain SPQ quality (no Decimal bridge)
            spq_q = self.amm.spq_quality_int()
            if spq_q.rate <= lob_quality.rate:
                # At or below CLOB quality, do not return AMM slice (CLOB preferred)
                return None
            cap_st = max_out_cap if max_out_cap is not None else out_amount
            seg = self.amm.synthetic_segment_for_quality(lob_quality, max_out_cap=cap_st)
            if seg is None:
                return None
            # Safety: do not return a slice better than the LOB threshold
            if seg.quality.rate > lob_quality.rate:
                return None
            return seg

        elif mode == "multi":
            # Multi-path mode: taker-pays base, Fibonacci scaling, respect MAX_AMM_ITERS
            if hasattr(self.amm_context, "maxItersReached") and callable(getattr(self.amm_context, "maxItersReached")):
                if self.amm_context.maxItersReached():
                    return None
            used_iters = getattr(self.amm_context, "ammUsedIters", 0)
            if used_iters >= MAX_AMM_ITERS:
                return None
            # Pre-compute a combined OUT cap (min of out_amount and max_out_cap if both provided)
            combined_cap = None
            if out_amount is not None and (not out_amount.is_zero()):
                combined_cap = out_amount
            if max_out_cap is not None and (not max_out_cap.is_zero()):
                combined_cap = max_out_cap if combined_cap is None or max_out_cap < combined_cap else combined_cap
            # Base sizing on Decimal reserves (heuristic), then convert to STAmount
            base = self.amm.x / Decimal("40000")
            f = used_iters + 1
            if f > MAX_AMM_ITERS:
                return None
            fib = fibonacci(f)
            in_dec = base * fib
            if in_dec <= 0:
                return None
            in_st = STAmount.from_decimal(in_dec)
            if in_st.is_zero():
                # Minimal bump to one unit on X grid to avoid zero due to quantisation
                in_st = STAmount.from_components(1, -15, 1)
            out_st = self.amm.swap_out_given_in_st(in_st)
            if out_st.is_zero():
                return None
            # Apply combined OUT cap first to reduce rework
            if combined_cap is not None and out_st > combined_cap:
                in_st2 = self.amm.swap_in_given_out_st(combined_cap)
                if in_st2.is_zero():
                    return None
                out_st2 = self.amm.swap_out_given_in_st(in_st2)
                if out_st2.is_zero():
                    return None
                out_st = out_st2 if out_st2 <= combined_cap else combined_cap
                in_st = in_st2
            q = Quality.from_amounts(out_st, in_st)
            return Segment(
                src="AMM",
                quality=q,
                out_max=out_st,
                in_at_out_max=in_st,
                in_is_xrp=self.amm.x_is_xrp,
                out_is_xrp=self.amm.y_is_xrp,
                raw_quality=q,
            )
        else:
            raise ValueError(f"Unknown AMM liquidity mode: {mode}")