

"""AMM liquidity routing helpers matching the whitepaper (1.2.7).

Provides:
 - AMMOffer: lightweight AMM offer descriptor (for router diagnostics).
 - AMMLiquidity: helper with get_offer supporting three modes:
    * "single" (1.2.7.1): exact swap for requested in/out.
    * "anchored" (1.2.7.2): anchored to CLOB quality, only if SPQ > lob_quality.
    * "multi" (1.2.7.3): multi-path Fibonacci slicing, respecting MAX_AMM_ITERS.

Note: The returned object is a Segment (to minimize integration changes for now).
Both `quality` (bucketed) and `raw_quality` are filled.

IMPORTANT: The caller must call `AMMContext.setAMMUsed()` *only when* the returned segment is actually consumed.
"""
from __future__ import annotations
from decimal import Decimal
from typing import Optional, List, Literal

from .amm import AMM
from .core import (
    Segment,
    quantize_quality,
    quality_bucket,
    calc_quality,
    clamp_nonneg,
)

# Maximum allowed AMM iterations for multi-path mode
MAX_AMM_ITERS = 30

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
        amm_context: context with .ammUsedIters (int) and .setAMMUsed() method.
        """
        self.amm = amm
        self.amm_context = amm_context

    def get_offer(
        self,
        mode: Literal["single", "anchored", "multi"],
        *,
        in_amount: Optional[Decimal] = None,
        out_amount: Optional[Decimal] = None,
        lob_quality: Optional[Decimal] = None,
        max_out_cap: Optional[Decimal] = None,
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
            # Single mode: exact swap, using in_amount or out_amount
            if in_amount is not None:
                in_amt = clamp_nonneg(in_amount)
                out_amt = self.amm.swap_out_given_in(in_amt)
                if out_amt <= 0:
                    return None
                quality = calc_quality(out_amt, in_amt)
                q_bucket = quality_bucket(quality)
                return Segment(
                    src="AMM",
                    quality=q_bucket,
                    out_max=out_amt,
                    in_at_out_max=in_amt,
                    in_is_xrp=self.amm.x_is_xrp,
                    out_is_xrp=self.amm.y_is_xrp,
                    raw_quality=quantize_quality(quality),
                )
            elif out_amount is not None:
                out_amt = clamp_nonneg(out_amount)
                try:
                    in_amt = self.amm.swap_in_given_out(out_amt)
                except Exception:
                    return None
                if in_amt <= 0:
                    return None
                # Recompute actual out (fee/grid effects)
                out_got = self.amm.swap_out_given_in(in_amt)
                if out_got <= 0:
                    return None
                # Cap to requested out
                if out_got > out_amt:
                    out_got = out_amt
                quality = calc_quality(out_got, in_amt)
                q_bucket = quality_bucket(quality)
                return Segment(
                    src="AMM",
                    quality=q_bucket,
                    out_max=out_got,
                    in_at_out_max=in_amt,
                    in_is_xrp=self.amm.x_is_xrp,
                    out_is_xrp=self.amm.y_is_xrp,
                    raw_quality=quantize_quality(quality),
                )
            else:
                return None

        elif mode == "anchored":
            # Anchored mode: anchored to lob_quality, only if SPQ > lob_quality
            if lob_quality is None or lob_quality <= 0:
                return None
            spq = self.amm.spq()
            if spq <= lob_quality:
                # At or below CLOB quality, do not return AMM slice (CLOB preferred)
                return None
            # Use synthetic_segment_for_quality (1.2.7.2)
            seg = self.amm.synthetic_segment_for_quality(
                lob_quality,
                max_out_cap=max_out_cap if max_out_cap is not None else out_amount,
            )
            if seg is None:
                return None
            # Fill raw_quality
            seg.raw_quality = quantize_quality(calc_quality(seg.out_max, seg.in_at_out_max))
            return seg

        elif mode == "multi":
            # Multi-path mode: use taker-pays base, Fibonacci scaling, respect MAX_AMM_ITERS
            # Only proceed if AMM has not reached max iters
            used_iters = getattr(self.amm_context, "ammUsedIters", 0)
            cap_iters = getattr(self.amm_context, "maxIters", MAX_AMM_ITERS)
            if used_iters >= cap_iters or used_iters >= MAX_AMM_ITERS:
                return None
            # Taker-pays base = x_reserve / 40000 (whitepaper), scale by Fibonacci(f)
            base = self.amm.x / Decimal("40000")
            # f = used_iters + 1 (always ≥1)
            f = used_iters + 1
            if f > MAX_AMM_ITERS:
                return None
            fib = fibonacci(f)
            in_amt = base * fib
            in_amt = clamp_nonneg(in_amt)
            if in_amt <= 0:
                return None
            out_amt = self.amm.swap_out_given_in(in_amt)
            if out_amt <= 0:
                return None
            # If target_out is provided and smaller, cap by recomputing in via swap_in_given_out
            if out_amount is not None and out_amount > 0 and out_amount < out_amt:
                # Recompute in_amt for capped out
                try:
                    in_amt_new = self.amm.swap_in_given_out(out_amount)
                except Exception:
                    return None
                if in_amt_new <= 0:
                    return None
                out_amt_new = self.amm.swap_out_given_in(in_amt_new)
                if out_amt_new <= 0:
                    return None
                if out_amt_new > out_amount:
                    out_amt_new = out_amount
                in_amt = in_amt_new
                out_amt = out_amt_new
            # If a max_out_cap is provided (e.g., anchored coexistence), cap the slice
            if max_out_cap is not None and max_out_cap > 0 and out_amt > max_out_cap:
                try:
                    in_amt_new = self.amm.swap_in_given_out(max_out_cap)
                except Exception:
                    return None
                if in_amt_new <= 0:
                    return None
                out_amt_new = self.amm.swap_out_given_in(in_amt_new)
                if out_amt_new <= 0:
                    return None
                if out_amt_new > max_out_cap:
                    out_amt_new = max_out_cap
                in_amt = in_amt_new
                out_amt = out_amt_new
            quality = calc_quality(out_amt, in_amt)
            q_bucket = quality_bucket(quality)
            return Segment(
                src="AMM",
                quality=q_bucket,
                out_max=out_amt,
                in_at_out_max=in_amt,
                in_is_xrp=self.amm.x_is_xrp,
                out_is_xrp=self.amm.y_is_xrp,
                raw_quality=quantize_quality(quality),
            )
        else:
            raise ValueError(f"Unknown AMM liquidity mode: {mode}")