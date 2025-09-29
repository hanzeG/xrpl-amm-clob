from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Iterable, Optional, Tuple

from .core import Segment, quality_bucket
from .router import route, RouteConfig
from .amm_context import AMMContext


class Step:
    """Abstract payment step interface (whitepaper §1.3.1).

    Each step supports reverse (rev) and forward (fwd) execution and can
    advertise a conservative quality upper bound for sorting (multi-path).
    """

    _cached_in: Decimal
    _cached_out: Decimal

    def rev(self, sandbox: "PaymentSandbox", out_req: Decimal) -> Tuple[Decimal, Decimal]:
        raise NotImplementedError

    def fwd(self, sandbox: "PaymentSandbox", in_cap: Decimal) -> Tuple[Decimal, Decimal]:
        raise NotImplementedError

    def quality_upper_bound(self) -> Decimal:
        raise NotImplementedError

    def cached_in(self) -> Decimal:
        return getattr(self, "_cached_in", Decimal(0))

    def cached_out(self) -> Decimal:
        return getattr(self, "_cached_out", Decimal(0))


@dataclass
class BookStepAdapter(Step):
    """Adapter that treats the existing router as a single Book/Direct step.

    This bridges the whitepaper flow without changing the current router.
    """

    segments_provider: Callable[[], Iterable[Segment]]
    amm_anchor: Optional[Callable[[Decimal, Decimal], Optional[Segment]]] = None
    amm_curve: Optional[Callable[[Decimal], Iterable[Segment]]] = None
    amm_context: Optional[AMMContext] = None
    limit_quality: Optional[Decimal] = None

    def _exec(self,
              sandbox: "PaymentSandbox",
              after_iteration: Optional[Callable[[Decimal, Decimal], None]],
              *,
              target_out: Optional[Decimal] = None,
              send_max: Optional[Decimal] = None) -> Tuple[Decimal, Decimal]:
        assert (target_out is None) ^ (send_max is None), "use either target_out or send_max"
        segs = list(self.segments_provider())
        if not segs:
            self._cached_in = Decimal(0)
            self._cached_out = Decimal(0)
            return self._cached_in, self._cached_out

        res = route(
            target_out=target_out if target_out is not None else Decimal("1e40"),
            segments=segs,
            config=RouteConfig(preserve_quality_on_limit=True),
            amm_anchor=self.amm_anchor,
            amm_curve=self.amm_curve,
            amm_context=self.amm_context,
            send_max=send_max,
            limit_quality=self.limit_quality,
            after_iteration=after_iteration,
        )
        self._cached_in = res.spent_in
        self._cached_out = res.filled_out
        return self._cached_in, self._cached_out

    def rev(self, sandbox: "PaymentSandbox", out_req: Decimal) -> Tuple[Decimal, Decimal]:
        # Reverse pass: do not stage writebacks.
        return self._exec(sandbox, None, target_out=out_req, send_max=None)

    def fwd(self, sandbox: "PaymentSandbox", in_cap: Decimal) -> Tuple[Decimal, Decimal]:
        # Forward pass: stage writebacks using sandbox callback.
        return self._exec(sandbox, sandbox.stage_after_iteration, target_out=None, send_max=in_cap)

    def quality_upper_bound(self) -> Decimal:
        try:
            segs = list(self.segments_provider())
        except Exception:
            return Decimal(0)
        if not segs:
            return Decimal(0)
        return max(quality_bucket(s.quality) for s in segs)