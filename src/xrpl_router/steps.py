from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Iterable, Optional, Tuple, List

from .core import Segment

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
class DirectStepI(Step):
    """IOU→IOU direct transfer step (whitepaper §1.3.1).

    NOTE: Skeleton placeholder for Batch 2. The actual fee and issuer‑transfer
    accounting (ownerGives/stpAmt) will be implemented when BookStep is
    introduced. Do not instantiate in production yet.
    """
    in_is_xrp: bool = False
    out_is_xrp: bool = False

    def rev(self, sandbox: "PaymentSandbox", out_req: Decimal) -> Tuple[Decimal, Decimal]:
        # Placeholder: identity mapping (no fees). To be replaced.
        self._cached_in = out_req
        self._cached_out = out_req
        return self._cached_in, self._cached_out

    def fwd(self, sandbox: "PaymentSandbox", in_cap: Decimal) -> Tuple[Decimal, Decimal]:
        # Placeholder: pass‑through, no writebacks here (Direct transfer writebacks
        # will be staged by sandbox in Batch 2).
        self._cached_in = in_cap
        self._cached_out = in_cap
        return self._cached_in, self._cached_out

    def quality_upper_bound(self) -> Decimal:
        # Neutral bound (1.0) as placeholder.
        return Decimal(1)


@dataclass
class XRPEndpointStep(Step):
    """XRP endpoint step (source/sink), whitepaper §1.3.1.

    NOTE: Skeleton placeholder for Batch 2. Wallet balance checks and staging
    to sandbox will be added with proper ledger plumbing.
    """
    is_source: bool = True

    def rev(self, sandbox: "PaymentSandbox", out_req: Decimal) -> Tuple[Decimal, Decimal]:
        # Placeholder: identity mapping.
        self._cached_in = out_req
        self._cached_out = out_req
        return self._cached_in, self._cached_out

    def fwd(self, sandbox: "PaymentSandbox", in_cap: Decimal) -> Tuple[Decimal, Decimal]:
        # Placeholder pass‑through.
        self._cached_in = in_cap
        self._cached_out = in_cap
        return self._cached_in, self._cached_out

    def quality_upper_bound(self) -> Decimal:
        return Decimal(1)

