from __future__ import annotations

from typing import Tuple, TYPE_CHECKING
from .core import STAmount, Quality

if TYPE_CHECKING:
    from .flow import PaymentSandbox

class Step:
    """Abstract payment step interface (integer domain, whitepaper §1.3.1).

    Reverse (rev): estimate required IN for a requested OUT (out_req).
    Forward (fwd): spend up to in_cap and produce OUT. Both may stage effects into PaymentSandbox.

    Each step supports reverse (rev) and forward (fwd) execution and can
    advertise a conservative quality upper bound for sorting (multi-path).
    """

    _cached_in: STAmount
    _cached_out: STAmount

    def rev(self, sandbox: "PaymentSandbox", out_req: STAmount) -> Tuple[STAmount, STAmount]:
        raise NotImplementedError

    def fwd(self, sandbox: "PaymentSandbox", in_cap: STAmount) -> Tuple[STAmount, STAmount]:
        raise NotImplementedError

    def quality_upper_bound(self) -> Quality:
        raise NotImplementedError

    def cached_in(self) -> STAmount:
        return getattr(self, "_cached_in", STAmount.zero())

    def cached_out(self) -> STAmount:
        return getattr(self, "_cached_out", STAmount.zero())
