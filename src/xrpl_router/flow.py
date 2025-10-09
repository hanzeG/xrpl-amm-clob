from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Callable

from .amm_context import AMMContext
from .core import STAmount, Quality

# ----------------------------
# Integer-domain helpers
# ----------------------------

ZERO = STAmount.zero()


def _gt_zero(x: STAmount) -> bool:
    return (not x.is_zero()) and (x > ZERO)


def _compose_path_quality(step_quals: List[Quality]) -> Optional[Quality]:
    """Multiply per-step qualities to get a composed path quality.

    Quality is an STAmount-based rate. We multiply mantissas and add exponents
    in the integer domain, then normalise via STAmount.from_components.
    """
    if not step_quals:
        return None
    m = 1
    e = 0
    for q in step_quals:
        # If any step has zero/invalid rate, path quality is unusable
        if q.rate.is_zero():
            return None
        m *= q.rate.mantissa
        e += q.rate.exponent
    rate = STAmount.from_components(m, e, 1)
    if rate.is_zero():
        return None
    return Quality(rate)


# ----------------------------
# Payment sandbox
# ----------------------------

@dataclass
class PaymentSandbox:
    """Stage AMM/CLOB writebacks between reverse and forward passes.

    Values are tracked as STAmount pairs (dx, dy). The actual sink is injected
    by the caller (e.g., to mutate pools/books) and is only called on success.
    """
    staged: List[Tuple[STAmount, STAmount]]

    def __init__(self) -> None:
        self.staged = []

    def stage_after_iteration(self, dx: STAmount, dy: STAmount) -> None:
        self.staged.append((dx, dy))

    def stage_fee(self, dx: STAmount, dy: STAmount) -> None:
        self.staged.append((dx, dy))

    def apply(self, sink: Optional[Callable[[STAmount, STAmount], None]]) -> None:
        if sink is None:
            self.staged.clear()
            return
        for dx, dy in self.staged:
            sink(dx, dy)
        self.staged.clear()


# ----------------------------
# Main scheduler (reverse → forward)
# ----------------------------

def flow(
    payment_sandbox: PaymentSandbox,
    strands: Iterable[List["Step"]],  # runtime duck-typed; must expose quality_upper_bound/rev/fwd
    out_req: STAmount,
    *,
    send_max: Optional[STAmount] = None,
    limit_quality: Optional[Quality] = None,   # optional composed-path floor
    amm_context: Optional[AMMContext] = None,
    apply_sink: Optional[Callable[[STAmount, STAmount], None]] = None,
) -> Tuple[STAmount, STAmount]:
    """Execute strands per whitepaper (§1.3.2) in the integer domain.

    Returns a pair (total_in, total_out) as STAmount. The caller is responsible
    for any Decimal/display conversion at I/O boundaries.
    """
    if out_req.is_zero() or not _gt_zero(out_req):
        return (ZERO, ZERO)

    # Materialise strands for deterministic iteration across rounds
    strands = list(strands)

    remaining_out: STAmount = out_req
    remaining_in: Optional[STAmount] = send_max

    actual_in = ZERO
    actual_out = ZERO

    iters = 0

    while _gt_zero(remaining_out) and (remaining_in is None or remaining_in >= ZERO):
        # Collect active candidates with composed path quality
        active: List[Tuple[Quality, List["Step"]]] = []
        for strand in strands:
            try:
                step_qs = [s.quality_upper_bound() for s in strand]  # must return Quality
                q = _compose_path_quality(step_qs)
            except Exception:
                q = None
            if q is None:
                continue
            if limit_quality is not None and q.rate < limit_quality.rate:
                continue
            active.append((q, strand))

        if not active:
            break

        # Sort by numeric value of rate: mantissa * 10^exponent. Compare by (exponent, mantissa) desc.
        # Python's sort is stable, so equal keys preserve insertion order as a tie-break.
        active.sort(key=lambda t: (t[0].rate.exponent, t[0].rate.mantissa), reverse=True)

        # Set multipath flag (best-effort)
        if amm_context is not None:
            try:
                amm_context.setMultiPath(len(active) > 1)
            except Exception:
                pass

        # Try candidates sequentially until one succeeds
        attempt_succeeded = False
        for _, best in active:
            # Reverse pass: compute required IN and locate limiting step
            need = remaining_out
            sb = payment_sandbox
            limiting_idx: Optional[int] = None
            n_steps = len(best)

            for rev_pos, step in enumerate(reversed(best)):
                idx_forward = n_steps - 1 - rev_pos
                _in, _out = step.rev(sb, need)  # returns STAmount pairs
                if not _gt_zero(_out):
                    need = ZERO
                    limiting_idx = idx_forward
                    break
                if _out < need:
                    limiting_idx = idx_forward
                need = _in

            if not _gt_zero(need):
                payment_sandbox.apply(None)
                continue

            required_in = need

            # Determine start index for forward replay
            if remaining_in is not None and remaining_in < required_in:
                start_idx = 0
                in_cap0 = remaining_in  # budget is the cap
            else:
                start_idx = limiting_idx if limiting_idx is not None else 0
                in_cap0 = required_in

            # Forward replay from start_idx to end
            in_spent_add = ZERO
            out_propagate = ZERO
            ok = True
            for i in range(start_idx, len(best)):
                step_i = best[i]
                cap = in_cap0 if i == start_idx else out_propagate
                if not _gt_zero(cap):
                    ok = False
                    break
                _in2, _out2 = step_i.fwd(sb, cap)
                if not _gt_zero(_in2) or not _gt_zero(_out2):
                    ok = False
                    break
                if i == start_idx:
                    in_spent_add = _in2
                out_propagate = _out2

            if not ok:
                payment_sandbox.apply(None)
                continue

            # Commit accounting for the successful candidate
            actual_in = actual_in + in_spent_add
            actual_out = actual_out + out_propagate
            remaining_out = out_req - actual_out
            if remaining_in is not None:
                remaining_in = remaining_in - in_spent_add

            payment_sandbox.apply(apply_sink)
            attempt_succeeded = True
            break

        if not attempt_succeeded:
            break

        iters += 1
        if iters >= 128:  # MAX_FLOW_ITERS (kept local)
            break

    return actual_in, actual_out