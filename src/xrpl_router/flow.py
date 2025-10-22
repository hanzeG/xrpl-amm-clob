from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Callable

from .core import Amount, Quality, XRPAmount, IOUAmount

# ----------------------------
# Integer-domain helpers
# ----------------------------

def _zero_like(x: Amount) -> Amount:
    """Return a zero Amount of the same concrete type as x."""
    return XRPAmount(0) if isinstance(x, XRPAmount) else IOUAmount.zero()

def _gt_zero(x: Amount) -> bool:
    z = _zero_like(x)
    return (not x.is_zero()) and (x > z)


def _compose_path_quality(step_quals: List[Quality]) -> Optional[Quality]:
    """Multiply per-step qualities to get a composed path quality.

    Quality is an Amount-based rate. We multiply mantissas and add exponents
    in the integer domain, then normalise via IOUAmount.from_components.
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
    rate = IOUAmount.from_components(m, e)
    if rate.is_zero():
        return None
    return Quality(rate)


# ----------------------------
# Payment sandbox
# ----------------------------

@dataclass
class PaymentSandbox:
    """Stage AMM/CLOB writebacks between reverse and forward passes.

    Values are tracked as Amount pairs (dx, dy). The actual sink is injected
    by the caller (e.g., to mutate pools/books) and is only called on success.
    """
    staged: List[Tuple[Amount, Amount]]

    def __init__(self) -> None:
        self.staged = []

    def stage_after_iteration(self, dx: Amount, dy: Amount) -> None:
        self.staged.append((dx, dy))


    def apply(self, sink: Optional[Callable[[Amount, Amount], None]]) -> None:
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
    out_req: Amount,
    *,
    send_max: Optional[Amount] = None,
    limit_quality: Optional[Quality] = None,   # optional composed-path floor
    apply_sink: Optional[Callable[[Amount, Amount], None]] = None,
) -> Tuple[Amount, Amount]:
    """Execute strands per whitepaper (§1.3.2) in the integer domain.

    Returns a pair (total_in, total_out) as Amount. The caller is responsible
    for any Decimal/display conversion at I/O boundaries.
    """
    if out_req.is_zero() or not _gt_zero(out_req):
        z = _zero_like(out_req)
        return (z, z)

    # Materialise strands for deterministic iteration across rounds
    strands = list(strands)

    remaining_out: Amount = out_req
    remaining_in: Optional[Amount] = send_max
    actual_in: Optional[Amount] = None
    actual_out: Amount = _zero_like(out_req)

    iters = 0

    while _gt_zero(remaining_out) and (remaining_in is None or remaining_in >= _zero_like(remaining_in)):
        # The above allows remaining_in to be zero or positive; negative breaks.
        if remaining_in is not None and remaining_in < _zero_like(remaining_in):
            break

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
                _in, _out = step.rev(sb, need)  # returns Amount pairs
                if not _gt_zero(_out):
                    need = _zero_like(_out)
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
            in_spent_add = _zero_like(out_req)
            out_propagate = _zero_like(out_req)
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
            actual_out = actual_out + out_propagate
            remaining_out = out_req - actual_out
            if actual_in is None:
                actual_in = in_spent_add
            else:
                actual_in = actual_in + in_spent_add
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

    if actual_in is None:
        actual_in = _zero_like(out_req)
    return actual_in, actual_out