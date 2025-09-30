from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, List, Optional, Tuple, Callable

from .amm_context import AMMContext
from .steps import Step

# --- Numerical guards for budget-constrained replay ---
from decimal import Decimal
BUDGET_EPS = Decimal("1e-9")        # absolute tolerance on budget (Decimal)
MAX_FLOW_ITERS = 128                # hard cap on reverse→forward replay rounds
# Local step-level epsilon (kept at 0 to preserve behaviour)
STEP_EPS = Decimal("0")

def _budget_eps(budget0: Optional[Decimal]) -> Decimal:
    """Compute a tolerant epsilon based on initial budget magnitude.

    We allow a tiny overshoot due to rounding/bucketing: eps = max(abs_epsilon, rel*|budget0|).
    """
    if budget0 is None:
        return BUDGET_EPS
    rel = (budget0.copy_abs() * BUDGET_EPS)
    return rel if rel > BUDGET_EPS else BUDGET_EPS


@dataclass
class PaymentSandbox:
    """Lightweight sandbox to stage AMM/CLOB writebacks between passes.

    Reverse pass computes needs, forward pass replays with limits;
    interim changes are staged and applied upon `apply()`. The actual
    pool/book mutation is delegated to `apply_sink`.
    """
    staged: List[Tuple[Decimal, Decimal]]

    def __init__(self) -> None:
        self.staged = []

    def stage_after_iteration(self, dx: Decimal, dy: Decimal) -> None:
        # Router calls this after each iteration; we record and defer actual pool updates
        self.staged.append((dx, dy))

    def apply(self, sink: Optional[Callable[[Decimal, Decimal], None]]) -> None:
        if sink is None:
            self.staged.clear()
            return
        for dx, dy in self.staged:
            sink(dx, dy)
        self.staged.clear()


def flow(payment_sandbox: PaymentSandbox,
         strands: Iterable[List[Step]],
         out_req: Decimal,
         *,
         send_max: Optional[Decimal] = None,
         limit_quality: Optional[Decimal] = None,   # reserved for per-strand use
         amm_context: Optional[AMMContext] = None,  # shared context if needed
         apply_sink: Optional[Callable[[Decimal, Decimal], None]] = None
         ) -> Tuple[Decimal, Decimal]:
    """Execute strands per whitepaper (§1.3.2) — minimal skeleton.

    Conservative single-strand adapter:
      * Sort strands by quality upper bound and execute the best first.
      * Reverse pass then forward replay (using adapter that collapses book exec).
      * Apply staged writebacks each iteration.
    """
    # Materialise strands to allow repeatable iteration across rounds
    strands = list(strands)

    remaining_out = out_req
    remaining_in = send_max

    budget0 = send_max  # remember initial budget to derive a stable epsilon
    eps = _budget_eps(budget0)
    iters = 0

    actual_in = Decimal(0)
    actual_out = Decimal(0)

    # Sort by quality bound (desc) each iteration
    while remaining_out > 0 and (remaining_in is None or remaining_in > -eps):
        active: List[Tuple[Decimal, List[Step]]] = []
        for strand in strands:
            try:
                # Strand bound = min step bounds (conservative)
                q = min((s.quality_upper_bound() for s in strand), default=Decimal(0))
            except Exception:
                q = Decimal(0)
            if q > 0:
                active.append((q, strand))
        if not active:
            break
        active.sort(key=lambda t: t[0], reverse=True)
        _, best = active[0]

        # Reverse pass (no writebacks during rev): record step requirements and limiting step
        need = remaining_out
        sb = payment_sandbox
        rev_records: List[Tuple[int, Step, Decimal, Decimal]] = []  # (idx_forward, step, out_req, out_got)
        limiting_idx: Optional[int] = None
        n_steps = len(best)
        for rev_pos, step in enumerate(reversed(best)):
            idx_forward = n_steps - 1 - rev_pos
            _in, _out = step.rev(sb, need)
            rev_records.append((idx_forward, step, need, _out))
            if _out <= STEP_EPS:
                need = Decimal(0)
                limiting_idx = idx_forward
                break
            if _out < need:
                # Step limited our requested OUT; mark this step as limiting
                limiting_idx = idx_forward
            need = _in
        if need <= 0:
            break
        required_in = need

        # Determine start index for forward replay:
        # - If budget-limited (remaining_in < required_in), start at the first step (0) using the budget as cap.
        # - Else if a particular step was limiting in reverse, start from that step.
        # - Else start from 0.
        budget_limited = (remaining_in is not None and (remaining_in + eps) < required_in)
        if budget_limited:
            start_idx = 0
            in_cap0 = remaining_in  # budget is the cap
        else:
            start_idx = limiting_idx if limiting_idx is not None else 0
            in_cap0 = required_in

        # Forward replay from start_idx to end
        in_spent_add = Decimal(0)
        out_propagate = Decimal(0)
        ok = True
        for i in range(start_idx, len(best)):
            step_i = best[i]
            cap = in_cap0 if i == start_idx else out_propagate
            if cap is None or cap <= STEP_EPS:
                ok = False
                break
            _in2, _out2 = step_i.fwd(sb, cap)
            if _in2 <= STEP_EPS or _out2 <= STEP_EPS:
                ok = False
                break
            # Only the first executed step consumes *user* IN; subsequent steps transform amounts internally
            if i == start_idx:
                in_spent_add = _in2
            out_propagate = _out2

        if not ok:
            break

        # Commit accounting
        actual_in += in_spent_add
        actual_out += out_propagate
        remaining_out = out_req - actual_out
        if remaining_in is not None:
            remaining_in -= in_spent_add
        if remaining_in is not None and remaining_in < 0 and (-remaining_in) <= eps:
            remaining_in = Decimal(0)

        # Apply staged AMM/CLOB updates for this iteration
        # Writebacks are applied after each successful iteration, not during reverse/forward passes
        payment_sandbox.apply(apply_sink)

        iters += 1
        if iters >= MAX_FLOW_ITERS:
            break

    return actual_in, actual_out