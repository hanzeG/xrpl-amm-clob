from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, List, Optional, Tuple, Callable

from .amm_context import AMMContext
from .steps import Step
from .core import compose_path_quality

# --- Numerical guards for budget-constrained replay ---
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

    def stage_fee(self, dx: Decimal, dy: Decimal) -> None:
        """Stage a pure fee adjustment (e.g. issuer fee). Same format as stage_after_iteration."""
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

    Conservative multi-path scheduler:
      * Compose path quality as the product of step bounds; sort candidates by quality (desc).
      * Each round, set AMM multi-path flag based on active candidate count.
      * For each round, try candidates in order; reverse pass then forward replay.
      * On failure, clear staged writes and try the next candidate; on success, apply staged writes.
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
                step_qs = [s.quality_upper_bound() for s in strand]
                q = compose_path_quality(step_qs)
            except Exception:
                q = Decimal(0)
            if q <= 0:
                continue
            if limit_quality is not None and q < limit_quality:
                continue
            active.append((q, strand))
        if not active:
            break
        # Sort paths by composed quality (desc) and set multi-path flag for this round
        active.sort(key=lambda t: t[0], reverse=True)
        if amm_context is not None:
            amm_context.setMultiPath(len(active) > 1)

        # Try candidates in order until one succeeds; if none succeed, terminate
        attempt_succeeded = False
        for _, best in active:
            # Reverse pass (no writebacks during rev): record step requirements and limiting step
            need = remaining_out
            sb = payment_sandbox
            rev_records: List[Tuple[int, Step, Decimal, Decimal]] = []
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
                    limiting_idx = idx_forward
                need = _in
            if need <= 0:
                # This strand cannot progress; clear any staged writes defensively and try next
                payment_sandbox.apply(None)
                continue
            required_in = need

            # Determine start index for forward replay:
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
                if i == start_idx:
                    in_spent_add = _in2
                out_propagate = _out2

            if not ok:
                # Clear any staged writes for this failed attempt and try next candidate
                payment_sandbox.apply(None)
                continue

            # Commit accounting for the successful candidate
            actual_in += in_spent_add
            actual_out += out_propagate
            remaining_out = out_req - actual_out
            if remaining_in is not None:
                remaining_in -= in_spent_add
            if remaining_in is not None and remaining_in < 0 and (-remaining_in) <= eps:
                remaining_in = Decimal(0)

            # Apply staged AMM/CLOB updates for this iteration
            payment_sandbox.apply(apply_sink)
            attempt_succeeded = True
            break  # move to next outer iteration

        if not attempt_succeeded:
            break

        iters += 1
        if iters >= MAX_FLOW_ITERS:
            break

    return actual_in, actual_out