from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, List, Optional, Callable, Sequence, Tuple, Dict

from .core import Segment, RouteResult, ExecutionReport, ROUTING_CFG
from .exec_modes import run_trade_mode, ExecutionMode
# Hybrid flow/step imports
from .amm_context import AMMContext
from .amm import AMM
# Hybrid flow/step imports
from .flow import PaymentSandbox, flow
from .steps import Step, BookStepAdapter


@dataclass(frozen=True)
class AlphaPoint:
    """One point on the alpha curve: split q into AMM-only (alpha*q) and CLOB-only ((1-alpha)*q).

    Metrics are aggregated across the two legs; feasibility reflects whether both legs
    achieved their respective target outs (i.e., filled_ratio ≈ 1 on each leg when deliver_min is None).
    """
    alpha: Decimal
    out_total: Decimal
    in_total: Decimal
    avg_price: Decimal  # in_total / out_total when out_total>0 else 0
    feasible: bool
    res_amm: RouteResult
    res_clob: RouteResult


@dataclass(frozen=True)
class AlphaScanResult:
    """Alpha scan across [0,1] with fixed step or user-supplied grid."""
    points: List[AlphaPoint]
    alpha_star: Decimal
    best_price: Decimal


# --- Crossover analysis ---

@dataclass(frozen=True)
class CrossoverSample:
    q: Decimal
    price_amm: Decimal
    price_clob: Decimal
    feasible_amm: bool
    feasible_clob: bool
    res_amm: RouteResult
    res_clob: RouteResult


@dataclass(frozen=True)
class CrossoverResult:
    samples: List[CrossoverSample]
    q_star: Optional[Decimal]          # None if no bracket found; see best_abs
    price_at_q_star: Optional[Decimal]
    bracket: Optional[Tuple[Decimal, Decimal]]  # (q_lo, q_hi) with sign change if found
    best_abs: CrossoverSample          # sample with smallest |price_amm - price_clob|



def _split_budget(value: Optional[Decimal], alpha: Decimal) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    if value is None:
        return None, None
    # Proportional split to keep per-leg budget consistent with overall send_max/deliver_min
    a = (value * alpha)
    b = (value - a)
    return a, b

# --- Report unification helper ---

def _with_fee_totals(rr: RouteResult,
                     fee_pool_total: Decimal,
                     fee_tr_in_total: Decimal,
                     fee_tr_out_total: Decimal) -> RouteResult:
    """Return a new RouteResult whose report aggregates include the provided fee totals.

    If rr.report is None, the input is returned unchanged.
    """
    er = rr.report
    if er is None:
        return rr
    new_er = ExecutionReport(
        iterations=er.iterations,
        total_out=er.total_out,
        total_in=er.total_in,
        avg_price=er.avg_price,
        avg_quality=er.avg_quality,
        filled_ratio=er.filled_ratio,
        in_budget_ratio=er.in_budget_ratio,
        fee_pool_total=fee_pool_total,
        fee_tr_in_total=fee_tr_in_total,
        fee_tr_out_total=fee_tr_out_total,
        slippage_price_avg=er.slippage_price_avg,
    )
    return RouteResult(
        filled_out=rr.filled_out,
        spent_in=rr.spent_in,
        avg_quality=rr.avg_quality,
        usage=rr.usage,
        trace=rr.trace,
        report=new_er,
    )


# --- Crossover helper ---
def _eval_cost_pair(
    q: Decimal,
    *,
    segments: List[Segment],
    amm_anchor,
    amm_curve,
    amm_context,
    send_max: Optional[Decimal],
    deliver_min: Optional[Decimal],
    amm_for_fees: Optional[AMM] = None,
) -> CrossoverSample:
    # AMM-only cost
    fee_meter: Dict[str, Decimal] = {}
    res_amm = run_trade_mode(
        ExecutionMode.AMM_ONLY,
        target_out=q,
        segments=segments,
        send_max=send_max,
        deliver_min=deliver_min,
        amm_anchor=amm_anchor,
        amm_curve=amm_curve,
        amm_context=amm_context,
        apply_sink=_wrap_apply_sink_with_fee_meter(amm_for_fees, None, fee_meter),
    )
    # Attach fee totals into AMM leg's report if available
    res_amm = _with_fee_totals(
        res_amm,
        fee_pool_total=fee_meter.get('fee_pool', Decimal(0)),
        fee_tr_in_total=fee_meter.get('fee_tr_in', Decimal(0)),
        fee_tr_out_total=fee_meter.get('fee_tr_out', Decimal(0)),
    )
    price_amm = (res_amm.spent_in / res_amm.filled_out) if res_amm.filled_out > 0 else Decimal(0)
    feas_amm = (q == 0) or (res_amm.filled_out >= q)

    # CLOB-only cost
    res_clob = run_trade_mode(
        ExecutionMode.CLOB_ONLY,
        target_out=q,
        segments=segments,
        send_max=send_max,
        deliver_min=deliver_min,
        amm_anchor=amm_anchor,
        amm_curve=amm_curve,
        amm_context=amm_context,
    )
    price_clob = (res_clob.spent_in / res_clob.filled_out) if res_clob.filled_out > 0 else Decimal(0)
    feas_clob = (q == 0) or (res_clob.filled_out >= q)

    return CrossoverSample(
        q=q,
        price_amm=price_amm,
        price_clob=price_clob,
        feasible_amm=feas_amm,
        feasible_clob=feas_clob,
        res_amm=res_amm,
        res_clob=res_clob,
    )


def analyze_alpha_scan(
    *,
    target_out: Decimal,
    segments: Iterable[Segment] | List[Segment],
    amm_anchor: Optional[Callable[[Decimal, Decimal], Optional[Segment]]] = None,
    amm_curve: Optional[Callable[[Decimal], Iterable[Segment]]] = None,
    amm_context: Optional[AMMContext] = None,
    send_max: Optional[Decimal] = None,
    deliver_min: Optional[Decimal] = None,
    grid: Optional[Sequence[Decimal]] = None,
    step: Decimal = ROUTING_CFG.alpha_step_default,
    amm_for_fees: Optional[AMM] = None,
) -> AlphaScanResult:
    """Scan alpha ∈ [0,1] and find the split that minimises average price (in/out).

    Two-leg construction (external, apples-to-apples):
      - AMM leg: run AMM_ONLY with target_out = alpha * q.
      - CLOB leg: run CLOB_ONLY with target_out = (1-alpha) * q.
    Budgets/limits are split proportionally between the two legs when provided.

    Returns the full curve and the argmin alpha_star.
    """
    segs_list = list(segments)

    if grid is None:
        # Build a numeric grid [0,1] inclusive using `step`.
        alphas: List[Decimal] = []
        a = Decimal("0")
        one = Decimal("1")
        while a < one:
            alphas.append(a)
            a = (a + step) if step > 0 else one
        if alphas[-1] != one:
            alphas.append(one)
    else:
        alphas = [Decimal(str(x)) for x in grid]
        # Ensure 0 and 1 are included (deduplicate then sort)
        s = set(alphas)
        s.add(Decimal("0"))
        s.add(Decimal("1"))
        alphas = sorted(s)

    points: List[AlphaPoint] = []

    for a in alphas:
        q_amm = (target_out * a)
        q_clob = (target_out - q_amm)

        send_max_amm, send_max_clob = _split_budget(send_max, a)
        deliver_min_amm, deliver_min_clob = _split_budget(deliver_min, a)

        # AMM-only leg
        fee_meter: Dict[str, Decimal] = {}
        res_amm = run_trade_mode(
            ExecutionMode.AMM_ONLY,
            target_out=q_amm,
            segments=segs_list,  # ignored by AMM_ONLY
            send_max=send_max_amm,
            deliver_min=deliver_min_amm,
            limit_quality=None,
            amm_anchor=amm_anchor,
            amm_curve=amm_curve,
            amm_context=amm_context,
            apply_sink=_wrap_apply_sink_with_fee_meter(amm_for_fees, None, fee_meter),
        )
        # Attach fee totals into AMM leg's report if available
        res_amm = _with_fee_totals(
            res_amm,
            fee_pool_total=fee_meter.get('fee_pool', Decimal(0)),
            fee_tr_in_total=fee_meter.get('fee_tr_in', Decimal(0)),
            fee_tr_out_total=fee_meter.get('fee_tr_out', Decimal(0)),
        )
        # CLOB-only leg
        res_clob = run_trade_mode(
            ExecutionMode.CLOB_ONLY,
            target_out=q_clob,
            segments=segs_list,
            send_max=send_max_clob,
            deliver_min=deliver_min_clob,
            limit_quality=None,
            amm_anchor=amm_anchor,  # ignored by CLOB_ONLY
            amm_curve=amm_curve,    # ignored by CLOB_ONLY
            amm_context=amm_context,
        )

        out_total = res_amm.filled_out + res_clob.filled_out
        in_total = res_amm.spent_in + res_clob.spent_in
        avg_price = (in_total / out_total) if out_total > 0 else Decimal(0)

        # Feasible iff each leg hit its own target (within rounding)
        feas_amm = (q_amm == 0) or (res_amm.filled_out >= q_amm)
        feas_clob = (q_clob == 0) or (res_clob.filled_out >= q_clob)
        feasible = bool(feas_amm and feas_clob)

        points.append(AlphaPoint(
            alpha=a,
            out_total=out_total,
            in_total=in_total,
            avg_price=avg_price,
            feasible=feasible,
            res_amm=res_amm,
            res_clob=res_clob,
        ))

    # Choose alpha* among feasible points; if none feasible, fall back to best price overall
    feasible_pts = [p for p in points if p.feasible]
    if feasible_pts:
        best = min(feasible_pts, key=lambda p: p.avg_price if p.out_total > 0 else Decimal("Inf"))
    else:
        best = min(points, key=lambda p: p.avg_price if p.out_total > 0 else Decimal("Inf"))

    return AlphaScanResult(points=points, alpha_star=best.alpha, best_price=best.avg_price)


# --- Crossover search ---
def find_crossover_q(
    *,
    segments: Iterable[Segment] | List[Segment],
    amm_anchor: Optional[Callable[[Decimal, Decimal], Optional[Segment]]] = None,
    amm_curve: Optional[Callable[[Decimal], Iterable[Segment]]] = None,
    amm_context: Optional[AMMContext] = None,
    send_max: Optional[Decimal] = None,
    deliver_min: Optional[Decimal] = None,
    q_grid: Optional[Sequence[Decimal]] = None,
    q_min: Optional[Decimal] = None,
    q_max: Optional[Decimal] = None,
    coarse_steps: int = ROUTING_CFG.qscan_coarse_steps,
    refine_iters: int = ROUTING_CFG.qscan_refine_iters,
    amm_for_fees: Optional[AMM] = None,
) -> CrossoverResult:
    """Search for q* where AMM and CLOB average prices are equal (within tolerance).

    Strategy:
      1) Build a coarse grid on [q_min, q_max] (or from provided q_grid). If neither is provided,
         derive q_max from available CLOB segments' total out_max; set q_min to a small positive quantum.
      2) Evaluate f(q) = price_amm(q) - price_clob(q) across the grid to find a sign change bracket.
      3) If a bracket exists, refine by bisection on that interval (refine_iters), returning q* and price.
      4) If no bracket exists, return the sample with the smallest |f(q)| as `best_abs`.
    """
    segs_list = list(segments)

    # Heuristic defaults if bounds not provided
    if q_grid is None:
        if q_max is None:
            # Use 80% of observable CLOB liquidity as a conservative ceiling if available
            clob_total = sum((s.out_max for s in segs_list if s.src == "CLOB"), Decimal(0))
            q_max = clob_total * Decimal("0.8") if clob_total > 0 else Decimal("1")
        if q_min is None:
            q_min = q_max / Decimal("1000")  # small but non-zero
        # Linear coarse grid
        if coarse_steps < 2:
            coarse_steps = 2
        step = (q_max - q_min) / Decimal(coarse_steps)
        grid = [q_min + step * Decimal(i) for i in range(coarse_steps + 1)]
    else:
        grid = [Decimal(str(x)) for x in q_grid]
        grid = sorted({x for x in grid if x >= 0})
        if not grid:
            raise ValueError("q_grid is empty")

    samples: List[CrossoverSample] = []
    for q in grid:
        samples.append(_eval_cost_pair(
            q,
            segments=segs_list,
            amm_anchor=amm_anchor,
            amm_curve=amm_curve,
            amm_context=amm_context,
            send_max=send_max,
            deliver_min=deliver_min,
            amm_for_fees=amm_for_fees,
        ))

    # Find sign change in f(q) = price_amm - price_clob among feasible samples
    def fval(s: CrossoverSample) -> Decimal:
        return (s.price_amm - s.price_clob)

    feasible = [s for s in samples if s.feasible_amm and s.feasible_clob and s.q > 0]

    bracket: Optional[Tuple[Decimal, Decimal]] = None
    best_abs = min(samples, key=lambda s: abs(fval(s))) if samples else None

    for i in range(1, len(feasible)):
        s0, s1 = feasible[i-1], feasible[i]
        f0, f1 = fval(s0), fval(s1)
        if (f0 == 0) or (f1 == 0) or (f0 < 0 and f1 > 0) or (f0 > 0 and f1 < 0):
            bracket = (s0.q, s1.q)
            break

    q_star: Optional[Decimal] = None
    price_at_q_star: Optional[Decimal] = None

    if bracket is not None:
        lo, hi = bracket
        s_lo = next((s for s in samples if s.q == lo), None)
        s_hi = next((s for s in samples if s.q == hi), None)
        if s_lo is None or s_hi is None:
            s_lo = _eval_cost_pair(lo, segments=segs_list, amm_anchor=amm_anchor, amm_curve=amm_curve, amm_context=amm_context, send_max=send_max, deliver_min=deliver_min, amm_for_fees=amm_for_fees)
            s_hi = _eval_cost_pair(hi, segments=segs_list, amm_anchor=amm_anchor, amm_curve=amm_curve, amm_context=amm_context, send_max=send_max, deliver_min=deliver_min, amm_for_fees=amm_for_fees)

        # Bisection refinement on feasible region
        for _ in range(refine_iters):
            mid = (lo + hi) / 2
            s_mid = _eval_cost_pair(mid, segments=segs_list, amm_anchor=amm_anchor, amm_curve=amm_curve, amm_context=amm_context, send_max=send_max, deliver_min=deliver_min, amm_for_fees=amm_for_fees)
            if not (s_mid.feasible_amm and s_mid.feasible_clob) or s_mid.q <= 0:
                # If mid infeasible, shrink towards feasible side: pick the side with smaller price gap
                if abs(fval(s_lo)) <= abs(fval(s_hi)):
                    hi = mid
                else:
                    lo = mid
                continue
            f_lo = fval(s_lo)
            f_mid = fval(s_mid)
            # Update bracket
            if (f_lo == 0) or (f_mid == 0) or (f_lo < 0 and f_mid > 0) or (f_lo > 0 and f_mid < 0):
                hi = mid
                s_hi = s_mid
            else:
                lo = mid
                s_lo = s_mid
        # Final estimate
        q_star = (lo + hi) / 2
        price_at_q_star = (s_lo.price_amm + s_hi.price_amm) / 2  # approx; either side similar near root

        samples.extend([s_lo, s_hi])  # ensure endpoints are included

    # Choose best_abs if we didn't find a bracket
    if best_abs is None:
        raise RuntimeError("no samples collected; invalid inputs?")

    return CrossoverResult(
        samples=samples,
        q_star=q_star,
        price_at_q_star=price_at_q_star,
        bracket=bracket,
        best_abs=best_abs,
    )


__all__ = [
    "AlphaPoint",
    "AlphaScanResult",
    "analyze_alpha_scan",
    "CrossoverSample",
    "CrossoverResult",
    "find_crossover_q",
    "HybridFlowResult",
    "hybrid_flow",
    "HybridVsAlpha",
    "compare_hybrid_vs_alpha",
    "BatchRow", "batch_analyze", "batch_rows_to_csv",
]


# ---------------- Hybrid (multi-path) execution using flow() -----------------

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, List, Optional, Callable

@dataclass(frozen=True)
class HybridFlowResult:
    total_in: Decimal
    total_out: Decimal
    alpha_obs: Decimal                 # observed AMM share of OUT
    price_avg: Decimal                 # total_in / total_out
    # Optional fee totals (filled when an AMM instance is provided for fee previews)
    fee_pool_total: Decimal = Decimal(0)
    fee_tr_in_total: Decimal = Decimal(0)
    fee_tr_out_total: Decimal = Decimal(0)
def _wrap_apply_sink_with_fee_meter(amm: AMM | None,
                                    inner: Optional[Callable[[Decimal, Decimal], None]],
                                    fee_meter: Dict[str, Decimal]) -> Optional[Callable[[Decimal, Decimal], None]]:
    """Wrap apply_sink so that AMM fees are previewed and accumulated before delegating.

    This does **not** mutate pool state beyond whatever `inner` does; it only accumulates
    fee previews via AMM.preview_fees_for_fill(dx, dy) if an AMM instance is provided.
    """
    if amm is None and inner is None:
        return None
    def wrapped(dx: Decimal, dy: Decimal) -> None:
        if amm is not None:
            fp, fi, fo = amm.preview_fees_for_fill(dx, dy)
            fee_meter['fee_pool'] = fee_meter.get('fee_pool', Decimal(0)) + fp
            fee_meter['fee_tr_in'] = fee_meter.get('fee_tr_in', Decimal(0)) + fi
            fee_meter['fee_tr_out'] = fee_meter.get('fee_tr_out', Decimal(0)) + fo
        if inner is not None:
            inner(dx, dy)
    return wrapped


class _ClobStepForHybrid(Step):
    """Flow step that executes CLOB-only via router; tracks contribution."""
    def __init__(self,
                 segments_provider: Callable[[], Iterable[Segment]],
                 *,
                 limit_quality: Optional[Decimal] = None,
                 meter: dict):
        self._segments_provider = segments_provider
        self._limit_quality = limit_quality
        self._meter = meter
        self._cached_in = Decimal(0)
        self._cached_out = Decimal(0)

    def quality_upper_bound(self) -> Decimal:
        segs = list(self._segments_provider())
        if not segs:
            return Decimal(0)
        return max(s.quality for s in segs)

    def rev(self, sandbox: PaymentSandbox, out_req: Decimal):
        # Reverse pass: dry-run router without AMM hooks and without writebacks
        segs = list(self._segments_provider())
        res = run_trade_mode(
            ExecutionMode.CLOB_ONLY,
            target_out=out_req,
            segments=segs,
            limit_quality=self._limit_quality,
            amm_anchor=None,
            amm_curve=None,
            amm_context=None,
        )
        self._cached_in, self._cached_out = res.spent_in, res.filled_out
        return self._cached_in, self._cached_out

    def fwd(self, sandbox: PaymentSandbox, in_cap: Decimal):
        segs = list(self._segments_provider())
        res = run_trade_mode(
            ExecutionMode.CLOB_ONLY,
            target_out=in_cap,   # BookStepAdapter collapses; here we mirror cap-as-target
            segments=segs,
            limit_quality=self._limit_quality,
            amm_anchor=None,
            amm_curve=None,
            amm_context=None,
        )
        self._cached_in, self._cached_out = res.spent_in, res.filled_out
        # Track contribution
        self._meter.setdefault('clob_out', Decimal(0))
        self._meter['clob_out'] += self._cached_out
        return self._cached_in, self._cached_out


class _AmmStepForHybrid(Step):
    """Flow step that executes AMM-only via router; tracks contribution.

    For quality_upper_bound(), we probe amm_curve with a tiny target to get an
    upper-bound quality snapshot without mutating state.
    """
    def __init__(self,
                 amm_anchor: Optional[Callable[[Decimal, Decimal], Optional[Segment]]],
                 amm_curve: Optional[Callable[[Decimal], Iterable[Segment]]],
                 *,
                 amm_context: Optional[AMMContext] = None,
                 limit_quality: Optional[Decimal] = None,
                 probe: Decimal = Decimal("1"),
                 meter: dict):
        self._amm_anchor = amm_anchor
        self._amm_curve = amm_curve
        self._amm_context = amm_context
        self._limit_quality = limit_quality
        self._probe = probe
        self._meter = meter
        self._cached_in = Decimal(0)
        self._cached_out = Decimal(0)

    def quality_upper_bound(self) -> Decimal:
        if self._amm_curve is None:
            return Decimal(0)
        try:
            segs = list(self._amm_curve(self._probe))
        except TypeError:
            segs = []
        segs = [s for s in segs if s and s.out_max > 0 and s.in_at_out_max > 0 and s.quality > 0]
        if not segs:
            return Decimal(0)
        return max(s.quality for s in segs)

    def rev(self, sandbox: PaymentSandbox, out_req: Decimal):
        # Reverse pass: AMM-only, bootstrap via amm_curve inside router; no writebacks
        res = run_trade_mode(
            ExecutionMode.AMM_ONLY,
            target_out=out_req,
            segments=[],
            limit_quality=self._limit_quality,
            amm_anchor=self._amm_anchor,
            amm_curve=self._amm_curve,
            amm_context=self._amm_context,
        )
        self._cached_in, self._cached_out = res.spent_in, res.filled_out
        return self._cached_in, self._cached_out

    def fwd(self, sandbox: PaymentSandbox, in_cap: Decimal):
        res = run_trade_mode(
            ExecutionMode.AMM_ONLY,
            target_out=in_cap,
            segments=[],
            limit_quality=self._limit_quality,
            amm_anchor=self._amm_anchor,
            amm_curve=self._amm_curve,
            amm_context=self._amm_context,
        )
        self._cached_in, self._cached_out = res.spent_in, res.filled_out
        # Track contribution
        self._meter.setdefault('amm_out', Decimal(0))
        self._meter['amm_out'] += self._cached_out
        return self._cached_in, self._cached_out


def hybrid_flow(
    *,
    target_out: Decimal,
    clob_segments: Iterable[Segment] | List[Segment],
    amm_anchor: Optional[Callable[[Decimal, Decimal], Optional[Segment]]],
    amm_curve: Optional[Callable[[Decimal], Iterable[Segment]]],
    amm_context: Optional[AMMContext] = None,
    send_max: Optional[Decimal] = None,
    deliver_min: Optional[Decimal] = None,
    limit_quality: Optional[Decimal] = None,
    apply_sink: Optional[Callable[[Decimal, Decimal], None]] = None,
    amm_for_fees: Optional[AMM] = None,
) -> HybridFlowResult:
    """Run hybrid multi-path execution (CLOB vs AMM) via flow().

    Returns totals plus observed AMM share alpha_obs.
    """
    segs_list = list(clob_segments)
    meter: dict = {}
    fee_meter: Dict[str, Decimal] = {}

    # Shared sandbox and context
    sb = PaymentSandbox()
    ctx = amm_context or AMMContext(False)
    ctx.set_multi_path(True)

    clob_step = _ClobStepForHybrid(lambda: segs_list, limit_quality=limit_quality, meter=meter)
    amm_step = _AmmStepForHybrid(amm_anchor, amm_curve, amm_context=ctx, limit_quality=limit_quality, meter=meter)

    total_in, total_out = flow(
        sb,
        strands=[[clob_step], [amm_step]],
        out_req=target_out,
        send_max=send_max,
        limit_quality=limit_quality,
        amm_context=ctx,
        apply_sink=_wrap_apply_sink_with_fee_meter(amm_for_fees, apply_sink, fee_meter),
    )

    amm_out = meter.get('amm_out', Decimal(0))
    alpha_obs = (amm_out / total_out) if total_out > 0 else Decimal(0)
    price_avg = (total_in / total_out) if total_out > 0 else Decimal(0)

    fee_pool_total = fee_meter.get('fee_pool', Decimal(0))
    fee_tr_in_total = fee_meter.get('fee_tr_in', Decimal(0))
    fee_tr_out_total = fee_meter.get('fee_tr_out', Decimal(0))

    return HybridFlowResult(
        total_in=total_in,
        total_out=total_out,
        alpha_obs=alpha_obs,
        price_avg=price_avg,
        fee_pool_total=fee_pool_total,
        fee_tr_in_total=fee_tr_in_total,
        fee_tr_out_total=fee_tr_out_total,
    )


@dataclass(frozen=True)
class HybridVsAlpha:
    alpha_obs: Decimal
    alpha_star: Decimal
    price_hybrid: Decimal
    price_alpha_star: Decimal
    hybrid: HybridFlowResult
    alpha_scan: AlphaScanResult


def compare_hybrid_vs_alpha(
    *,
    target_out: Decimal,
    segments: Iterable[Segment] | List[Segment],
    amm_anchor: Optional[Callable[[Decimal, Decimal], Optional[Segment]]],
    amm_curve: Optional[Callable[[Decimal], Iterable[Segment]]],
    amm_context: Optional[AMMContext] = None,
    send_max: Optional[Decimal] = None,
    deliver_min: Optional[Decimal] = None,
    limit_quality: Optional[Decimal] = None,
    apply_sink: Optional[Callable[[Decimal, Decimal], None]] = None,
    alpha_step: Decimal = ROUTING_CFG.alpha_step_default,
    amm_for_fees: Optional[AMM] = None,
) -> HybridVsAlpha:
    """Run hybrid flow (inner selection) and alpha scan (outer split) and compare."""
    hyb = hybrid_flow(
        target_out=target_out,
        clob_segments=segments,
        amm_anchor=amm_anchor,
        amm_curve=amm_curve,
        amm_context=amm_context,
        send_max=send_max,
        deliver_min=deliver_min,
        limit_quality=limit_quality,
        apply_sink=apply_sink,
        amm_for_fees=amm_for_fees,
    )
    scan = analyze_alpha_scan(
        target_out=target_out,
        segments=segments,
        amm_anchor=amm_anchor,
        amm_curve=amm_curve,
        amm_context=amm_context,
        send_max=send_max,
        deliver_min=deliver_min,
        step=alpha_step,
        amm_for_fees=amm_for_fees,
    )
    return HybridVsAlpha(
        alpha_obs=hyb.alpha_obs,
        alpha_star=scan.alpha_star,
        price_hybrid=hyb.price_avg,
        price_alpha_star=scan.best_price,
        hybrid=hyb,
        alpha_scan=scan,
    )
# ---- Batch analysis helpers ----


@dataclass(frozen=True)
class BatchRow:
    q: Decimal
    mode: str            # 'CLOB', 'AMM', or 'HYBRID'
    total_out: Decimal
    total_in: Decimal
    avg_price: Decimal
    fee_pool_total: Decimal
    fee_tr_in_total: Decimal
    fee_tr_out_total: Decimal


def batch_analyze(
    *,
    q_list: Sequence[Decimal],
    segments: Iterable[Segment] | List[Segment],
    amm_anchor: Optional[Callable[[Decimal, Decimal], Optional[Segment]]],
    amm_curve: Optional[Callable[[Decimal], Iterable[Segment]]],
    amm_context: Optional[AMMContext] = None,
    send_max: Optional[Decimal] = None,
    deliver_min: Optional[Decimal] = None,
    limit_quality: Optional[Decimal] = None,
    amm_for_fees: Optional[AMM] = None,
) -> List[BatchRow]:
    """Run a batch over target_out values for CLOB-only, AMM-only, and Hybrid modes.

    Returns a flat list of rows suitable for tabular export.
    """
    rows: List[BatchRow] = []
    segs_list = list(segments)
    for q in q_list:
        # CLOB-only
        res_c = run_trade_mode(
            ExecutionMode.CLOB_ONLY,
            target_out=q,
            segments=segs_list,
            send_max=send_max,
            deliver_min=deliver_min,
            limit_quality=limit_quality,
            amm_anchor=None,
            amm_curve=None,
            amm_context=amm_context,
        )
        rows.append(BatchRow(
            q=q, mode="CLOB", total_out=res_c.filled_out, total_in=res_c.spent_in,
            avg_price=(res_c.spent_in / res_c.filled_out) if res_c.filled_out > 0 else Decimal(0),
            fee_pool_total=Decimal(0), fee_tr_in_total=Decimal(0), fee_tr_out_total=Decimal(0),
        ))
        # AMM-only
        fee_meter: Dict[str, Decimal] = {}
        res_a = run_trade_mode(
            ExecutionMode.AMM_ONLY,
            target_out=q,
            segments=segs_list,
            send_max=send_max,
            deliver_min=deliver_min,
            limit_quality=limit_quality,
            amm_anchor=amm_anchor,
            amm_curve=amm_curve,
            amm_context=amm_context,
            apply_sink=_wrap_apply_sink_with_fee_meter(amm_for_fees, None, fee_meter),
        )
        rows.append(BatchRow(
            q=q, mode="AMM", total_out=res_a.filled_out, total_in=res_a.spent_in,
            avg_price=(res_a.spent_in / res_a.filled_out) if res_a.filled_out > 0 else Decimal(0),
            fee_pool_total=fee_meter.get('fee_pool', Decimal(0)),
            fee_tr_in_total=fee_meter.get('fee_tr_in', Decimal(0)),
            fee_tr_out_total=fee_meter.get('fee_tr_out', Decimal(0)),
        ))
        # HYBRID
        hyb = hybrid_flow(
            target_out=q,
            clob_segments=segs_list,
            amm_anchor=amm_anchor,
            amm_curve=amm_curve,
            amm_context=amm_context,
            send_max=send_max,
            deliver_min=deliver_min,
            limit_quality=limit_quality,
            apply_sink=None,
            amm_for_fees=amm_for_fees,
        )
        rows.append(BatchRow(
            q=q, mode="HYBRID", total_out=hyb.total_out, total_in=hyb.total_in,
            avg_price=hyb.price_avg,
            fee_pool_total=hyb.fee_pool_total, fee_tr_in_total=hyb.fee_tr_in_total, fee_tr_out_total=hyb.fee_tr_out_total,
        ))
    return rows


def batch_rows_to_csv(rows: Sequence[BatchRow]) -> str:
    """Render rows into a simple CSV string (header included)."""
    headers = [
        "q", "mode", "total_out", "total_in", "avg_price",
        "fee_pool_total", "fee_tr_in_total", "fee_tr_out_total",
    ]
    def _fmt(x: Decimal) -> str:
        return format(x, 'f')
    lines = [",".join(headers)]
    for r in rows:
        lines.append(
            ",".join([
                _fmt(r.q), r.mode, _fmt(r.total_out), _fmt(r.total_in), _fmt(r.avg_price),
                _fmt(r.fee_pool_total), _fmt(r.fee_tr_in_total), _fmt(r.fee_tr_out_total),
            ])
        )
    return "\n".join(lines)