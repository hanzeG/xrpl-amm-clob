# tests/example/test_examples_from_whitepaper.py
# Whitepaper-aligned, story-style examples (Alice/Bob/Carol etc.)
# Each test references the section it corresponds to for easy cross-checking.

from decimal import Decimal
def _fmt(x: Decimal) -> str:
    return format(x, '.9f')
import pytest

from xrpl_router.exec_modes import run_trade_mode, ExecutionMode
from xrpl_router.amm_context import AMMContext
from xrpl_router.book_step import BookStep
from xrpl_router.flow import PaymentSandbox
from xrpl_router.amm import AMM

# We reuse fixtures defined in tests/conftest.py:
# - clob_segments_default
# - amm_curve_real
# - amm_anchor_real


# §1.2.4 / §1.2.5 — Alice → Carol: single-path CLOB payment
# Intent: only CLOB is available; average price is non-negative; route executes end-to-end.
# Note: run_trade_mode encapsulates rev/fwd passes for this experiment-style check.
@pytest.mark.parametrize("amount", [Decimal("20"), Decimal("50")])
def test_wp_alice_to_carol_clob_only(clob_segments_default, amount):
    print("[WP §1.2.4/§1.2.5] Alice→Carol via CLOB_ONLY, amount=", amount)
    res = run_trade_mode(
        ExecutionMode.CLOB_ONLY,
        target_out=amount,
        segments=clob_segments_default,
    )
    er = res.report
    assert er is not None
    assert er.total_out > 0
    assert er.avg_price >= 0
    print(f"  -> mode=CLOB_ONLY | out_req={_fmt(amount)} | filled={_fmt(er.total_out)} | avg_price={_fmt(er.avg_price)}")


# §1.2.7.1 — Bob uses AMM: single-path swap (no Fibonacci slicing)
# We keep AMMContext single-path to reflect whitepaper: Fibonacci is for multi-path.
@pytest.mark.parametrize("amount", [Decimal("20"), Decimal("80"), Decimal("160")])
def test_wp_bob_amm_single_path(amm_curve_real, clob_segments_default, amount):
    print("[WP §1.2.7.1] Bob via AMM_ONLY single-path, amount=", amount)
    ctx = AMMContext(False)  # single-path
    res = run_trade_mode(
        ExecutionMode.AMM_ONLY,
        target_out=amount,
        segments=clob_segments_default,  # ignored in AMM_ONLY
        amm_curve=amm_curve_real,
        amm_context=ctx,
    )
    er = res.report
    assert er is not None and er.total_out > 0
    # Marginal prices per iteration should be non-decreasing within the route.
    marginals = [it.price_effective for it in er.iterations if it.out_filled > 0]
    for a, b in zip(marginals, marginals[1:]):
        assert b >= a
    print(f"  -> mode=AMM_ONLY(single-path) | out_req={_fmt(amount)} | filled={_fmt(er.total_out)} | avg_price={_fmt(er.avg_price)}")
    _m = [it.price_effective for it in er.iterations if it.out_filled > 0]
    _m_show = ','.join(_fmt(m) for m in _m[:3])
    print(f"     marginals[0:3]={_m_show} (non-decreasing expected)")


# §1.2.7.2 — Coexistence at equal quality: prefer CLOB
# We anchor AMM at (or infinitesimally below) the LOB top quality; only CLOB should be consumed.
@pytest.mark.parametrize("amount", [Decimal("40")])
def test_wp_coexistence_prefer_clob(clob_segments_default, amm_anchor_real, amount):
    print("[WP §1.2.7.2] CLOB vs AMM coexistence at equal quality (prefer CLOB), amount=", amount)
    ctx = AMMContext(False)  # single-path enough; tie-break prefers CLOB
    res = run_trade_mode(
        ExecutionMode.HYBRID,
        target_out=amount,
        segments=clob_segments_default,
        amm_anchor=amm_anchor_real,
        amm_context=ctx,
    )
    er = res.report
    assert er is not None and er.total_out > 0
    # AMM not used → counter stays zero (tie-break prefers CLOB)
    assert ctx.ammUsedIters == 0
    # Match CLOB-only outcome at this amount (within numerical tolerance)
    res_clob = run_trade_mode(
        ExecutionMode.CLOB_ONLY,
        target_out=amount,
        segments=clob_segments_default,
    )
    er_clob = res_clob.report
    assert er_clob is not None
    # total_out must match; avg_price equal within tiny tolerance
    assert er.total_out == er_clob.total_out
    diff = (er.avg_price - er_clob.avg_price).copy_abs()
    assert diff <= Decimal("1e-12")
    print(
        "  -> HYBRID(tie) vs CLOB_ONLY | out_req=" + _fmt(amount) +
        " | hybrid(filled=" + _fmt(er.total_out) + ", avg=" + _fmt(er.avg_price) + ")" +
        " | clob_only(filled=" + _fmt(er_clob.total_out) + ", avg=" + _fmt(er_clob.avg_price) + ")"
    )


# §1.2.7.3 — Multi-path with Fibonacci sizing (advance only when AMM consumed; ≤ 30)
# Here we explicitly enable multi-path flag in AMMContext to reflect the whitepaper setting.
def test_wp_multipath_fibonacci(clob_segments_default, amm_anchor_real):
    print("[WP §1.2.7.3] Multi-path with Fibonacci sizing (HYBRID), target_out=", end="")
    ctx = AMMContext(False)
    ctx.setMultiPath(True)  # enable whitepaper multi-path semantics
    clob_total = sum(seg.out_max for seg in clob_segments_default)
    amount = clob_total + Decimal("10")  # ensure CLOB alone cannot satisfy
    print(amount)

    def competitive_amm_curve(target_out: Decimal):
        amm = AMM(
            x_reserve=Decimal("5000"),
            y_reserve=Decimal("5000"),  # SPQ ≈ 1.0 to be competitive with LOB top
            fee=Decimal("0.003"),
            x_is_xrp=True,
            y_is_xrp=False,
            tr_in=Decimal("0"),
            tr_out=Decimal("0"),
        )
        return list(amm.segments_for_out(target_out, max_segments=30, start_fraction=Decimal("5e-2")))

    res = run_trade_mode(
        ExecutionMode.HYBRID,
        target_out=amount,
        segments=clob_segments_default,
        amm_curve=competitive_amm_curve,
        amm_anchor=amm_anchor_real,
        amm_context=ctx,
    )
    er = res.report
    assert er is not None and er.total_out > 0
    # Hybrid should fill beyond pure CLOB capacity if AMM contributes
    clob_cap = sum(seg.out_max for seg in clob_segments_default)
    assert er.total_out + Decimal("1e-12") >= clob_cap  # at least reach cap (tolerate rounding)
    assert er.total_out > clob_cap - Decimal("1e-9")   # and be effectively at/above cap
    print(
        "  -> HYBRID(multipath) | out_req=" + _fmt(amount) +
        " | clob_cap=" + _fmt(clob_cap) +
        " | filled=" + _fmt(er.total_out) +
        " (expect ≥ clob_cap; AMM provides remainder)"
    )


# §1.3.2.4 — Execution-time fees (CLOB out-side issuer fees staged to sandbox)
# We use a simple hook that stages a nominal fee; production builds can inject real fee logic.
@pytest.mark.parametrize("amount", [Decimal("50")])
def test_wp_execution_time_fees_clob(clob_segments_default, amount):
    print("[WP §1.3.2.4] Execution-time fees on CLOB out-side, amount=", amount)
    staged = {"fee_calls": 0, "fee_count": 0}

    def fee_hook(trace, sandbox):
        # Minimal demonstration: whenever we see a CLOB slice, stage a small fee.
        # In production, the hook would compute per-slice issuer fees and stage them.
        any_clob = any(item.get("src") == "CLOB" and item.get("take_out", Decimal(0)) > 0 for item in trace)
        if any_clob:
            staged["fee_calls"] += 1
            sandbox.stage_fee(Decimal(0), Decimal("-0.1"))
            staged["fee_count"] += 1

    ctx = AMMContext(False)
    # Build a BookStep with fee_hook and run rev→fwd manually
    step = BookStep(segments_provider=lambda: clob_segments_default, amm_context=ctx, fee_hook=fee_hook)
    # Reverse pass: determine required IN for target OUT
    need_in, got_out = step.rev(Decimal("1e18"), amount)
    assert got_out > 0 and need_in > 0
    # Forward pass: execute with a sandbox to stage fees
    sandbox = PaymentSandbox()
    spent_in, filled_out = step.fwd(in_cap=need_in, sandbox=sandbox)
    assert filled_out > 0 and spent_in > 0
    # Our hook should have run and staged at least one fee entry
    assert staged["fee_calls"] >= 1
    assert any(dy < 0 for (_, dy) in getattr(sandbox, "staged", []))
    _staged = getattr(sandbox, 'staged', [])
    print(f"  -> fee_hook calls={staged['fee_calls']} | staged_entries={len(_staged)}")