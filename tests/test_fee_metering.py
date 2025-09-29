from decimal import Decimal

import pytest

from xrpl_router.exec_modes import run_trade_mode, ExecutionMode
from xrpl_router.efficiency_scan import hybrid_flow, summarize_hybrid


# ---------------------------
# Fee metering consistency tests
# ---------------------------


def test_clob_only_has_zero_fees(clob_segments_default):
    res = run_trade_mode(
        ExecutionMode.CLOB_ONLY,
        target_out=Decimal("50"),
        segments=clob_segments_default,
    )
    er = res.report
    assert er is not None
    assert er.fee_pool_total == 0
    assert er.fee_tr_in_total == 0
    assert er.fee_tr_out_total == 0


def test_amm_only_has_positive_fees(clob_segments_default, amm_curve_default, amm_ctx_multipath, amm_instance_default):
    res = hybrid_flow(
        target_out=Decimal("50"),
        clob_segments=[],              # force AMM-only via hybrid wrapper
        amm_curve=amm_curve_default,
        amm_anchor=None,
        amm_context=amm_ctx_multipath,
        amm_for_fees=amm_instance_default,
    )
    total_fees = res.fee_pool_total + res.fee_tr_in_total + res.fee_tr_out_total
    assert total_fees > 0


def test_hybrid_flow_has_positive_fees(clob_segments_default, amm_curve_default, amm_anchor_default, amm_ctx_multipath, amm_instance_default):
    res = hybrid_flow(
        target_out=Decimal("300"),
        clob_segments=clob_segments_default,
        amm_curve=amm_curve_default,
        amm_anchor=amm_anchor_default,
        amm_context=amm_ctx_multipath,
        amm_for_fees=amm_instance_default,
    )
    row = summarize_hybrid(res)
    total_fees = row["fee_pool_total"] + row["fee_tr_in_total"] + row["fee_tr_out_total"]
    assert total_fees > 0
