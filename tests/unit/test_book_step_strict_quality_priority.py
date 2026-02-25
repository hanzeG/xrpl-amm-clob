from decimal import Decimal

from xrpl_router.amm import AMM
from xrpl_router.book_step import _execute_one_iteration
from xrpl_router.core import Quality, XRPAmount
from xrpl_router.core.datatypes import Segment


def _mk_clob_segment(q: Quality) -> Segment:
    return Segment(
        src="CLOB",
        quality=q,
        out_max=XRPAmount(100),
        in_at_out_max=XRPAmount(100),
        source_id="clob-1",
    )


def _mk_amm() -> AMM:
    return AMM(
        Decimal("1000"),
        Decimal("1000"),
        Decimal("0"),
        x_is_xrp=True,
        y_is_xrp=True,
    )


def test_equal_spq_does_not_call_anchor_and_prefers_clob():
    q_top = Quality.from_amounts(XRPAmount(100), XRPAmount(100))
    clob = _mk_clob_segment(q_top)
    amm = _mk_amm()
    anchor_calls = []

    def anchor(q, cap):
        anchor_calls.append((q, cap))
        return Segment(
            src="AMM",
            quality=q,
            out_max=XRPAmount(100),
            in_at_out_max=XRPAmount(100),
            source_id="amm-1",
        )

    _, _, _, amm_used, _, _, _, trace = _execute_one_iteration(
        [clob],
        target_out=XRPAmount(50),
        send_max=None,
        limit_quality=None,
        amm_anchor=anchor,
        amm_spq=q_top,  # equal to CLOB top quality
        amm_obj=amm,
    )

    assert len(anchor_calls) == 0
    assert amm_used is False
    assert len(trace) == 1
    assert trace[0]["src"] == "CLOB"


def test_strictly_better_spq_calls_anchor_and_can_use_amm():
    q_top = Quality.from_amounts(XRPAmount(100), XRPAmount(100))
    q_better = Quality.from_amounts(XRPAmount(101), XRPAmount(100))
    clob = _mk_clob_segment(q_top)
    amm = _mk_amm()
    anchor_calls = []

    def anchor(q, cap):
        anchor_calls.append((q, cap))
        return Segment(
            src="AMM",
            quality=q,
            out_max=XRPAmount(100),
            in_at_out_max=XRPAmount(100),
            source_id="amm-1",
        )

    _, _, _, amm_used, _, _, _, trace = _execute_one_iteration(
        [clob],
        target_out=XRPAmount(50),
        send_max=None,
        limit_quality=None,
        amm_anchor=anchor,
        amm_spq=q_better,  # strictly better than CLOB top quality
        amm_obj=amm,
    )

    assert len(anchor_calls) == 1
    assert amm_used is True
    assert len(trace) == 1
    assert trace[0]["src"] == "AMM"

