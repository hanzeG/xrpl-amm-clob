from decimal import Decimal

from xrpl_router.core.fmt import amount_to_decimal
from xrpl_router.book_offers import build_clob_segments_from_offers

RUSD_HEX = "524C555344000000000000000000000000000000"
XRP = "XRP"


def test_builder_keeps_in_amount_without_quality_roundtrip_drift():
    # IN is exactly 4.336217 XRP (4336217 drops).
    offers = [
        {
            "index": "ABCDEF0000000000000000000000000000009075",
            "TakerGets": {"currency": RUSD_HEX, "value": "8.412263"},
            "TakerPays": "4336217",
        }
    ]

    segs = build_clob_segments_from_offers(offers, in_cur=XRP, out_cur=RUSD_HEX)

    assert len(segs) == 1
    assert amount_to_decimal(segs[0].out_max) == Decimal("8.412263")
    assert amount_to_decimal(segs[0].in_at_out_max) == Decimal("4.336217")


def test_builder_stable_sort_keeps_source_id_alignment_on_ties():
    offers = [
        {
            "index": "000000000000000000000000000000000000AAAA",
            "TakerGets": {"currency": RUSD_HEX, "value": "10"},
            "TakerPays": "5000000",
        },
        {
            "index": "000000000000000000000000000000000000BBBB",
            "TakerGets": {"currency": RUSD_HEX, "value": "8"},
            "TakerPays": "4000000",
        },
        {
            "index": "000000000000000000000000000000000000CCCC",
            "TakerGets": {"currency": RUSD_HEX, "value": "9"},
            "TakerPays": "5000000",
        },
    ]

    segs = build_clob_segments_from_offers(offers, in_cur=XRP, out_cur=RUSD_HEX)
    ids = [str(s.source_id)[-4:] for s in segs]

    # AAAA and BBBB have equal quality; stable sort should preserve input order.
    assert ids == ["AAAA", "BBBB", "CCCC"]
    assert amount_to_decimal(segs[0].out_max) == Decimal("10")
    assert amount_to_decimal(segs[1].out_max) == Decimal("8")

