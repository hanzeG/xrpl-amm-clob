"""Build CLOB segments directly from XRPL book offer payloads.

This module avoids lossy round-trips like:
  (out, in) -> quality -> price -> in
which can introduce +1 grid-unit drift near rounding boundaries.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR
from typing import Any, Iterable, List

from .core import IOUAmount, XRPAmount
from .core.constants import IOU_QUANTUM, XRP_QUANTUM
from .core.datatypes import Segment
from .core.fmt import amount_to_decimal, quantize_down, quantize_up
from .core.quality import Quality

XRP = "XRP"


def _parse_offer_amount(a: Any) -> tuple[str, Decimal]:
    if isinstance(a, str):
        return XRP, Decimal(a) * XRP_QUANTUM
    if isinstance(a, dict):
        return str(a.get("currency")), Decimal(str(a.get("value")))
    raise ValueError(f"Unsupported amount payload: {a!r}")


def _book_offer_id(o: dict[str, Any]) -> str:
    if o.get("index"):
        return str(o.get("index"))
    if o.get("Account") is not None and o.get("Sequence") is not None:
        return f"{o.get('Account')}:{o.get('Sequence')}"
    return "unknown-offer-id"


def _amt_floor(cur: str, v: Decimal):
    if cur == XRP:
        drops = int((v / XRP_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
        return XRPAmount(value=drops)
    units = int((v / IOU_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
    return IOUAmount.from_components(units, -15)


def build_clob_segments_from_offers(
    offers: Iterable[dict[str, Any]],
    in_cur: str,
    out_cur: str,
    *,
    iou_in_transfer_rate: Decimal = Decimal("1"),
) -> List[Segment]:
    """Convert direction-aligned offers to CLOB segments with source ids.

    Rounding semantics:
    - OUT is rounded down on its native quantum.
    - IN is rounded up on its native quantum.
    - Quality is computed directly from snapped (out, in) amounts.
    """
    out: List[Segment] = []
    for o in offers:
        gets_cur, gets_val = _parse_offer_amount(o.get("TakerGets"))
        pays_cur, pays_val = _parse_offer_amount(o.get("TakerPays"))

        if "taker_gets_funded" in o:
            try:
                fg_cur, fg_val = _parse_offer_amount(o.get("taker_gets_funded"))
                if fg_cur == gets_cur:
                    gets_val = fg_val
            except Exception:
                pass
        if "taker_pays_funded" in o:
            try:
                fp_cur, fp_val = _parse_offer_amount(o.get("taker_pays_funded"))
                if fp_cur == pays_cur:
                    pays_val = fp_val
            except Exception:
                pass

        try:
            if "owner_funds" in o and Decimal(str(o.get("owner_funds"))) <= 0:
                continue
        except Exception:
            pass

        if gets_cur == out_cur and pays_cur == in_cur:
            out_max_dec = gets_val
            in_need_dec = pays_val
        elif pays_cur == out_cur and gets_cur == in_cur:
            out_max_dec = pays_val
            in_need_dec = gets_val
        else:
            continue
        if out_max_dec <= 0 or in_need_dec <= 0:
            continue

        if in_cur != XRP and iou_in_transfer_rate > 0:
            in_need_dec = in_need_dec * iou_in_transfer_rate

        out_q = XRP_QUANTUM if out_cur == XRP else IOU_QUANTUM
        in_q = XRP_QUANTUM if in_cur == XRP else IOU_QUANTUM

        out_max_dec = quantize_down(out_max_dec, out_q)
        in_need_dec = quantize_up(in_need_dec, in_q)
        if out_max_dec <= 0 or in_need_dec <= 0:
            continue

        out_amt = _amt_floor(out_cur, out_max_dec)
        in_amt = _amt_floor(in_cur, in_need_dec)
        if amount_to_decimal(out_amt) <= 0 or amount_to_decimal(in_amt) <= 0:
            continue

        q = Quality.from_amounts(out_amt, in_amt)
        out.append(
            Segment(
                src="CLOB",
                quality=q,
                out_max=out_amt,
                in_at_out_max=in_amt,
                raw_quality=q,
                source_id=_book_offer_id(o),
            )
        )

    # Stable sort by quality desc; equal-quality rows preserve input order.
    out.sort(key=lambda s: (s.quality.rate.exponent, s.quality.rate.mantissa), reverse=True)
    return out


__all__ = ["build_clob_segments_from_offers"]
