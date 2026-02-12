#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract rUSD/XRP-related execution evidence from rippled ledger NDJSON.
Goal: "least-miss" (不漏项优先), with light matching:
- Candidate tx: any tx whose meta shows rUSD RippleState balance change OR
  has explicit rUSD/XRP in tx fields OR has Offer node legs involving rUSD/XRP OR
  has AMM signals (AMM ledger entry / AMMID / AMM-like fields).
- Output NDJSON: one line per ledger, containing:
  - ledger_index, close_time_iso, tx_total
  - tx_candidates: list of candidate tx summaries (rich enough for later filtering)
  - evidence: per-tx evidence blocks (rusd_trustline_deltas, xrp_accountroot_deltas, offer_fills, amm_signals)
This script is designed to avoid missing; you'll prune later.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

RUSD_HEX_DEFAULT = "524C555344000000000000000000000000000000"
RUSD_ISSUER_DEFAULT = "rMxCKbEDwqr76QuheSUMdEGf4B9xJ8m5De"

Amount = Union[str, Dict[str, Any]]  # XRP drops as str; IOU as dict


# -------------------------
# Amount helpers
# -------------------------

def is_xrp_amount(a: Any) -> bool:
    return isinstance(a, str) and a.isdigit()


def is_iou_amount(a: Any) -> bool:
    return isinstance(a, dict) and "currency" in a and "issuer" in a and "value" in a


def is_rusd_amount(a: Any, rusd_hex: str, rusd_issuer: str) -> bool:
    if not is_iou_amount(a):
        return False
    return str(a.get("currency")) == rusd_hex and str(a.get("issuer")) == rusd_issuer


def amount_to_key(a: Any) -> Tuple[str, Optional[str]]:
    if is_xrp_amount(a):
        return ("XRP", None)
    if is_iou_amount(a):
        return (str(a.get("currency")), str(a.get("issuer")))
    return ("", None)


def amount_to_float(a: Any) -> Optional[float]:
    try:
        if is_xrp_amount(a):
            return float(a)
        if is_iou_amount(a):
            return float(a.get("value"))
    except Exception:
        return None
    return None


def subtract_amount(prev: Any, final: Any) -> Optional[float]:
    # prev - final
    if is_xrp_amount(prev) and is_xrp_amount(final):
        try:
            return float(int(prev) - int(final))
        except Exception:
            return None
    if is_iou_amount(prev) and is_iou_amount(final):
        if amount_to_key(prev) != amount_to_key(final):
            return None
        try:
            return float(prev["value"]) - float(final["value"])
        except Exception:
            return None
    return None


# -------------------------
# Recursive scan utilities
# -------------------------

def iter_amount_like_objects(obj: Any) -> List[Amount]:
    out: List[Amount] = []

    def rec(x: Any):
        if is_xrp_amount(x):
            out.append(x)
            return
        if is_iou_amount(x):
            out.append(x)
            # still traverse nested to be safe
            for v in x.values():
                rec(v)
            return
        if isinstance(x, dict):
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)

    rec(obj)
    return out


def tx_has_rusd_anywhere(tx: Dict[str, Any], rusd_hex: str, rusd_issuer: str) -> bool:
    return any(is_rusd_amount(a, rusd_hex, rusd_issuer) for a in iter_amount_like_objects(tx))


def tx_has_xrp_anywhere(tx: Dict[str, Any]) -> bool:
    return any(is_xrp_amount(a) for a in iter_amount_like_objects(tx))


def offer_legs(tx: Dict[str, Any]) -> Optional[Tuple[Any, Any]]:
    tg = tx.get("TakerGets")
    tp = tx.get("TakerPays")
    if tg is None or tp is None:
        return None
    if not (is_xrp_amount(tg) or is_iou_amount(tg) or is_xrp_amount(tp) or is_iou_amount(tp)):
        return None
    return (tg, tp)


def is_rusd_xrp_pair_amounts(a: Any, b: Any, rusd_hex: str, rusd_issuer: str) -> bool:
    return (
        (is_rusd_amount(a, rusd_hex, rusd_issuer) and is_xrp_amount(b)) or
        (is_xrp_amount(a) and is_rusd_amount(b, rusd_hex, rusd_issuer))
    )


# -------------------------
# Ledger record helpers
# -------------------------

def extract_ledger_from_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(rec, dict):
        raise ValueError("record is not a dict")

    if "ledger" in rec and isinstance(rec["ledger"], dict):
        return rec["ledger"]

    r = rec.get("result")
    if isinstance(r, dict) and "ledger" in r and isinstance(r["ledger"], dict):
        return r["ledger"]

    return rec


def ledger_close_time_iso(ledger: Dict[str, Any]) -> str:
    for k in ("close_time_iso", "close_time_human", "close_time", "close_time_datetime"):
        v = ledger.get(k)
        if v is None:
            continue
        return str(v)
    return ""


# -------------------------
# Meta parsing: nodes
# -------------------------

def iter_affected_nodes(tx: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    meta = tx.get("metaData") or tx.get("meta")
    if not isinstance(meta, dict):
        return []
    nodes = meta.get("AffectedNodes")
    if not isinstance(nodes, list):
        return []

    out: List[Tuple[str, Dict[str, Any]]] = []
    for n in nodes:
        if not isinstance(n, dict) or not n:
            continue
        kind, body = next(iter(n.items()), (None, None))
        if kind in ("ModifiedNode", "DeletedNode", "CreatedNode") and isinstance(body, dict):
            out.append((kind, body))
    return out


def get_tx_index(tx: Dict[str, Any]) -> Optional[int]:
    meta = tx.get("metaData") or tx.get("meta")
    if isinstance(meta, dict):
        v = meta.get("TransactionIndex")
        try:
            return int(v) if v is not None else None
        except Exception:
            return None
    return None


# -------------------------
# Evidence extraction
# -------------------------

def extract_rusd_ripplestate_deltas(
    tx: Dict[str, Any],
    rusd_hex: str,
    rusd_issuer: str,
) -> List[Dict[str, Any]]:
    """
    Look at RippleState nodes:
      - Find nodes whose Balance currency == rusd_hex (and issuer matches either side).
      - Compute delta = prev - final (so + means balance decreased? depends on side; we keep raw).
    We keep raw fields to avoid missing; interpretation later.
    """
    out: List[Dict[str, Any]] = []
    for kind, body in iter_affected_nodes(tx):
        if body.get("LedgerEntryType") != "RippleState":
            continue

        prev = body.get("PreviousFields") if isinstance(body.get("PreviousFields"), dict) else {}
        final = body.get("FinalFields") if isinstance(body.get("FinalFields"), dict) else {}
        newf = body.get("NewFields") if isinstance(body.get("NewFields"), dict) else {}

        # Balance can be in FinalFields/NewFields/PreviousFields
        prev_bal = prev.get("Balance")
        final_bal = final.get("Balance", newf.get("Balance"))
        if prev_bal is None or final_bal is None:
            # created/deleted nodes may not have both; still record if any is rUSD
            bal_ref = final_bal if final_bal is not None else prev_bal
            if not is_iou_amount(bal_ref) or str(bal_ref.get("currency")) != rusd_hex:
                continue
            out.append({
                "node_kind": kind,
                "ledger_entry_type": "RippleState",
                "ledger_index": body.get("LedgerIndex"),
                "has_prev_balance": prev_bal is not None,
                "has_final_balance": final_bal is not None,
                "balance_ref": bal_ref,
                "high_limit": final.get("HighLimit", newf.get("HighLimit")),
                "low_limit": final.get("LowLimit", newf.get("LowLimit")),
                "flags": final.get("Flags", newf.get("Flags")),
            })
            continue

        if not (is_iou_amount(prev_bal) and is_iou_amount(final_bal)):
            continue
        if str(prev_bal.get("currency")) != rusd_hex or str(final_bal.get("currency")) != rusd_hex:
            continue

        # issuer match: Balance issuer is usually rrrrr... for RippleState; use limits to infer counterpart
        # We accept if either HighLimit or LowLimit issuer is the rusd_issuer (least-miss).
        high = final.get("HighLimit", newf.get("HighLimit"))
        low = final.get("LowLimit", newf.get("LowLimit"))
        issuer_match = False
        for lim in (high, low):
            if isinstance(lim, dict) and str(lim.get("currency")) == rusd_hex and str(lim.get("issuer")) == rusd_issuer:
                issuer_match = True
                break
        # If no match, still keep it (不漏项), but tag it.
        delta = subtract_amount(prev_bal, final_bal)

        out.append({
            "node_kind": kind,
            "ledger_entry_type": "RippleState",
            "ledger_index": body.get("LedgerIndex"),
            "issuer_match_via_limits": issuer_match,
            "prev_balance": prev_bal,
            "final_balance": final_bal,
            "delta_prev_minus_final": delta,
            "high_limit": high,
            "low_limit": low,
            "flags": final.get("Flags", newf.get("Flags")),
        })

    return out


def extract_xrp_accountroot_deltas(tx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    XRP balance changes appear in AccountRoot.Balance (drops string).
    We compute delta = prev - final (drops).
    """
    out: List[Dict[str, Any]] = []
    for kind, body in iter_affected_nodes(tx):
        if body.get("LedgerEntryType") != "AccountRoot":
            continue
        prev = body.get("PreviousFields") if isinstance(body.get("PreviousFields"), dict) else {}
        final = body.get("FinalFields") if isinstance(body.get("FinalFields"), dict) else {}
        newf = body.get("NewFields") if isinstance(body.get("NewFields"), dict) else {}

        prev_bal = prev.get("Balance")
        final_bal = final.get("Balance", newf.get("Balance"))
        acct = final.get("Account", newf.get("Account"))

        if prev_bal is None or final_bal is None:
            continue
        if not (is_xrp_amount(prev_bal) and is_xrp_amount(final_bal)):
            continue

        delta = subtract_amount(prev_bal, final_bal)
        out.append({
            "node_kind": kind,
            "ledger_entry_type": "AccountRoot",
            "ledger_index": body.get("LedgerIndex"),
            "account": acct,
            "prev_balance_drops": prev_bal,
            "final_balance_drops": final_bal,
            "delta_prev_minus_final_drops": delta,
        })
    return out


def extract_offer_node_fills(
    tx: Dict[str, Any],
    ledger_index: int,
    close_time_iso: str,
    rusd_hex: str,
    rusd_issuer: str,
) -> List[Dict[str, Any]]:
    """
    For Offer nodes (CLOB executions): use PreviousFields and FinalFields deltas on TakerGets/TakerPays.
    """
    meta = tx.get("metaData") or tx.get("meta")
    if not isinstance(meta, dict):
        return []

    nodes = meta.get("AffectedNodes")
    if not isinstance(nodes, list):
        return []

    out: List[Dict[str, Any]] = []
    tx_hash = tx.get("hash")
    tx_index = meta.get("TransactionIndex")

    for n in nodes:
        if not isinstance(n, dict) or not n:
            continue
        kind, body = next(iter(n.items()), (None, None))
        if kind not in ("ModifiedNode", "DeletedNode", "CreatedNode") or not isinstance(body, dict):
            continue
        if body.get("LedgerEntryType") != "Offer":
            continue

        prev = body.get("PreviousFields") if isinstance(body.get("PreviousFields"), dict) else {}
        final = body.get("FinalFields") if isinstance(body.get("FinalFields"), dict) else {}
        newf = body.get("NewFields") if isinstance(body.get("NewFields"), dict) else {}

        maker = None
        for src in (final, newf, body):
            if isinstance(src, dict) and "Account" in src:
                maker = src.get("Account")
                break

        prev_tg = prev.get("TakerGets")
        prev_tp = prev.get("TakerPays")
        final_tg = final.get("TakerGets")
        final_tp = final.get("TakerPays")

        # reference legs (to decide if rUSD/XRP)
        legs_ref = None
        if prev_tg is not None and prev_tp is not None:
            legs_ref = (prev_tg, prev_tp)
        elif final_tg is not None and final_tp is not None:
            legs_ref = (final_tg, final_tp)
        else:
            continue

        if not is_rusd_xrp_pair_amounts(legs_ref[0], legs_ref[1], rusd_hex, rusd_issuer):
            continue

        delta_tg = subtract_amount(prev_tg, final_tg) if (prev_tg is not None and final_tg is not None) else None
        delta_tp = subtract_amount(prev_tp, final_tp) if (prev_tp is not None and final_tp is not None) else None

        tg, tp = legs_ref
        direction = None
        if is_xrp_amount(tp) and is_rusd_amount(tg, rusd_hex, rusd_issuer):
            direction = "XRP->rUSD"
        elif is_rusd_amount(tp, rusd_hex, rusd_issuer) and is_xrp_amount(tg):
            direction = "rUSD->XRP"

        out.append({
            "ledger_index": ledger_index,
            "close_time_iso": close_time_iso,
            "tx_index": tx_index,
            "tx_hash": tx_hash,
            "node_kind": kind,
            "offer_ledger_index": body.get("LedgerIndex"),
            "maker_account": maker,
            "offer_taker_gets_currency": amount_to_key(tg)[0],
            "offer_taker_gets_issuer": amount_to_key(tg)[1],
            "offer_taker_pays_currency": amount_to_key(tp)[0],
            "offer_taker_pays_issuer": amount_to_key(tp)[1],
            "direction": direction,
            "executed_taker_gets_delta": delta_tg,
            "executed_taker_pays_delta": delta_tp,
        })

    return out


def extract_amm_signals(tx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Light AMM signals from tx/meta:
    - Any affected node with LedgerEntryType == "AMM"
    - Any AccountRoot with AMMID field in NewFields/FinalFields
    - Presence of fields like "AMMID" in tx/meta
    """
    has_amm_ledgerentry = False
    amm_ledger_entries: List[str] = []
    has_ammid_accountroot = False
    ammid_values: List[str] = []
    has_amm_in_tx_fields = False

    # scan nodes
    for _, body in iter_affected_nodes(tx):
        let = body.get("LedgerEntryType")
        if let == "AMM":
            has_amm_ledgerentry = True
            if body.get("LedgerIndex") is not None:
                amm_ledger_entries.append(body.get("LedgerIndex"))

        if let == "AccountRoot":
            final = body.get("FinalFields") if isinstance(body.get("FinalFields"), dict) else {}
            newf = body.get("NewFields") if isinstance(body.get("NewFields"), dict) else {}
            for src in (final, newf):
                if isinstance(src, dict) and "AMMID" in src:
                    has_ammid_accountroot = True
                    v = src.get("AMMID")
                    if isinstance(v, str):
                        ammid_values.append(v)

    # quick string-key scan of tx dict (not too expensive; tx is small)
    # we avoid deep recursion for performance, but keep minimal.
    def has_key_anywhere(obj: Any, key: str) -> bool:
        if isinstance(obj, dict):
            if key in obj:
                return True
            return any(has_key_anywhere(v, key) for v in obj.values())
        if isinstance(obj, list):
            return any(has_key_anywhere(v, key) for v in obj)
        return False

    if has_key_anywhere(tx, "AMMID") or has_key_anywhere(tx, "AMM") or has_key_anywhere(tx, "AuctionSlot"):
        has_amm_in_tx_fields = True

    return {
        "has_amm_ledgerentry": has_amm_ledgerentry,
        "amm_ledger_entries": amm_ledger_entries,
        "has_ammid_accountroot": has_ammid_accountroot,
        "ammid_values": list(dict.fromkeys(ammid_values)),  # unique preserve order
        "has_amm_in_tx_fields": has_amm_in_tx_fields,
    }


# -------------------------
# Candidate decision (least-miss)
# -------------------------

def is_candidate_tx(
    tx: Dict[str, Any],
    rusd_hex: str,
    rusd_issuer: str,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Candidate if any of:
    - rUSD RippleState delta exists
    - Offer fill exists for rUSD/XRP
    - tx mentions both rUSD and XRP anywhere
    - AMM signals exist (for later linking)
    """
    rusd_state = extract_rusd_ripplestate_deltas(tx, rusd_hex, rusd_issuer)
    offer_fills = extract_offer_node_fills(tx, -1, "", rusd_hex, rusd_issuer)  # ledger/time filled later
    amm_sig = extract_amm_signals(tx)
    has_both_anywhere = tx_has_rusd_anywhere(tx, rusd_hex, rusd_issuer) and tx_has_xrp_anywhere(tx)

    ok = (
        len(rusd_state) > 0 or
        len(offer_fills) > 0 or
        has_both_anywhere or
        amm_sig.get("has_amm_ledgerentry") or
        amm_sig.get("has_ammid_accountroot") or
        amm_sig.get("has_amm_in_tx_fields")
    )

    reason = {
        "has_rusd_ripplestate_delta": len(rusd_state) > 0,
        "has_offer_fills_rusd_xrp": len(offer_fills) > 0,
        "has_rusd_and_xrp_anywhere": has_both_anywhere,
        "amm_signals": amm_sig,
    }
    return ok, reason


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-ndjson", required=True, help="Input NDJSON, one ledger record per line (from fetch_ledger_dump_and_csv.py range mode)")
    ap.add_argument("--out-ndjson", required=True, help="Output NDJSON, one processed ledger per line")
    ap.add_argument("--rusd-hex", default=RUSD_HEX_DEFAULT)
    ap.add_argument("--rusd-issuer", default=RUSD_ISSUER_DEFAULT)
    ap.add_argument("--max-ledgers", type=int, default=0, help="0 means no limit; otherwise stop after N ledgers")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_ndjson) or ".", exist_ok=True)

    n_in = 0
    n_out = 0
    n_err = 0

    with open(args.in_ndjson, "r", encoding="utf-8") as fin, open(args.out_ndjson, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            n_in += 1
            if args.max_ledgers and n_in > args.max_ledgers:
                break

            try:
                rec = json.loads(line)
                ledger = extract_ledger_from_record(rec)

                ledger_index = int(ledger.get("ledger_index"))
                close_iso = ledger_close_time_iso(ledger)

                txs = ledger.get("transactions", [])
                if not isinstance(txs, list):
                    raise RuntimeError("ledger.transactions is not a list")

                tx_candidates: List[Dict[str, Any]] = []

                # ledger-level counts (helpful sanity)
                total_offer_fill_rows = 0
                total_rusd_state_rows = 0
                total_xrp_root_rows = 0
                total_amm_ledgerentry_txs = 0

                for tx in txs:
                    if not isinstance(tx, dict):
                        continue

                    tx_hash = tx.get("hash")
                    tx_type = tx.get("TransactionType")
                    account = tx.get("Account")
                    tx_index = get_tx_index(tx)

                    ok, reason = is_candidate_tx(tx, args.rusd_hex, args.rusd_issuer)
                    if not ok:
                        continue

                    # evidence blocks
                    rusd_state = extract_rusd_ripplestate_deltas(tx, args.rusd_hex, args.rusd_issuer)
                    xrp_root = extract_xrp_accountroot_deltas(tx)
                    offer_fills = extract_offer_node_fills(tx, ledger_index, close_iso, args.rusd_hex, args.rusd_issuer)
                    amm_sig = reason.get("amm_signals") or extract_amm_signals(tx)

                    # enrich offer fills with correct ledger/time already passed; ensure filled
                    for f in offer_fills:
                        f["ledger_index"] = ledger_index
                        f["close_time_iso"] = close_iso

                    total_offer_fill_rows += len(offer_fills)
                    total_rusd_state_rows += len(rusd_state)
                    total_xrp_root_rows += len(xrp_root)
                    if amm_sig.get("has_amm_ledgerentry"):
                        total_amm_ledgerentry_txs += 1

                    tx_candidates.append({
                        "ledger_index": ledger_index,
                        "close_time_iso": close_iso,
                        "tx_index": tx_index,
                        "tx_hash": tx_hash,
                        "tx_type": tx_type,
                        "account": account,
                        "candidate_reason": reason,
                        # “不漏项”原则：尽量把证据都留着（后续你再严格筛）
                        "evidence": {
                            "rusd_ripplestate_deltas": rusd_state,
                            "xrp_accountroot_deltas": xrp_root,
                            "offer_node_fills_rusd_xrp": offer_fills,
                            "amm_signals": amm_sig,
                            # small extra: delivered_amount if present
                            "delivered_amount": (tx.get("metaData") or tx.get("meta") or {}).get("delivered_amount"),
                        }
                    })

                out_obj = {
                    "ledger_index": ledger_index,
                    "close_time_iso": close_iso,
                    "counts": {
                        "tx_total": len(txs),
                        "tx_candidates": len(tx_candidates),
                        "rusd_ripplestate_delta_rows": total_rusd_state_rows,
                        "xrp_accountroot_delta_rows": total_xrp_root_rows,
                        "offer_fill_rows_rusd_xrp": total_offer_fill_rows,
                        "tx_with_amm_ledgerentry": total_amm_ledgerentry_txs,
                    },
                    "tx_candidates": tx_candidates,
                }

                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                n_out += 1

            except Exception as e:
                n_err += 1
                fout.write(json.dumps({
                    "error": True,
                    "error_msg": str(e),
                    "line_no": n_in,
                }, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"Input ledgers: {n_in}")
    print(f"Output lines : {n_out} (ok)")
    print(f"Errors       : {n_err}")
    print(f"NDJSON written to: {args.out_ndjson}")


if __name__ == "__main__":
    main()