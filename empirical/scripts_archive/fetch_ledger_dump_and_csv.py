#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests


def rpc_call(rpc_url: str, method: str, params: List[Dict[str, Any]], timeout: int = 60) -> Dict[str, Any]:
    """
    JSON-RPC call wrapper.
    Returns the 'result' object (not the envelope).
    Raises RuntimeError for non-success responses.
    """
    payload = {"method": method, "params": params}
    r = requests.post(rpc_url, json=payload, timeout=timeout)
    r.raise_for_status()
    out = r.json()

    if "result" not in out:
        raise RuntimeError(f"Bad RPC response (no 'result'): {out}")

    result = out["result"]
    # rippled sometimes omits "status"; if present and not success => treat as error
    if result.get("status") not in (None, "success"):
        raise RuntimeError(f"RPC status not success: {result}")

    return result


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpc", required=True, help="rippled JSON-RPC URL, e.g. http://127.0.0.1:5005")
    ap.add_argument("--ledger-min", type=int, required=True, help="start ledger_index (inclusive)")
    ap.add_argument("--ledger-max", type=int, required=True, help="end ledger_index (inclusive)")
    ap.add_argument("--out-ndjson", required=True, help="output NDJSON path, e.g. data/output/ledgers_1000_2000.ndjson")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--sleep-ms", type=int, default=0, help="optional sleep between requests (ms)")
    args = ap.parse_args()

    if args.ledger_max < args.ledger_min:
        raise SystemExit("--ledger-max must be >= --ledger-min")

    out_dir = os.path.dirname(os.path.abspath(args.out_ndjson))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sleep_s = max(args.sleep_ms, 0) / 1000.0

    total = args.ledger_max - args.ledger_min + 1
    ok_cnt = 0
    err_cnt = 0

    with open(args.out_ndjson, "w", encoding="utf-8") as f:
        for i, ledger_idx in enumerate(range(args.ledger_min, args.ledger_max + 1), start=1):
            record: Dict[str, Any] = {
                "ok": False,
                "ledger_index": ledger_idx,
                "fetched_at_utc": utc_now_iso(),
                "result": None,
                "error": None,
            }

            try:
                res = rpc_call(
                    args.rpc,
                    "ledger",
                    [{
                        "ledger_index": ledger_idx,
                        "transactions": True,
                        "expand": True,
                        "owner_funds": True,
                        "binary": False,
                    }],
                    timeout=args.timeout,
                )
                record["ok"] = True
                record["result"] = res
                ok_cnt += 1
            except Exception as e:
                # keep going; store error string
                record["error"] = str(e)
                err_cnt += 1

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # lightweight progress log to stdout
            if i == 1 or i == total or i % 50 == 0:
                print(f"[{i}/{total}] ledger={ledger_idx} ok={record['ok']} (ok={ok_cnt}, err={err_cnt})")

            if sleep_s > 0:
                time.sleep(sleep_s)

    print("\nDone.")
    print(f"NDJSON written to: {args.out_ndjson}")
    print(f"Ledgers: {total}  ok: {ok_cnt}  err: {err_cnt}")


if __name__ == "__main__":
    main()