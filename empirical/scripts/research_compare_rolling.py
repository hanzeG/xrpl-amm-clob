# compare_rolling.py
# Rolling-window model simulation: rUSD-XRP over a ledger window.
#
# Rolling differences:
# - AMM pool reserves are rolled forward using MODEL execution only.
# - Per tx, AMM pre-state comes from rolling reserves; first time init from real before reserves in amm_swaps row.
# - CLOB book snapshots use same logic: prefer (L-1), else fallback to latest < L.
# - target_out is still REAL total out (AMM out + CLOB legs out) so we have exact-out target.
#
# Output:
# - prints per-tx comparisons (optional, PRINT_PER_TX=0 to disable)
# - writes parquet: {ROOT}/compare_results_rolling.parquet
# - prints compact rolling metrics:
#   * direction-level cumulative drift (sum model_in - real_in)
#   * avg_px drift
#   * perfect-path rate (presence signature: AMM-only vs AMM+CLOB)
#   * amount buckets (simplified)
#   * drift trend samples (cum diff_in over time)
#   * rolling reserve warnings count

import json
import os
import glob
import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, getcontext
from typing import Any, Dict, List, Optional, Tuple

from pyspark.sql import SparkSession, functions as F, types as T

from xrpl_router.amm import AMM
from xrpl_router.book_step import RouterQuoteView
from xrpl_router.core import Quality, IOUAmount, XRPAmount
from xrpl_router.core.datatypes import Segment
from xrpl_router.core.constants import XRP_QUANTUM, IOU_QUANTUM
from xrpl_router.core.fmt import amount_to_decimal, quantize_down

getcontext().prec = 40

# -----------------------------
# Window config (edit if needed)
# -----------------------------
ROOT = "rlusd_xrp_100891230_100913230_1d"
LEDGER_MIN = 100891230
LEDGER_MAX = 100913230

AMM_SWAPS_PARQUET = f"{ROOT}/amm_rusd_xrp_100891230_100913230"
AMM_FEES_PARQUET  = f"{ROOT}/amm_fees_100891230_100913230"

CLOB_LEGS_PARQUET = (
    f"{ROOT}/clob_rusd_xrp_DATE_2025-12-15_to_2025-12-16_"
    f"TIME_20251215_110800Z_to_20251216_110310Z_with_idx"
)

BOOK_GETS_XRP_NDJSON  = f"{ROOT}/book_rusd_xrp_getsXRP.ndjson"
BOOK_GETS_RUSD_NDJSON = f"{ROOT}/book_rusd_xrp_getsrUSD.ndjson"
OUT_PARQUET = f"{ROOT}/compare_results_rolling.parquet"

# currencies
RUSD_HEX = "524C555344000000000000000000000000000000"
XRP = "XRP"

PRINT_DP: Optional[int] = 6

# Debug toggles
PRINT_PER_TX = bool(int(os.environ.get("PRINT_PER_TX", "1")))  # 0 to reduce spam
DEBUG_DIR_RUSD_TO_XRP = bool(int(os.environ.get("DEBUG_DIR_RUSD_TO_XRP", "1")))

# Optional issuer transfer rate for IOU inputs (e.g. rUSD transfer fee).
IOU_IN_TRANSFER_RATE = Decimal(os.environ.get("IOU_IN_TRANSFER_RATE", "1"))

# Rolling drift trend sampling cadence (every N ok tx per direction)
DRIFT_SAMPLE_EVERY = int(os.environ.get("DRIFT_SAMPLE_EVERY", "10"))


def _pick_one(pattern: str, what: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {what} matched pattern: {pattern}")
    return matches[0]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Research rolling comparison over a ledger window.")
    p.add_argument("--root", default=ROOT, help="Dataset root folder")
    p.add_argument("--pair", default="rlusd_xrp", help="Pair key used in output naming")
    p.add_argument("--ledger-start", "--ledger-min", dest="ledger_start", type=int, default=LEDGER_MIN, help="Ledger lower bound (inclusive)")
    p.add_argument("--ledger-end", "--ledger-max", dest="ledger_end", type=int, default=LEDGER_MAX, help="Ledger upper bound (inclusive)")
    p.add_argument("--amm-swaps", default=None, help="AMM swaps parquet path (optional)")
    p.add_argument("--amm-fees", default=None, help="AMM fees parquet path (optional)")
    p.add_argument("--clob-legs", default=None, help="CLOB legs parquet path (optional)")
    p.add_argument("--book-gets-xrp", default=None, help="Book snapshot getsXRP ndjson path (optional)")
    p.add_argument("--book-gets-rusd", default=None, help="Book snapshot getsrUSD ndjson path (optional)")
    p.add_argument("--output-dir", default=None, help="Output directory")
    p.add_argument("--output-name", default=None, help="Output parquet filename")
    return p.parse_args()


def _configure_from_args(args: argparse.Namespace) -> None:
    global ROOT, LEDGER_MIN, LEDGER_MAX
    global AMM_SWAPS_PARQUET, AMM_FEES_PARQUET, CLOB_LEGS_PARQUET
    global BOOK_GETS_XRP_NDJSON, BOOK_GETS_RUSD_NDJSON, OUT_PARQUET

    ROOT = args.root
    LEDGER_MIN = int(args.ledger_start)
    LEDGER_MAX = int(args.ledger_end)

    AMM_SWAPS_PARQUET = args.amm_swaps or _pick_one(f"{ROOT}/amm_rusd_xrp_*", "AMM swaps parquet")
    AMM_FEES_PARQUET = args.amm_fees or _pick_one(f"{ROOT}/amm_fees_*", "AMM fees parquet")
    CLOB_LEGS_PARQUET = args.clob_legs or _pick_one(f"{ROOT}/clob_rusd_xrp*_with_idx", "CLOB legs parquet")
    BOOK_GETS_XRP_NDJSON = args.book_gets_xrp or f"{ROOT}/book_rusd_xrp_getsXRP.ndjson"
    BOOK_GETS_RUSD_NDJSON = args.book_gets_rusd or f"{ROOT}/book_rusd_xrp_getsrUSD.ndjson"
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join("artifacts", "compare", args.pair, f"ledger_{LEDGER_MIN}_{LEDGER_MAX}")
    os.makedirs(out_dir, exist_ok=True)
    output_name = args.output_name or f"compare_rolling_{args.pair}_ledger_{LEDGER_MIN}_{LEDGER_MAX}_v1.parquet"
    OUT_PARQUET = os.path.join(out_dir, output_name)


def _write_manifest(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# -----------------------------
# Utility formatting
# -----------------------------
def fmt(v: Any, dp: Optional[int] = None) -> str:
    if v is None:
        return "None"
    if dp is None:
        dp = PRINT_DP
    try:
        d = v if isinstance(v, Decimal) else Decimal(str(v))
    except Exception:
        return str(v)
    if dp is None:
        return str(d)
    q = Decimal(1).scaleb(-int(dp))
    try:
        return str(d.quantize(q))
    except Exception:
        return str(d)


def assert_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")


def D(x: Any) -> Decimal:
    if x is None:
        return Decimal("0")
    return Decimal(str(x))


def pretty_cur(c: str) -> str:
    return "rUSD" if c == RUSD_HEX else c


def is_xrp(cur: str) -> bool:
    return cur == XRP


def direction_label(in_cur: str, out_cur: str) -> str:
    return f"{pretty_cur(in_cur)}->{pretty_cur(out_cur)}"


# -----------------------------
# XRPL amount parsing
# -----------------------------
def parse_amount(a: Any) -> Tuple[str, Optional[str], Decimal]:
    if isinstance(a, str):
        return (XRP, None, Decimal(a) * XRP_QUANTUM)  # drops -> XRP
    if isinstance(a, dict):
        return (a.get("currency"), a.get("issuer"), Decimal(str(a.get("value"))))
    raise ValueError(f"Unrecognised amount: {a!r}")


def amt_floor(cur: str, v: Decimal):
    if is_xrp(cur):
        drops = int((v / XRP_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
        return XRPAmount(value=drops)
    units = int((v / IOU_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
    return IOUAmount.from_components(units, -15)


# -----------------------------
# Book snapshot loading
# -----------------------------
def load_book_ndjson(path: str) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            li = obj.get("ledger_index", None)
            if li is None:
                li = obj.get("ledger", None) or obj.get("ledger_current_index", None)
            if li is None:
                raise ValueError(f"NDJSON line {line_no} missing ledger_index: {path}")
            li = int(li)

            offers = obj.get("offers", None)
            if offers is None:
                offers = obj.get("result", {}).get("offers", [])
            if offers is None:
                offers = []
            out[li] = offers
    return out


# -----------------------------
# Pre-book selection with fallback
# -----------------------------
def build_sorted_keys(m: Dict[int, Any]) -> List[int]:
    return sorted(m.keys())


def latest_snapshot_before(keys: List[int], ledger_index: int) -> Optional[int]:
    lo, hi = 0, len(keys) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        k = keys[mid]
        if k < ledger_index:
            best = k
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def pick_prebook(
    ledger_index: int,
    book_xrp_map: Dict[int, List[Dict[str, Any]]],
    book_rusd_map: Dict[int, List[Dict[str, Any]]],
    xrp_keys: List[int],
    rusd_keys: List[int],
) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    want = ledger_index - 1
    offers_xrp = book_xrp_map.get(want, None)
    offers_rusd = book_rusd_map.get(want, None)

    if offers_xrp is not None or offers_rusd is not None:
        return want, (offers_xrp or []), (offers_rusd or [])

    fx = latest_snapshot_before(xrp_keys, ledger_index)
    fr = latest_snapshot_before(rusd_keys, ledger_index)

    if fx is None and fr is None:
        raise RuntimeError(f"No book snapshots exist before ledger {ledger_index}. Cannot compare.")

    if fx is None:
        used = fr
    elif fr is None:
        used = fx
    else:
        used = max(fx, fr)

    return used, (book_xrp_map.get(used, []) or []), (book_rusd_map.get(used, []) or [])


# -----------------------------
# Build tiers from XRPL offers
# -----------------------------
def quantize_up(v: Decimal, quantum: Decimal) -> Decimal:
    if quantum == 0:
        return v
    n = (v / quantum).to_integral_value(rounding=ROUND_CEILING)
    return n * quantum


def apply_iou_in_transfer_fee(in_cur: str, v: Decimal) -> Decimal:
    if is_xrp(in_cur):
        return v
    if IOU_IN_TRANSFER_RATE is None:
        return v
    if IOU_IN_TRANSFER_RATE <= 0:
        return v
    return v * IOU_IN_TRANSFER_RATE


def build_tiers_from_offers(
    offers: List[Dict[str, Any]],
    in_cur: str,
    out_cur: str,
) -> List[Tuple[Quality, Any, Any, Any]]:
    tiers: List[Tuple[Quality, Any, Any, Any]] = []

    for o in offers:
        gets_cur, _, gets_val = parse_amount(o.get("TakerGets"))
        pays_cur, _, pays_val = parse_amount(o.get("TakerPays"))

        if "taker_gets_funded" in o:
            try:
                fg_cur, _, fg_val = parse_amount(o.get("taker_gets_funded"))
                if fg_cur == gets_cur and fg_val is not None:
                    gets_val = fg_val
            except Exception:
                pass
        if "taker_pays_funded" in o:
            try:
                fp_cur, _, fp_val = parse_amount(o.get("taker_pays_funded"))
                if fp_cur == pays_cur and fp_val is not None:
                    pays_val = fp_val
            except Exception:
                pass

        try:
            if "owner_funds" in o and D(o.get("owner_funds")) <= 0:
                continue
        except Exception:
            pass

        xrpl_q_raw = o.get("quality")

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

        out_q = XRP_QUANTUM if is_xrp(out_cur) else IOU_QUANTUM
        in_q  = XRP_QUANTUM if is_xrp(in_cur)  else IOU_QUANTUM

        out_max_dec = quantize_down(out_max_dec, out_q)

        in_need_dec = apply_iou_in_transfer_fee(in_cur, in_need_dec)
        in_need_dec = quantize_up(in_need_dec, in_q)

        out_amt = amt_floor(out_cur, out_max_dec)
        in_amt  = amt_floor(in_cur,  in_need_dec)

        if amount_to_decimal(out_amt) <= 0 or amount_to_decimal(in_amt) <= 0:
            continue

        q = Quality.from_amounts(out_amt, in_amt)
        tiers.append((q, out_amt, in_amt, xrpl_q_raw))

    if not tiers:
        raise RuntimeError("No direction-aligned offers found for this IN/OUT.")

    tiers.sort(key=lambda t: t[0], reverse=True)
    return tiers


# -----------------------------
# RouterQuoteView helpers
# -----------------------------
def tier_price_in_per_out(in_cur: str, out_cur: str, out_amt: Any, in_amt: Any) -> Optional[Decimal]:
    """Price as input-units per 1 output-unit, using Decimal converted amounts."""
    try:
        out_d = amount_to_decimal(out_amt)
        in_d = amount_to_decimal(in_amt)
        if out_d == 0:
            return None
        return in_d / out_d
    except Exception:
        return None


def tiers_to_segments(
    tiers: List[Tuple[Quality, Any, Any, Any]],
    src: str = "CLOB",
) -> List[Segment]:
    segs: List[Segment] = []
    for i, (q, out_amt, in_amt, xrpl_q_raw) in enumerate(tiers):
        segs.append(
            Segment(
                src=src,
                quality=q,
                out_max=out_amt,
                in_at_out_max=in_amt,
                raw_quality=q,
                source_id=f"tier:{i}",
            )
        )
    return segs


def run_best_quote(
    amm: AMM,
    tiers: List[Tuple[Quality, Any, Any, Any]],
    target_out_amt: Any,
) -> Tuple[Dict[str, Any], List[Any]]:
    segs = tiers_to_segments(tiers, src="CLOB")
    view = RouterQuoteView(lambda: segs, amm=amm)
    q = view.preview_out(target_out_amt)
    return q, segs


# -----------------------------
# Slice parsing
# -----------------------------
def summarise_model_slices(quote: Dict[str, Any]) -> Tuple[Decimal, Decimal, List[str]]:
    """
    Returns:
      (amm_in_total, amm_out_total, path_steps_raw)
    """
    amm_in = Decimal("0")
    amm_out = Decimal("0")
    path: List[str] = []

    slices = quote.get("slices", []) or []
    for s in slices:
        if not isinstance(s, dict):
            continue
        src = str(s.get("src") or s.get("kind") or "?")
        path.append(src)

        in_take = amount_to_decimal(s.get("in_take")) if s.get("in_take") is not None else Decimal("0")
        out_take = amount_to_decimal(s.get("out_take")) if s.get("out_take") is not None else Decimal("0")

        if "AMM" in src.upper():
            amm_in += in_take
            amm_out += out_take

    return amm_in, amm_out, path


# -----------------------------
# Path normalisation + unified signature
# -----------------------------
def normalise_src(src: str) -> str:
    s = (src or "").upper()
    if "AMM" in s:
        return "AMM"
    if "CLOB" in s or "BOOK" in s or "ORDER" in s:
        return "CLOB"
    return "?"


def collapse_steps(path_raw: List[str]) -> List[str]:
    """Normalise to AMM/CLOB and merge consecutive duplicates; drop unknowns."""
    out: List[str] = []
    prev: Optional[str] = None
    for p in path_raw:
        n = normalise_src(p)
        if n == "?":
            continue
        if prev is None or n != prev:
            out.append(n)
            prev = n
    return out


def presence_signature(uses_amm: bool, uses_clob: bool) -> str:
    """Unified signature comparable across REAL and MODEL."""
    if uses_amm and uses_clob:
        return "AMM+CLOB"
    if uses_amm:
        return "AMM"
    if uses_clob:
        return "CLOB"
    return "NONE"


def real_presence_signature(n_legs: int) -> str:
    # From your dataset definition, REAL total_out always includes AMM_out and CLOB legs_out.
    # So REAL always uses AMM; CLOB is used iff legs>0.
    return presence_signature(uses_amm=True, uses_clob=(n_legs > 0))


def model_presence_signature(path_raw: List[str]) -> str:
    steps = collapse_steps(path_raw)
    uses_amm = any(x == "AMM" for x in steps)
    uses_clob = any(x == "CLOB" for x in steps)
    return presence_signature(uses_amm, uses_clob)


def model_switch_stats(path_raw: List[str]) -> Tuple[int, int, int, str]:
    """
    Returns:
      (switch_count, clob_seg_count, amm_seg_count, collapsed_path_str)
    switch_count is number of transitions between AMM<->CLOB after collapsing.
    """
    steps = collapse_steps(path_raw)
    if not steps:
        return 0, 0, 0, ""
    clob_segs = sum(1 for x in steps if x == "CLOB")
    amm_segs = sum(1 for x in steps if x == "AMM")
    switches = max(0, len(steps) - 1)
    return switches, clob_segs, amm_segs, " + ".join(steps)


# -----------------------------
# Rolling AMM state
# -----------------------------
class PoolState:
    """
    Maintains rolling reserves for a pool in symmetric asset ordering.
    state.x is reserve for asset_a, state.y is reserve for asset_b.
    """
    def __init__(self, x_reserve: Decimal, y_reserve: Decimal):
        self.x = x_reserve
        self.y = y_reserve


def pool_key(amm_account: str, asset1: str, asset2: str) -> str:
    a, b = sorted([asset1, asset2])
    return f"{amm_account}|{a}|{b}"


def orient_reserves(
    state: PoolState,
    in_cur: str,
    out_cur: str,
    pool_asset_a: str,
    pool_asset_b: str,
) -> Tuple[Decimal, Decimal, bool]:
    if in_cur == pool_asset_a and out_cur == pool_asset_b:
        return state.x, state.y, False
    if in_cur == pool_asset_b and out_cur == pool_asset_a:
        return state.y, state.x, True
    raise RuntimeError("orient_reserves: currencies do not match pool assets")


def write_back_reserves(
    state: PoolState,
    new_x: Decimal,
    new_y: Decimal,
    flipped: bool,
):
    if not flipped:
        state.x = new_x
        state.y = new_y
    else:
        state.x = new_y
        state.y = new_x


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = _parse_args()
    _configure_from_args(args)
    for p in [AMM_SWAPS_PARQUET, AMM_FEES_PARQUET, CLOB_LEGS_PARQUET, BOOK_GETS_XRP_NDJSON, BOOK_GETS_RUSD_NDJSON]:
        assert_exists(p)

    print(f"\n=== Rolling compare: ledgers [{LEDGER_MIN}, {LEDGER_MAX}] rUSD-XRP ===")
    print(f"IOU_IN_TRANSFER_RATE  : {IOU_IN_TRANSFER_RATE}")
    print(f"PRINT_PER_TX          : {int(PRINT_PER_TX)}")
    print(f"DRIFT_SAMPLE_EVERY    : {DRIFT_SAMPLE_EVERY}")

    # Load book snapshots
    book_gets_xrp_map  = load_book_ndjson(BOOK_GETS_XRP_NDJSON)
    book_gets_rusd_map = load_book_ndjson(BOOK_GETS_RUSD_NDJSON)
    xrp_keys = build_sorted_keys(book_gets_xrp_map)
    rusd_keys = build_sorted_keys(book_gets_rusd_map)

    print(f"[book] getsXRP snapshots: {len(xrp_keys)} | range [{xrp_keys[0]}, {xrp_keys[-1]}]")
    print(f"[book] getsrUSD snapshots: {len(rusd_keys)} | range [{rusd_keys[0]}, {rusd_keys[-1]}]")

    spark = SparkSession.builder.appName("compare_rusd_xrp_rolling_window").getOrCreate()

    # Read tx list
    amm_swaps = (
        spark.read.parquet(AMM_SWAPS_PARQUET)
        .filter((F.col("ledger_index") >= F.lit(LEDGER_MIN)) & (F.col("ledger_index") <= F.lit(LEDGER_MAX)))
        .filter(
            (
                (F.col("amm_asset_currency") == F.lit(RUSD_HEX)) & (F.col("amm_asset2_currency") == F.lit(XRP))
            ) | (
                (F.col("amm_asset_currency") == F.lit(XRP)) & (F.col("amm_asset2_currency") == F.lit(RUSD_HEX))
            )
        )
        .select(
            "transaction_hash",
            "ledger_index",
            "transaction_index",
            "amm_account",
            "asset_in_currency",
            "asset_out_currency",
            "asset_in_value",
            "asset_out_value",
            "amm_asset_currency",
            "amm_asset2_currency",
            "amm_asset_balance_before",
            "amm_asset2_balance_before",
        )
    )

    amm_fees = (
        spark.read.parquet(AMM_FEES_PARQUET)
        .filter((F.col("ledger_index") >= F.lit(LEDGER_MIN)) & (F.col("ledger_index") <= F.lit(LEDGER_MAX)))
        .select("transaction_hash", "ledger_index", "amm_account", "trading_fee")
    )

    clob_legs = (
        spark.read.parquet(CLOB_LEGS_PARQUET)
        .filter((F.col("ledger_index") >= F.lit(LEDGER_MIN)) & (F.col("ledger_index") <= F.lit(LEDGER_MAX)))
        .select(
            F.col("tx_hash").alias("transaction_hash"),
            "ledger_index",
            "transaction_index",
            "base_currency",
            "counter_currency",
            "base_amount",
            "counter_amount",
        )
    )

    swaps = amm_swaps.orderBy(F.col("ledger_index").asc(), F.col("transaction_index").asc()).collect()
    if not swaps:
        raise RuntimeError(f"No rUSD-XRP swaps found in window [{LEDGER_MIN},{LEDGER_MAX}]")
    print(f"\nTx count (amm_swaps rows): {len(swaps)}")

    # Fee map
    fee_map: Dict[Tuple[str, str], Decimal] = {}
    for r in amm_fees.collect():
        h = r["transaction_hash"]
        a = r["amm_account"]
        f = D(r["trading_fee"])
        if h is not None and a is not None:
            fee_map[(str(h), str(a))] = f

    # CLOB legs map
    legs_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in clob_legs.collect():
        h = r["transaction_hash"]
        if h is None:
            continue
        legs_map.setdefault(str(h), []).append({
            "base_currency": r["base_currency"],
            "counter_currency": r["counter_currency"],
            "base_amount": r["base_amount"],
            "counter_amount": r["counter_amount"],
        })

    # Rolling pool states
    pool_states: Dict[str, Tuple[str, str, PoolState]] = {}

    # Results
    results: List[Dict[str, Any]] = []
    failed_txs: List[Tuple[int, int, str, str]] = []

    # Rolling warnings
    roll_negative_reserve_count = 0

    # Perfect path stats (presence signature)
    perfect_path_txs: List[Tuple[int, int, str, str, str]] = []
    perfect_path_by_dir: Dict[str, int] = {"XRP->rUSD": 0, "rUSD->XRP": 0}

    # Rolling drift trackers per direction
    drift_state = {
        "XRP->rUSD": {"cum_diff_in": Decimal("0"), "ok": 0, "samples": []},  # samples: [(ledger, tx, cum)]
        "rUSD->XRP": {"cum_diff_in": Decimal("0"), "ok": 0, "samples": []},
    }

    for tx in swaps:
        tx_hash = str(tx["transaction_hash"])
        ledger_index = int(tx["ledger_index"])
        tx_index = int(tx["transaction_index"]) if tx["transaction_index"] is not None else -1

        in_cur = tx["asset_in_currency"]
        out_cur = tx["asset_out_currency"]
        dir_lbl = direction_label(in_cur, out_cur)

        dir_key = None
        if in_cur == XRP and out_cur == RUSD_HEX:
            dir_key = "XRP->rUSD"
        elif in_cur == RUSD_HEX and out_cur == XRP:
            dir_key = "rUSD->XRP"

        # REAL components (used only as target_out + comparison)
        amm_in_real  = D(tx["asset_in_value"])
        amm_out_real = D(tx["asset_out_value"])

        legs = legs_map.get(tx_hash, [])
        real_clob_legs = len(legs)

        clob_in_sum = Decimal("0")
        clob_out_sum = Decimal("0")
        for r in legs:
            base_cur = r["base_currency"]
            counter_cur = r["counter_currency"]
            base_amt = D(r["base_amount"])
            counter_amt = D(r["counter_amount"])

            if base_cur == out_cur and counter_cur == in_cur:
                clob_out_sum += base_amt
                clob_in_sum  += counter_amt
            elif counter_cur == out_cur and base_cur == in_cur:
                clob_out_sum += counter_amt
                clob_in_sum  += base_amt

        real_total_out = amm_out_real + clob_out_sum
        real_total_in  = amm_in_real  + clob_in_sum

        # Pool id
        amm_account = str(tx["amm_account"])
        pool_asset1 = str(tx["amm_asset_currency"])
        pool_asset2 = str(tx["amm_asset2_currency"])
        pkey = pool_key(amm_account, pool_asset1, pool_asset2)

        # Init rolling state first time
        if pkey not in pool_states:
            side1_pre = D(tx["amm_asset_balance_before"])
            side2_pre = D(tx["amm_asset2_balance_before"])
            asset_a, asset_b = sorted([pool_asset1, pool_asset2])
            if asset_a == pool_asset1 and asset_b == pool_asset2:
                st = PoolState(side1_pre, side2_pre)
            else:
                st = PoolState(side2_pre, side1_pre)
            pool_states[pkey] = (asset_a, asset_b, st)

        asset_a, asset_b, st = pool_states[pkey]

        # Oriented reserves
        try:
            x_reserve, y_reserve, flipped = orient_reserves(st, in_cur, out_cur, asset_a, asset_b)
        except Exception as e:
            err = f"pool orient failed: {type(e).__name__}: {e}"
            failed_txs.append((ledger_index, tx_index, tx_hash, err))
            results.append({
                "ledger_index": ledger_index,
                "transaction_index": tx_index,
                "transaction_hash": tx_hash,
                "direction": dir_lbl,
                "used_prebook_ledger": None,
                "real_in": float(real_total_in),
                "real_out": float(real_total_out),
                "model_in": None,
                "model_out": None,
                "diff_in": None,
                "diff_out": None,
                "error": err,
                "rolling_x_pre": float(x_reserve) if x_reserve is not None else None,
                "rolling_y_pre": float(y_reserve) if y_reserve is not None else None,
                "rolling_x_post": None,
                "rolling_y_post": None,
                "model_amm_in": None,
                "model_amm_out": None,
                "path_sig_real": None,
                "path_sig_model": None,
                "real_clob_legs": int(real_clob_legs),
                "model_clob_segs": None,
                "model_switches": None,
            })
            continue

        trading_fee = fee_map.get((tx_hash, amm_account), Decimal("0"))
        amm = AMM(x_reserve, y_reserve, trading_fee, x_is_xrp=is_xrp(in_cur), y_is_xrp=is_xrp(out_cur))

        # Pick book snapshot
        try:
            used_prebook_ledger, offers_xrp, offers_rusd = pick_prebook(
                ledger_index=ledger_index,
                book_xrp_map=book_gets_xrp_map,
                book_rusd_map=book_gets_rusd_map,
                xrp_keys=xrp_keys,
                rusd_keys=rusd_keys,
            )
        except Exception as e:
            err = f"pick_prebook failed: {type(e).__name__}: {e}"
            failed_txs.append((ledger_index, tx_index, tx_hash, err))
            results.append({
                "ledger_index": ledger_index,
                "transaction_index": tx_index,
                "transaction_hash": tx_hash,
                "direction": dir_lbl,
                "used_prebook_ledger": None,
                "real_in": float(real_total_in),
                "real_out": float(real_total_out),
                "model_in": None,
                "model_out": None,
                "diff_in": None,
                "diff_out": None,
                "error": err,
                "rolling_x_pre": float(x_reserve),
                "rolling_y_pre": float(y_reserve),
                "rolling_x_post": None,
                "rolling_y_post": None,
                "model_amm_in": None,
                "model_amm_out": None,
                "path_sig_real": None,
                "path_sig_model": None,
                "real_clob_legs": int(real_clob_legs),
                "model_clob_segs": None,
                "model_switches": None,
            })
            continue

        # Direction-locked: choose the correct book for the swap direction.
        # getsrUSD book: taker_gets=rUSD, taker_pays=XRP  -> XRP->rUSD
        # getsXRP  book: taker_gets=XRP,  taker_pays=rUSD -> rUSD->XRP
        tiers: Optional[List[Tuple[Quality, Any, Any, Any]]] = None
        tiers_src: Optional[str] = None
        last_err: Optional[Exception] = None

        if in_cur == XRP and out_cur == RUSD_HEX:
            tiers_src = "getsrUSD"
            book = offers_rusd
        elif in_cur == RUSD_HEX and out_cur == XRP:
            tiers_src = "getsXRP"
            book = offers_xrp
        else:
            book = []

        if not book:
            err = f"build_tiers failed: empty book for direction {pretty_cur(in_cur)}->{pretty_cur(out_cur)} (src={tiers_src})"
            failed_txs.append((ledger_index, tx_index, tx_hash, err))
            results.append({
                "ledger_index": ledger_index,
                "transaction_index": tx_index,
                "transaction_hash": tx_hash,
                "direction": dir_lbl,
                "used_prebook_ledger": used_prebook_ledger,
                "real_in": float(real_total_in),
                "real_out": float(real_total_out),
                "model_in": None,
                "model_out": None,
                "diff_in": None,
                "diff_out": None,
                "error": err,
                "rolling_x_pre": float(x_reserve),
                "rolling_y_pre": float(y_reserve),
                "rolling_x_post": None,
                "rolling_y_post": None,
                "model_amm_in": None,
                "model_amm_out": None,
                "path_sig_real": real_presence_signature(real_clob_legs),
                "path_sig_model": None,
                "real_clob_legs": int(real_clob_legs),
                "model_clob_segs": None,
                "model_switches": None,
            })
            continue

        try:
            tiers = build_tiers_from_offers(book, in_cur, out_cur)
        except Exception as e:
            last_err = e
            tiers = None

        if tiers is None:
            err = f"build_tiers failed: {last_err} (src={tiers_src})"
            failed_txs.append((ledger_index, tx_index, tx_hash, err))
            results.append({
                "ledger_index": ledger_index,
                "transaction_index": tx_index,
                "transaction_hash": tx_hash,
                "direction": dir_lbl,
                "used_prebook_ledger": used_prebook_ledger,
                "real_in": float(real_total_in),
                "real_out": float(real_total_out),
                "model_in": None,
                "model_out": None,
                "diff_in": None,
                "diff_out": None,
                "error": err,
                "rolling_x_pre": float(x_reserve),
                "rolling_y_pre": float(y_reserve),
                "rolling_x_post": None,
                "rolling_y_post": None,
                "model_amm_in": None,
                "model_amm_out": None,
                "path_sig_real": real_presence_signature(real_clob_legs),
                "path_sig_model": None,
                "real_clob_legs": int(real_clob_legs),
                "model_clob_segs": None,
                "model_switches": None,
            })
            continue

        target_out_amt = amt_floor(out_cur, real_total_out)

        pred_in = None
        pred_out = None
        path_m_raw: List[str] = []
        model_amm_in = Decimal("0")
        model_amm_out = Decimal("0")
        model_err: Optional[str] = None

        # model path stats (filled if quote ok)
        model_switches: Optional[int] = None
        model_clob_segs: Optional[int] = None
        model_amm_segs: Optional[int] = None
        model_path_collapsed_str: str = ""

        try:
            quote, _ = run_best_quote(amm=amm, tiers=tiers, target_out_amt=target_out_amt)
            summary = quote.get("summary", {})
            if summary.get("is_partial"):
                raise RuntimeError("Model returned PARTIAL quote")

            pred_in  = amount_to_decimal(summary["total_in"])
            pred_out = amount_to_decimal(summary["total_out"])

            model_amm_in, model_amm_out, path_m_raw = summarise_model_slices(quote)
            model_switches, model_clob_segs, model_amm_segs, model_path_collapsed_str = model_switch_stats(path_m_raw)

        except Exception as e:
            model_err = f"model_quote failed: {type(e).__name__}: {e}"
            failed_txs.append((ledger_index, tx_index, tx_hash, model_err))

        # Print per tx
        if PRINT_PER_TX:
            print("\n---")
            print(f"ledger {ledger_index} | tx {tx_index} | {tx_hash}")
            print(f"dir: {pretty_cur(in_cur)} -> {pretty_cur(out_cur)} | book snapshot ledger: {used_prebook_ledger}")
            print(f"ROLL pre: x_reserve(in)={fmt(x_reserve)} {pretty_cur(in_cur)} | y_reserve(out)={fmt(y_reserve)} {pretty_cur(out_cur)}")

            # Unified REAL path: presence only + clob legs count as separate stat
            real_sig = real_presence_signature(real_clob_legs)
            print(f"REAL  path_sig(presence): {real_sig} | clob_legs={real_clob_legs}")
            print(f"REAL  TOT : in={fmt(real_total_in)} {pretty_cur(in_cur)}  out={fmt(real_total_out)} {pretty_cur(out_cur)}")

            if pred_in is None or pred_out is None:
                print(f"MODEL: FAILED ({model_err})")
                if tiers is not None:
                    print(f"TIERS src : {tiers_src} | tiers={len(tiers)}")
            else:
                model_sig = model_presence_signature(path_m_raw)
                print(f"MODEL path_sig(presence): {model_sig} | clob_segs={model_clob_segs} | switches={model_switches}")
                if model_path_collapsed_str:
                    print(f"MODEL path(collapsed): {model_path_collapsed_str}")
                print(f"MODEL TOT : in={fmt(pred_in)} {pretty_cur(in_cur)}  out={fmt(pred_out)} {pretty_cur(out_cur)}")
                print(f"DIFF  in : {fmt(pred_in - real_total_in)} {pretty_cur(in_cur)}")
                print(f"DIFF  out: {fmt(pred_out - real_total_out)} {pretty_cur(out_cur)}")
                print(f"MODEL AMM : in={fmt(model_amm_in)} {pretty_cur(in_cur)}  out={fmt(model_amm_out)} {pretty_cur(out_cur)}")
                # Book tiers debug: which snapshot was used + top-3 tier prices
                if tiers is not None:
                    print(f"TIERS src : {tiers_src} | tiers={len(tiers)}")
                    for j, (qj, out_max_j, in_need_j, xrpl_q_raw_j) in enumerate(tiers[:3], start=1):
                        px = tier_price_in_per_out(in_cur, out_cur, out_max_j, in_need_j)
                        px_s = "None" if px is None else fmt(px, 12)
                        print(
                            f"  tier{j}: px(in/out)={px_s} | out_max={fmt(amount_to_decimal(out_max_j))} {pretty_cur(out_cur)}"
                            f" | in_at_out_max={fmt(amount_to_decimal(in_need_j))} {pretty_cur(in_cur)} | raw_quality={xrpl_q_raw_j}"
                        )

        # Fail row
        if pred_in is None or pred_out is None:
            results.append({
                "ledger_index": ledger_index,
                "transaction_index": tx_index,
                "transaction_hash": tx_hash,
                "direction": dir_lbl,
                "used_prebook_ledger": used_prebook_ledger,
                "real_in": float(real_total_in),
                "real_out": float(real_total_out),
                "model_in": None,
                "model_out": None,
                "diff_in": None,
                "diff_out": None,
                "error": model_err,
                "rolling_x_pre": float(x_reserve),
                "rolling_y_pre": float(y_reserve),
                "rolling_x_post": None,
                "rolling_y_post": None,
                "model_amm_in": float(model_amm_in) if model_amm_in is not None else None,
                "model_amm_out": float(model_amm_out) if model_amm_out is not None else None,
                "path_sig_real": real_presence_signature(real_clob_legs),
                "path_sig_model": model_presence_signature(path_m_raw),
                "real_clob_legs": int(real_clob_legs),
                "model_clob_segs": int(model_clob_segs) if model_clob_segs is not None else None,
                "model_switches": int(model_switches) if model_switches is not None else None,
            })
            continue

        # Rolling update (MODEL ONLY)
        new_x = x_reserve + model_amm_in
        new_y = y_reserve - model_amm_out

        roll_warn = None
        if new_x < 0 or new_y < 0:
            roll_negative_reserve_count += 1
            roll_warn = f"ROLL_NEGATIVE_RESERVE x={new_x} y={new_y}"

        write_back_reserves(st, new_x, new_y, flipped)

        diff_in = pred_in - real_total_in
        diff_out = pred_out - real_total_out

        # Perfect same execution path (presence signature only)
        if dir_key is not None:
            real_sig = real_presence_signature(real_clob_legs)
            model_sig = model_presence_signature(path_m_raw)
            if real_sig == model_sig:
                perfect_path_txs.append((ledger_index, tx_index, tx_hash, dir_key, real_sig))
                perfect_path_by_dir[dir_key] += 1

        # Drift sampling
        if dir_key is not None:
            st_d = drift_state[dir_key]
            st_d["ok"] += 1
            st_d["cum_diff_in"] += diff_in
            if (st_d["ok"] % DRIFT_SAMPLE_EVERY) == 0:
                st_d["samples"].append((ledger_index, tx_index, st_d["cum_diff_in"]))

        results.append({
            "ledger_index": ledger_index,
            "transaction_index": tx_index,
            "transaction_hash": tx_hash,
            "direction": dir_lbl,
            "used_prebook_ledger": used_prebook_ledger,
            "real_in": float(real_total_in),
            "real_out": float(real_total_out),
            "model_in": float(pred_in),
            "model_out": float(pred_out),
            "diff_in": float(diff_in),
            "diff_out": float(diff_out),
            "error": roll_warn,
            "rolling_x_pre": float(x_reserve),
            "rolling_y_pre": float(y_reserve),
            "rolling_x_post": float(new_x),
            "rolling_y_post": float(new_y),
            "model_amm_in": float(model_amm_in),
            "model_amm_out": float(model_amm_out),
            "path_sig_real": real_presence_signature(real_clob_legs),
            "path_sig_model": model_presence_signature(path_m_raw),
            "real_clob_legs": int(real_clob_legs),
            "model_clob_segs": int(model_clob_segs) if model_clob_segs is not None else None,
            "model_switches": int(model_switches) if model_switches is not None else None,
        })

    # -----------------------------
    # Write parquet
    # -----------------------------
    out_path = OUT_PARQUET
    schema = T.StructType([
        T.StructField("ledger_index", T.LongType(), False),
        T.StructField("transaction_index", T.LongType(), False),
        T.StructField("transaction_hash", T.StringType(), False),
        T.StructField("direction", T.StringType(), False),
        T.StructField("used_prebook_ledger", T.LongType(), True),

        T.StructField("real_in", T.DoubleType(), True),
        T.StructField("real_out", T.DoubleType(), True),
        T.StructField("model_in", T.DoubleType(), True),
        T.StructField("model_out", T.DoubleType(), True),
        T.StructField("diff_in", T.DoubleType(), True),
        T.StructField("diff_out", T.DoubleType(), True),

        T.StructField("error", T.StringType(), True),

        T.StructField("rolling_x_pre", T.DoubleType(), True),
        T.StructField("rolling_y_pre", T.DoubleType(), True),
        T.StructField("rolling_x_post", T.DoubleType(), True),
        T.StructField("rolling_y_post", T.DoubleType(), True),

        T.StructField("model_amm_in", T.DoubleType(), True),
        T.StructField("model_amm_out", T.DoubleType(), True),

        # Unified path signature (presence)
        T.StructField("path_sig_real", T.StringType(), True),
        T.StructField("path_sig_model", T.StringType(), True),

        # Extra comparable diagnostics
        T.StructField("real_clob_legs", T.LongType(), True),
        T.StructField("model_clob_segs", T.LongType(), True),
        T.StructField("model_switches", T.LongType(), True),
    ])

    df_out = spark.createDataFrame(results, schema=schema)
    df_out.coalesce(1).write.mode("overwrite").parquet(out_path)
    print(f"\n[OK] compare results written to: {out_path}")
    _write_manifest(
        os.path.join(os.path.dirname(out_path), "manifest.json"),
        {
            "script": "research_compare_rolling.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "pair": args.pair,
            "window": {"ledger_start": LEDGER_MIN, "ledger_end": LEDGER_MAX},
            "inputs": {
                "amm_swaps": AMM_SWAPS_PARQUET,
                "amm_fees": AMM_FEES_PARQUET,
                "clob_legs": CLOB_LEGS_PARQUET,
                "book_gets_xrp": BOOK_GETS_XRP_NDJSON,
                "book_gets_rusd": BOOK_GETS_RUSD_NDJSON,
            },
            "output_parquet": out_path,
            "tx_count": len(swaps),
            "failed_tx_count": len(failed_txs),
        },
    )

    # =========================================================
    # Compact rolling summaries
    # =========================================================
    def _d(x: Any) -> Decimal:
        return Decimal(str(x)) if x is not None else Decimal("0")

    def _rel_err(real_in: Decimal, model_in: Decimal) -> Optional[Decimal]:
        if real_in == 0:
            return None
        return (model_in - real_in) / real_in

    # Collect ok rows per direction
    dir_rows: Dict[str, List[Dict[str, Any]]] = {"XRP->rUSD": [], "rUSD->XRP": []}
    for r in results:
        if r.get("model_in") is None or r.get("model_out") is None:
            continue
        d = r.get("direction")
        if d == "XRP->rUSD":
            dir_rows["XRP->rUSD"].append(r)
        elif d == "rUSD->XRP":
            dir_rows["rUSD->XRP"].append(r)

    def print_dir_compact(dir_name: str, in_unit: str, out_unit: str) -> None:
        rows = dir_rows[dir_name]
        if not rows:
            print(f"\n=== {dir_name} ===")
            print("No ok txs.")
            return

        real_in_sum  = sum((_d(x["real_in"]) for x in rows), start=Decimal("0"))
        real_out_sum = sum((_d(x["real_out"]) for x in rows), start=Decimal("0"))
        model_in_sum = sum((_d(x["model_in"]) for x in rows), start=Decimal("0"))
        model_out_sum= sum((_d(x["model_out"]) for x in rows), start=Decimal("0"))

        cum_diff_in = model_in_sum - real_in_sum
        real_px = (real_in_sum / real_out_sum) if real_out_sum != 0 else None
        model_px = (model_in_sum / model_out_sum) if model_out_sum != 0 else None
        px_diff = (model_px - real_px) if (real_px is not None and model_px is not None) else None

        rel_errs: List[Decimal] = []
        for x in rows:
            e = _rel_err(_d(x["real_in"]), _d(x["model_in"]))
            if e is not None:
                rel_errs.append(e)
        mean_rel = (sum(rel_errs, start=Decimal("0")) / Decimal(len(rel_errs))) if rel_errs else None

        # perfect path rate (presence signature)
        perfect = perfect_path_by_dir.get(dir_name, 0)
        rate = (Decimal(perfect) / Decimal(len(rows))) if rows else Decimal("0")

        print(f"\n=== {dir_name} (ok tx={len(rows)}) ===")
        print(f"REAL  TOT : in={fmt(real_in_sum)} {in_unit}  out={fmt(real_out_sum)} {out_unit}")
        print(f"MODEL TOT : in={fmt(model_in_sum)} {in_unit}  out={fmt(model_out_sum)} {out_unit}")
        print(f"CUM_DIFF : in={fmt(cum_diff_in)} {in_unit}")
        if real_px is not None and model_px is not None:
            print(f"avg_px   : REAL={fmt(real_px)} {in_unit}/{out_unit} | MODEL={fmt(model_px)} | DIFF={fmt(px_diff)}")
        if mean_rel is not None:
            print(f"mean_rel_err_in = {fmt(mean_rel * Decimal('100'), 4)}%")
        print(f"perfect_path_rate(presence) = {perfect}/{len(rows)} ({fmt(rate * Decimal('100'), 3)}%)")

        # drift samples
        st_d = drift_state[dir_name]
        if st_d["samples"]:
            print(f"drift trend samples (every {DRIFT_SAMPLE_EVERY} ok tx):")
            for (li, ti, cum) in st_d["samples"][:10]:
                print(f"  ledger {li} tx {ti} | cum_diff_in={fmt(cum)} {in_unit}")
            if len(st_d["samples"]) > 10:
                print(f"  ... ({len(st_d['samples'])-10} more)")

    # Amount buckets (simplified)
    BUCKETS = [
        ("<=10", Decimal("10")),
        ("10-100", Decimal("100")),
        ("100-1k", Decimal("1000")),
        ("1k-10k", Decimal("10000")),
        (">10k", None),
    ]

    def bucket_label(v: Decimal) -> str:
        for name, hi in BUCKETS:
            if hi is None:
                return name
            if v <= hi:
                return name
        return BUCKETS[-1][0]

    def print_amount_buckets(dir_name: str, in_unit: str, out_unit: str) -> None:
        rows = dir_rows[dir_name]
        if not rows:
            return

        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for x in rows:
            outv = _d(x["real_out"])
            bl = bucket_label(outv)
            buckets.setdefault(bl, []).append(x)

        print(f"\n[{dir_name}] amount buckets by real_out ({out_unit}):")
        for name, _ in BUCKETS:
            br = buckets.get(name, [])
            if not br:
                continue

            real_out_sum = sum((_d(x["real_out"]) for x in br), start=Decimal("0"))
            real_in_sum  = sum((_d(x["real_in"]) for x in br), start=Decimal("0"))

            rel_errs = []
            for x in br:
                e = _rel_err(_d(x["real_in"]), _d(x["model_in"]))
                if e is not None:
                    rel_errs.append(e)
            mean_rel = (sum(rel_errs, start=Decimal("0")) / Decimal(len(rel_errs))) if rel_errs else None

            mean_rel_str = "None" if mean_rel is None else f"{fmt(mean_rel * Decimal('100'), 4)}%"
            print(
                f"  bucket {name:>7} | tx={len(br):>3} | out_sum={fmt(real_out_sum):>12} {out_unit}"
                f" | in_sum={fmt(real_in_sum):>12} {in_unit} | mean_rel_err={mean_rel_str}"
            )

    # Print compact summaries
    print_dir_compact("XRP->rUSD", "XRP", "rUSD")
    print_amount_buckets("XRP->rUSD", "XRP", "rUSD")

    print_dir_compact("rUSD->XRP", "rUSD", "XRP")
    print_amount_buckets("rUSD->XRP", "rUSD", "XRP")

    # Global perfect path
    total_ok = len(dir_rows["XRP->rUSD"]) + len(dir_rows["rUSD->XRP"])
    total_perfect = len(perfect_path_txs)
    ratio = (Decimal(total_perfect) / Decimal(total_ok)) if total_ok > 0 else Decimal("0")

    print("\n=== PATH + ROLLING WARNINGS ===")
    print(f"perfect_path total(presence) = {total_perfect}/{total_ok} ({fmt(ratio * Decimal('100'), 3)}%)")
    print(f"  XRP->rUSD : {perfect_path_by_dir['XRP->rUSD']}/{len(dir_rows['XRP->rUSD'])}")
    print(f"  rUSD->XRP : {perfect_path_by_dir['rUSD->XRP']}/{len(dir_rows['rUSD->XRP'])}")
    print(f"rolling negative reserve warnings = {roll_negative_reserve_count}")

    if failed_txs:
        print("\n=== FAILURES (first 10) ===")
        for (li, ti, h, e) in failed_txs[:10]:
            print(f"  ledger {li} tx {ti} hash={h}")
            print(f"    err: {e}")
        if len(failed_txs) > 10:
            print(f"  ... ({len(failed_txs)-10} more)")

    spark.stop()


if __name__ == "__main__":
    main()
