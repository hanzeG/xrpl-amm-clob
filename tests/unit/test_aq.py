import pytest
from decimal import Decimal
from xrpl_router.clob import make_ladder
from xrpl_router.amm import AMM
from xrpl_router.book_step import RouterQuoteView
from xrpl_router.core.fmt import amount_to_decimal, quality_price_to_decimal
from xrpl_router.core.exc import InsufficientLiquidityError

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Independent mode flags
IN_IS_XRP = False
OUT_IS_XRP = False
AMM_X0 = Decimal("1900")
AMM_Y0 = Decimal("2000")
AMM_FEE0 = Decimal("0.003")

# Quantisation constants
from xrpl_router.core.constants import XRP_QUANTUM, IOU_QUANTUM
from xrpl_router.core import XRPAmount, IOUAmount
from decimal import ROUND_FLOOR

def _amt_floor_out(d: Decimal):
    if OUT_IS_XRP:
        drops = int((d / XRP_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
        return XRPAmount(value=drops)
    else:
        units = int((d / IOU_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
        return IOUAmount.from_components(units, -15)

# Basic hybrid cost-metric test scaffold

def test_hybrid_cost_metrics_basic():
    ladder = make_ladder(
        depth=6,
        top_quality=Decimal("1.00"),
        qty_per_level=Decimal("40"),
        decay=Decimal("0.985"),
        in_is_xrp=IN_IS_XRP,
        out_is_xrp=OUT_IS_XRP,
    )
    amm = AMM(AMM_X0, AMM_Y0, AMM_FEE0, x_is_xrp=IN_IS_XRP, y_is_xrp=OUT_IS_XRP)

    # Scan cost curves for CLOB-only, AMM-only and HYBRID over increasing sizes
    step = Decimal("20")
    q = step

    rows = []

    while True:
        # HYBRID: current router logic with both ladder + AMM
        try:
            view_h = RouterQuoteView(lambda: ladder, amm=AMM(AMM_X0, AMM_Y0, AMM_FEE0, x_is_xrp=IN_IS_XRP, y_is_xrp=OUT_IS_XRP))
            quote_h = view_h.preview_out(_amt_floor_out(q))
        except InsufficientLiquidityError:
            # If even hybrid cannot fill, we stop the scan
            break

        sum_h = quote_h["summary"]
        # Stop when HYBRID can no longer fully fill the requested size
        if sum_h.get("is_partial"):
            break

        h_in = amount_to_decimal(sum_h["total_in"])
        h_out = amount_to_decimal(sum_h["total_out"])
        if h_out == 0:
            break
        h_avg = h_in / h_out

        # CLOB-only: same q, but only ladder
        c_in = None
        c_out = None
        c_avg = None
        try:
            view_c = RouterQuoteView(lambda: ladder, amm=None)
            quote_c = view_c.preview_out(_amt_floor_out(q))
            sum_c = quote_c["summary"]
            # If partial, treat as unavailable at this size
            if not sum_c.get("is_partial"):
                c_in = amount_to_decimal(sum_c["total_in"])
                c_out = amount_to_decimal(sum_c["total_out"])
                if c_out != 0:
                    c_avg = c_in / c_out
        except InsufficientLiquidityError:
            c_in = c_out = c_avg = None

        # AMM-only: same q, but only AMM
        a_in = None
        a_out = None
        a_avg = None
        try:
            view_a = RouterQuoteView(lambda: [], amm=AMM(AMM_X0, AMM_Y0, AMM_FEE0, x_is_xrp=IN_IS_XRP, y_is_xrp=OUT_IS_XRP))
            quote_a = view_a.preview_out(_amt_floor_out(q))
            sum_a = quote_a["summary"]
            if not sum_a.get("is_partial"):
                a_in = amount_to_decimal(sum_a["total_in"])
                a_out = amount_to_decimal(sum_a["total_out"])
                if a_out != 0:
                    a_avg = a_in / a_out
        except InsufficientLiquidityError:
            a_in = a_out = a_avg = None

        rows.append({
            "q": q,
            "clob_avg": c_avg,
            "amm_avg": a_avg,
            "hybrid_avg": h_avg,
        })

        print("Cost row")
        print(f"  q={q}")
        print(f"    clob_avg={c_avg if c_avg is not None else 'N/A'}")
        print(f"    amm_avg={a_avg if a_avg is not None else 'N/A'}")
        print(f"    hybrid_avg={h_avg}")

        q += step

    # Add an explicit q=0 point so that trade size starts from 0 on the plots
    if rows:
        first = rows[0]
        rows.insert(0, {
            "q": Decimal("0"),
            "clob_avg": first["clob_avg"],
            "amm_avg": first["amm_avg"],
            "hybrid_avg": first["hybrid_avg"],
        })

    print("\nCost curves summary")
    for r in rows:
        q = r["q"]
        print(f"  q={q} | clob_avg={r['clob_avg'] if r['clob_avg'] is not None else 'N/A'}, "
              f"amm_avg={r['amm_avg'] if r['amm_avg'] is not None else 'N/A'}, hybrid_avg={r['hybrid_avg']}")

    # Plot cost curves using seaborn: three modes on one figure
    if rows:
        records = []
        for r in rows:
            q_val = float(r["q"])  # Decimal -> float for plotting
            if r["clob_avg"] is not None:
                records.append({"q": q_val, "mode": "CLOB-Only", "avg_price": float(r["clob_avg"])})
            if r["amm_avg"] is not None:
                records.append({"q": q_val, "mode": "AMM-Only", "avg_price": float(r["amm_avg"])})
            # Router (hybrid) is always available within the scan range
            records.append({"q": q_val, "mode": "Router", "avg_price": float(r["hybrid_avg"])})

        df = pd.DataFrame(records)

        sns.set(style="whitegrid")
        # Globally increase font sizes slightly for better readability
        plt.rcParams.update({
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        })
        fig, ax = plt.subplots(figsize=(8, 5))

        # Main plot
        sns.lineplot(data=df, x="q", y="avg_price", hue="mode", ax=ax)
        ax.set_xlabel("Trade Size (q)")
        ax.set_ylabel("Average Execution Price")
        ax.set_title("Cost Curves: CLOB-Only vs AMM-Only vs Router")
        # Remove legend title, keep only entries
        ax.legend(loc="lower right")

        # Compute AMM depth (largest q where AMM-only is available)
        amm_depths = [float(r["q"]) for r in rows if r["amm_avg"] is not None]
        if amm_depths:
            amm_depth = max(amm_depths)
            if amm_depth != 0.0:
                ax.axvline(amm_depth, color="orange", linestyle="--", linewidth=1.2)
                # Minimal ticker label for AMM depth line (centre of the line, slightly to the right)
                y_min, y_max = ax.get_ylim()
                y_mid = 0.5 * (y_min + y_max)
                x_min, x_max = ax.get_xlim()
                depth_x_offset = 0.01 * (x_max - x_min)
                depth_label = f"AMM {amm_depth:.0f}"
                ax.text(amm_depth + depth_x_offset, y_mid, depth_label, rotation=270, va="center", ha="left",
                        fontsize=10, color="orange")

        # Compute hybrid total depth (largest scanned q)
        hybrid_depth = float(max(r["q"] for r in rows))
        if hybrid_depth != 0.0:
            ax.axvline(hybrid_depth, color="grey", linestyle="--", linewidth=1.2)
            # Minimal ticker label for Router (hybrid) depth line (centre of the line, slightly to the right)
            y_min, y_max = ax.get_ylim()
            y_mid = 0.5 * (y_min + y_max)
            x_min, x_max = ax.get_xlim()
            depth_x_offset = 0.01 * (x_max - x_min)
            depth_label = f"Router {hybrid_depth:.0f}"
            ax.text(hybrid_depth + depth_x_offset, y_mid, depth_label, rotation=270, va="center", ha="left",
                    fontsize=10, color="grey")

        # Determine CLOB-effective range for use in both main and inset plots
        clob_rows = [r for r in rows if r["clob_avg"] is not None]
        x0 = None
        x1 = None
        clob_index_by_q = {}
        if clob_rows:
            min_q = float(min(r["q"] for r in clob_rows))
            max_q = float(max(r["q"] for r in clob_rows))
            # Force inset x-range to include 0 so that the 0 tick is shown
            x0, x1 = 0.0, max_q + 0.5 * float(step)
            # Assign a compact ladder index per distinct CLOB size (excluding q=0)
            idx = 1
            for r in clob_rows:
                q_val = float(r["q"])
                if q_val == 0.0:
                    continue
                if q_val not in clob_index_by_q:
                    clob_index_by_q[q_val] = idx
                    idx += 1

        # Horizontal offset for placing CLOB marker tickers slightly to the right of the line
        x_min, x_max = ax.get_xlim()
        clob_x_offset = 0.01 * (x_max - x_min)

        # CLOB segment "depth" markers: finite-height vertical lines up to each CLOB avg price
        handles, labels = ax.get_legend_handles_labels()
        clob_color = None
        for h, lab in zip(handles, labels):
            if lab == "CLOB-Only":
                # Line2D from seaborn carries the colour used for CLOB curve
                if hasattr(h, "get_color"):
                    clob_color = h.get_color()
                break
        if clob_color is None:
            clob_color = "C0"  # fallback to first default colour

        for r in rows:
            if r["clob_avg"] is not None:
                x = float(r["q"])
                if x == 0.0:
                    continue
                y_top = float(r["clob_avg"])
                # Vertical segment from price axis baseline up to the CLOB avg price at that size
                ax.vlines(x, ymin=0.0, ymax=y_top, colors=clob_color, linestyles=":", linewidth=0.8, alpha=0.7)
                # Minimal ticker label for each CLOB segment marker:
                # if this x is inside the inset x-range, do not label on the main plot
                if x0 is not None and x1 is not None and x0 <= x <= x1:
                    continue
                idx = clob_index_by_q.get(x)
                if idx is None:
                    continue
                label = f"L{idx} {int(x)}"
                y_mid = y_top / 2.0
                ax.text(x + clob_x_offset, y_mid, label, rotation=270, va="center", ha="left",
                        fontsize=10, color=clob_color)

        # Inset zoom for the CLOB-effective range (where clob_avg is defined)
        if clob_rows and x0 is not None and x1 is not None:
            # Force inset x‑range to include 0 so that the 0 tick is shown
            # Subset dataframe to this x-range for y-limits
            sub_df = df[(df["q"] >= x0) & (df["q"] <= x1)]
            if not sub_df.empty:
                y_min = sub_df["avg_price"].min()
                y_max = sub_df["avg_price"].max()
                y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.01

                # Place inset in upper-left area to avoid legend (top-left anchored, expanded right/down)
                ax_ins = fig.add_axes([0.18, 0.45, 0.45, 0.45])
                sns.lineplot(data=sub_df, x="q", y="avg_price", hue="mode", ax=ax_ins, legend=False)
                # CLOB depth markers in inset as well
                ins_x_offset = 0.01 * (x1 - x0)
                for r in rows:
                    if r["clob_avg"] is not None:
                        x = float(r["q"])
                        if x == 0.0 or x < x0 or x > x1:
                            continue
                        y_top = float(r["clob_avg"])
                        ax_ins.vlines(x, ymin=0.0, ymax=y_top, colors=clob_color, linestyles=":", linewidth=0.6, alpha=0.7)
                        # Minimal ticker label for inset CLOB markers at mid-line (within visible y-range), slightly to the right
                        idx = clob_index_by_q.get(x)
                        if idx is None:
                            continue
                        label = f"L{idx} {int(x)}"
                        y_mid = 0.5 * (y_min + y_top)
                        ax_ins.text(x + ins_x_offset, y_mid, label, rotation=270, va="center", ha="left",
                                    fontsize=10, color=clob_color)
                ax_ins.set_xlim(x0, x1)
                ax_ins.set_ylim(y_min - y_pad, y_max + y_pad)
                ax_ins.set_xlabel("")
                ax_ins.set_ylabel("")
                ax_ins.tick_params(labelsize=10)

        plt.tight_layout()
        plt.show()
