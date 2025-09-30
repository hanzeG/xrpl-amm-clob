from __future__ import annotations
"""Central API surface for repository-internal imports.

This module re-exports commonly used types and helpers so other files/tests in this
repository can import from `xrpl_router` directly instead of deep module paths.
"""

# Core data & config
from .core import Segment, RouteResult, ExecutionReport, ROUTING_CFG

# Routing (single-venue)
from .router import route
# Transitional convenience API (kept for compatibility within this repo)
from .exec_modes import ExecutionMode, run_trade_mode

# AMM multi-path context/state
from .amm_context import AMMContext
# AMM interface type (preview_fees_for_fill)
from .amm import AMM

# Efficiency / analysis layer
from .efficiency_scan import (
    hybrid_flow,
    analyze_alpha_scan,
    find_crossover_q,
    summarize_hybrid,
    summarize_alpha_scan,
    summarize_crossover,
    batch_analyze,
    batch_rows_to_csv,
)