from __future__ import annotations
"""Central API surface for repository-internal imports.

This module re-exports commonly used types and helpers so other files/tests in this
repository can import from `xrpl_router` directly instead of deep module paths.
"""

# Stable API (intra-repo convenience)
from .core import Segment, RouteResult, ExecutionReport, ROUTING_CFG
from .router import route
from .exec_modes import ExecutionMode, run_trade_mode
from .amm_context import AMMContext
from .amm import AMM

# Analysis / research layer (heavyweight helpers). These are convenient for notebooks
# and internal studies but are not a stable ABI. Keep imports at the end to reduce
# risk of circular import during module initialisation.
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