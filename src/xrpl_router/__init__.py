from __future__ import annotations
"""Central API surface for repository-internal imports.

Production vs research layering
-------------------------------
- **Production (whitepaper-aligned)**: use `BookStep` for execution (per-iteration
  Phase A/B inside the step), `AMM` for pool maths, and `AMMContext` for AMM usage
  bookkeeping. These symbols are exported at module top-level.
- **Research / legacy**: the old router (`route`) and execution-mode helpers live
  under the research section below and are not part of the stable production API.

This separation keeps production code aligned with the whitepaper while still
making it convenient to run experiments and scans from the same package.
"""

# -----------------
# Production (stable intra-repo surface)
# -----------------
from .core import Segment, RouteResult, ExecutionReport
from .book_step import BookStep
from .amm_context import AMMContext
from .amm import AMM

__all__ = [
    # core data types
    "Segment",
    "RouteResult",
    "ExecutionReport",
    # execution building blocks
    "BookStep",
    "AMMContext",
    "AMM",
]

# -----------------
# Research / legacy (non-stable)
# -----------------
# Heavyweight helpers for notebooks and internal studies; API may change.
from .path_builder import route  # legacy router (research-only)
from .exec_modes import ExecutionMode, run_trade_mode
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