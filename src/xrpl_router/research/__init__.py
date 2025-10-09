"""
Research utilities (non-stable API).

Only efficiency_scan is retained for post-hoc analysis. This is NOT part of the
stable, production-facing API and may change without notice.
"""
from __future__ import annotations

from .efficiency_scan import (
    AlphaPoint,
    AlphaScanResult,
    analyze_alpha_scan,
    CrossoverSample,
    CrossoverResult,
    find_crossover_q,
    HybridFlowResult,
    hybrid_flow,
    HybridVsAlpha,
    compare_hybrid_vs_alpha,
    BatchRow,
    batch_analyze,
    batch_rows_to_csv,
    summarize_hybrid,
    summarize_alpha_scan,
    summarize_crossover,
    ExecutionMode,
    run_trade_mode,
)

__all__ = [
    "AlphaPoint",
    "AlphaScanResult",
    "analyze_alpha_scan",
    "CrossoverSample",
    "CrossoverResult",
    "find_crossover_q",
    "HybridFlowResult",
    "hybrid_flow",
    "HybridVsAlpha",
    "compare_hybrid_vs_alpha",
    "BatchRow",
    "batch_analyze",
    "batch_rows_to_csv",
    "summarize_hybrid",
    "summarize_alpha_scan",
    "summarize_crossover",
    "ExecutionMode",
    "run_trade_mode",
]