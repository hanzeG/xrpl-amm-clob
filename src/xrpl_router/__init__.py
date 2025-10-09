
"""Top-level API for xrpl_router (integer-domain).

This module exposes the stable, production-facing interface aligned with the XRPL whitepaper:
  - BookStep: unified market step (CLOB/AMM)
  - AMM: automated market maker logic
  - AMMContext: AMM usage bookkeeping (iteration caps, multi-path guards)

Core data types and execution primitives are fully integer-domain and follow XRPL STAmount semantics.

Research-oriented helpers (legacy router, efficiency scans, execution modes)
remain under the `xrpl_router.research` subpackage and are **not** part of the
stable API surface.
"""

# NOTE:
#   This package exposes only production-ready, integer-domain interfaces aligned with XRPL semantics.
#   Decimal-based utilities and experimental routing logic remain under `xrpl_router.research`.
#   Import those explicitly when running analysis or performance experiments.

from __future__ import annotations

"""Top-level API for xrpl_router (integer-domain).

This module exposes the stable, production-facing interface aligned with the XRPL whitepaper:
  - BookStep: unified market step (CLOB/AMM)
  - AMM: automated market maker logic
  - AMMContext: AMM usage bookkeeping (iteration caps, multi-path guards)

Core data types and execution primitives are fully integer-domain and follow XRPL STAmount semantics.

Research-oriented helpers (legacy router, efficiency scans, execution modes)
remain under the `xrpl_router.research` subpackage and are **not** part of the
stable API surface.
"""

# Stable intra-repo surface (production)
from .book_step import BookStep
from .amm_context import AMMContext
from .amm import AMM

# Core data types: keep exported if other modules rely on them.
from .core import (
    Segment,
    RouteResult,
    ExecutionReport,
    STAmount,
    Quality,
    round_in_min,
    round_out_max,
)

__all__ = [
    # execution building blocks
    "BookStep",
    "AMMContext",
    "AMM",
    # core data types (kept for compatibility)
    "Segment",
    "RouteResult",
    "ExecutionReport",
    # core integer-domain types
    "STAmount",
    "Quality",
    "round_in_min",
    "round_out_max",
]

# NOTE:
# Research / legacy utilities (route, efficiency scans, execution modes)
# are intentionally *not* imported at the top-level.
# Use: `from xrpl_router import research` and import from there.