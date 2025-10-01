from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

# Per whitepaper §1.2.7.3: cap the number of iterations in which AMM liquidity
# is actually consumed to avoid runaway growth of synthetic AMM offers and to
# keep path ordering stable in multi-path execution.
MAX_AMM_ITERS = 30


@dataclass
class AMMContext:
    """
    Execution context for AMM synthetic offer generation and usage bookkeeping.

    Whitepaper-aligned semantics (see §1.2.7.3):
      - Track whether we are in multi-path mode (flag only; useful when wiring
        this context into a full Strand/Step flow later).
      - Count iterations in which AMM liquidity was actually consumed.
      - Stop generating AMM synthetic offers once the cap (30) is reached.

    Notes:
      * In this codebase we currently model a single-path router. The context is
        introduced to preserve parity with the whitepaper and to make it easy
        to extend to multi-path/Strand execution without further refactors.
      * “AMM liquidity consumed” means that in a given router iteration the
        aggregated AMM in/out was strictly positive (i.e., the iteration filled
        some amount from the AMM). Only those iterations advance the counter.
      * This context only tracks the multipath flag and AMM-using iteration count,
        per whitepaper §1.2.7.3.
    """

    # Whether the current execution is multi-path (pre-wired; not strictly used
    # by the single-path router but kept for future Strand/Step integration).
    multi_path: bool = False

    # Internal counter of iterations in which AMM liquidity was consumed.
    _amm_used_iters: int = 0

    def setMultiPath(self, flag: bool) -> None:
        """Enable/disable multi-path mode (for future Strand/Step integration)."""
        self.multi_path = bool(flag)

    def setAMMUsed(self) -> None:
        """
        Record that the current iteration consumed AMM liquidity.

        IMPORTANT:
        Call this once per router iteration where AMM in/out totals are > 0.
        Does nothing if max iterations reached.
        """
        if self._amm_used_iters < MAX_AMM_ITERS:
            self._amm_used_iters += 1

    @property
    def ammUsedIters(self) -> int:
        """Return the number of iterations in which AMM liquidity was consumed."""
        return self._amm_used_iters

    def reset_iters(self) -> None:
        """Reset the AMM-used iterations counter to zero.

        Safe to call between trades or before running a new routing session.
        """
        self._amm_used_iters = 0

    def maxItersReached(self) -> bool:
        """
        Return True if the maximum number of AMM-consuming iterations was reached.

        When this returns True, callers (router/anchor/curve providers) MUST stop
        generating AMM synthetic offers for the current and subsequent iterations,
        per the whitepaper’s iteration cap.
        """
        return self._amm_used_iters >= MAX_AMM_ITERS

    def reset_all(self) -> None:
        """Reset iteration counter.

        Intended for test setup or to start a fresh trade session without
        carrying over state.
        """
        self._amm_used_iters = 0

__all__ = ["AMMContext", "MAX_AMM_ITERS"]