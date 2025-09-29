from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

# Per whitepaper §1.2.7.3: cap the number of iterations in which AMM liquidity
# is actually consumed to avoid runaway growth of synthetic AMM offers and to
# keep path ordering stable in multi-path execution.
_MAX_ITERS_WITH_AMM = 30


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
    """

    # Whether the current execution is multi-path (pre-wired; not strictly used
    # by the single-path router but kept for future Strand/Step integration).
    multi_path: bool = False

    # Internal counter of iterations in which AMM liquidity was consumed.
    _amm_used_iters: int = 0

    _fib_prev: Decimal = Decimal(0)
    _fib_curr: Decimal = Decimal(0)
    _fib_inited: bool = False

    def set_multi_path(self, flag: bool) -> None:
        """Enable/disable multi-path mode (for future Strand/Step integration)."""
        self.multi_path = bool(flag)

    def mark_amm_used_this_iter(self) -> None:
        """
        Record that the current iteration consumed AMM liquidity.

        IMPORTANT:
        Call this once per router iteration where AMM in/out totals are > 0.
        """
        self._amm_used_iters += 1

    @property
    def amm_used_iters(self) -> int:
        """Return the number of iterations in which AMM liquidity was consumed."""
        return self._amm_used_iters

    def max_iters_reached(self) -> bool:
        """
        Return True if the maximum number of AMM-consuming iterations was reached.

        When this returns True, callers (router/anchor/curve providers) MUST stop
        generating AMM synthetic offers for the current and subsequent iterations,
        per the whitepaper’s iteration cap.
        """
        return self._amm_used_iters >= _MAX_ITERS_WITH_AMM

    # ---------------- Fibonacci cap for multi-path AMM (whitepaper §1.2.7.3) ----------------
    def reset_fib(self, base: Decimal) -> None:
        """Initialise Fibonacci state with a positive base cap (in OUT units).
        Subsequent caps follow base, base, 2*base, 3*base, 5*base, ...
        """
        if base <= 0:
            base = Decimal(0)
        self._fib_prev = base
        self._fib_curr = base
        self._fib_inited = True

    def current_fib_cap(self, base: Decimal) -> Decimal:
        """Return the current Fibonacci-based OUT cap. Lazy-initialises if needed."""
        if not self._fib_inited:
            self.reset_fib(base)
        return self._fib_curr

    def advance_fib(self) -> None:
        """Advance Fibonacci state to the next cap: prev, curr = curr, prev+curr."""
        if not self._fib_inited:
            return
        nxt = self._fib_prev + self._fib_curr
        self._fib_prev = self._fib_curr
        self._fib_curr = nxt

__all__ = ["AMMContext"]