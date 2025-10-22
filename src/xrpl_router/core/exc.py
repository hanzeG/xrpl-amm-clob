"""
Core exception types for xrpl_router.core.

These are dependency-free and may be imported by all core modules.
"""

__all__ = [
    "AmountDomainError",
    "NormalisationError",
    "InvariantViolation",
    "InsufficientLiquidityError",
    "InsufficientBudgetError",
]


class AmountDomainError(Exception):
    """Raised when inputs violate the non-negative domain or basic preconditions."""
    pass


class NormalisationError(Exception):
    """Raised when a value cannot be normalised within IOUAmount bounds."""
    pass


class InvariantViolation(Exception):
    """Raised when arithmetic or guards would break core invariants."""
    pass


class InsufficientLiquidityError(Exception):
    """Raised when requested output exceeds available combined liquidity.

    Attributes
    ----------
    requested_out : Any
        The requested OUT amount (domain object), for context.
    max_fill_out : Any
        The maximum OUT amount achievable under current constraints.
    filled_out : Any | None
        OUT actually filled before abort (if any, e.g., during a strict run that decides to abort late).
    spent_in : Any | None
        IN actually spent corresponding to filled_out (if any).
    trace : list | None
        Optional trace of segments considered/consumed up to the decision point.
    """

    def __init__(self, requested_out, max_fill_out, *, filled_out=None, spent_in=None, trace=None):
        super().__init__(
            f"Requested out={requested_out} exceeds available liquidity={max_fill_out}"
        )
        self.requested_out = requested_out
        self.max_fill_out = max_fill_out
        self.filled_out = filled_out
        self.spent_in = spent_in
        self.trace = trace


class InsufficientBudgetError(Exception):
    """Raised when send_max / budget prevents achieving requested output."""

    def __init__(self, requested_out, send_max, *, filled_out=None, spent_in=None, trace=None):
        super().__init__(
            f"Budget send_max={send_max} insufficient for requested out={requested_out}"
        )
        self.requested_out = requested_out
        self.send_max = send_max
        self.filled_out = filled_out
        self.spent_in = spent_in
        self.trace = trace
