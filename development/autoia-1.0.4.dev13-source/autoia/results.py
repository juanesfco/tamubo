"""Public result types without research-version suffixes."""

from .controller import (
    AutoIAResult,
    AdaptiveSubdivisionResult,
    PiecewiseResult,
    AutoIAPathwiseResult, AutoIAConfiguredResult,
)

PathwiseResult = AutoIAPathwiseResult

__all__ = [
    "AutoIAResult", "AdaptiveSubdivisionResult", "PiecewiseResult",
    "AutoIAPathwiseResult", "PathwiseResult", "AutoIAConfiguredResult",
]
