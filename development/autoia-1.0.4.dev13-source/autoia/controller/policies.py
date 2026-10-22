"""Controller policy objects.

These dataclasses control optional work only.  Soundness never depends on a
policy choosing a sharp or inexpensive strategy; skipped optional tightening
retains an already rigorous enclosure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RangePolicy:
    """Generic structure- and cost-aware range policy."""

    internal_method: str = "termwise"
    final_method: str = "adaptive"
    bernstein_max_coefficients: int = 256
    proof_bernstein_max_coefficients: int = 256

    def __post_init__(self):
        if self.internal_method not in {"termwise", "monotonicity", "bernstein", "portfolio"}:
            raise ValueError("Invalid internal_method.")
        if self.final_method not in {
            "adaptive", "termwise", "monotonicity", "bernstein", "portfolio"
        }:
            raise ValueError("Invalid final_method.")
        if self.bernstein_max_coefficients < 1:
            raise ValueError("bernstein_max_coefficients must be positive.")
        if self.proof_bernstein_max_coefficients < 1:
            raise ValueError("proof_bernstein_max_coefficients must be positive.")


@dataclass(frozen=True)
class SubdivisionPolicy:
    """Generic adaptive-subdivision policy.

    ``split_method="cheap"`` uses the release-oriented structural score and
    certifies only the selected children. ``split_method="lookahead"`` retains
    the expensive Version-0.11 one-step probe controller as a research oracle.
    """

    max_branches: int = 16
    max_depth: int = 6
    max_probe_directions: int = 2
    min_structural_gain: float = 0.05
    min_relative_width_gain: float = 1.0e-3
    split_method: str = "cheap"

    def __post_init__(self):
        if self.split_method not in {"cheap", "lookahead"}:
            raise ValueError("split_method must be 'cheap' or 'lookahead'.")
        if self.max_branches < 1:
            raise ValueError("max_branches must be positive.")
        if self.max_depth < 0:
            raise ValueError("max_depth must be nonnegative.")
        if self.max_probe_directions < 1:
            raise ValueError("max_probe_directions must be positive.")
        if self.min_structural_gain < 0.0:
            raise ValueError("min_structural_gain must be nonnegative.")
        if self.min_relative_width_gain < 0.0:
            raise ValueError("min_relative_width_gain must be nonnegative.")


@dataclass(frozen=True)
class BranchPolicy:
    """Policy for rigorous uncertain Python control flow.

    ``max_split_depth_per_decision`` controls how many geometric box splits may
    be attempted for one unresolved comparison before path enumeration is used.
    ``max_branches`` and ``max_path_decisions`` are hard fail-closed budgets.
    """

    max_branches: int = 64
    max_path_decisions: int = 24
    max_split_depth_per_decision: int = 2
    max_total_splits: int = 64
    max_domain_split_depth_per_obligation: int = 6
    prefer_geometric_split: bool = True

    def __post_init__(self):
        if self.max_branches < 1:
            raise ValueError("max_branches must be positive.")
        if self.max_path_decisions < 1:
            raise ValueError("max_path_decisions must be positive.")
        if self.max_split_depth_per_decision < 0:
            raise ValueError("max_split_depth_per_decision must be nonnegative.")
        if self.max_total_splits < 0:
            raise ValueError("max_total_splits must be nonnegative.")
        if self.max_domain_split_depth_per_obligation < 0:
            raise ValueError("max_domain_split_depth_per_obligation must be nonnegative.")


__all__ = ["RangePolicy", "SubdivisionPolicy", "BranchPolicy"]
