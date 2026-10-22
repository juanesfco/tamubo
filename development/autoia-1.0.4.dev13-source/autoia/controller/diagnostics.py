"""Explicit schemas for controller diagnostics and strategy events.

The runtime representation remains ordinary dictionaries for backward
compatibility with Version 0.11.  These ``TypedDict`` definitions freeze the
field meanings without forcing an algorithmic or serialization change during
the Step-1 migration.
"""

from __future__ import annotations

from typing import Any, TypedDict

from ..trusted import Interval


class FinalRangeDiagnostic(TypedDict, total=False):
    method: str
    adaptive: bool
    termwise: Interval
    monotonicity: Interval
    cheap_intersection: Interval
    relative_disagreement: float
    degree_vector: tuple[int, ...]
    estimated_bernstein_coefficients: int
    estimated_reduced_bernstein_coefficients: int
    unresolved_variables: tuple[int, ...]
    unresolved_fraction: float
    bernstein_used: bool
    bernstein_reason: str
    bernstein: Interval | None
    bernstein_meta: dict[str, Any] | None
    termwise_meta: dict[str, Any]
    monotonicity_meta: dict[str, Any]


class ProofRangePolicyEvent(TypedDict, total=False):
    kind: str
    methods_tried: tuple[str, ...]
    final_range: Interval
    estimated_bernstein_coefficients: int
    success: bool


class BranchControllerMetrics(TypedDict, total=False):
    unresolved_variables: tuple[int, ...]
    num_unresolved: int
    worst_reduced_bernstein_cost: int
    worst_full_bernstein_cost: int
    capped_outputs: tuple[int, ...]
    needs_subdivision: bool
    total_width: float
    max_width: float
    output_unresolved: tuple[tuple[int, ...], ...]


class SubdivisionProbeDiagnostic(TypedDict, total=False):
    direction: int
    failed: bool
    error: str
    structural_gain: float
    relative_width_gain: float
    unresolved_gain: int
    parent_cost: int
    child_cost: int
    parent_unresolved: int
    child_unresolved: int
    ambiguity_score: float
    predicted_unresolved_gain: int
    predicted_local_resolution_gain: int
    predicted_structural_gain: float
    predicted_tree_cost_gain: float
    predicted_relative_width_gain: float
    predicted_child_cost: int
    predicted_child_unresolved: int
    predicted_total_width: float


class SubdivisionEvent(TypedDict, total=False):
    branch_id: int
    depth: int
    action: str
    split_method: str
    selected_direction: int
    parent_metrics: BranchControllerMetrics
    selected_metrics: dict[str, Any]
    probes: list[SubdivisionProbeDiagnostic]


__all__ = [
    "FinalRangeDiagnostic",
    "ProofRangePolicyEvent",
    "BranchControllerMetrics",
    "SubdivisionProbeDiagnostic",
    "SubdivisionEvent",
]
