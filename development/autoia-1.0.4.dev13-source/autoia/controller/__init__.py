"""AutoIA strategy/controller layer."""

from .policies import RangePolicy, SubdivisionPolicy, BranchPolicy
from .configuration import (
    ConfigurationPolicy, AutoIAConfiguredResult, configuration_candidates, automatic_escalation_candidates, interval_result_dominates,
    select_dominating_result, rigorous_intersection_bounds, structural_summary,
)
from .diagnostics import (
    FinalRangeDiagnostic, ProofRangePolicyEvent, BranchControllerMetrics,
    SubdivisionProbeDiagnostic, SubdivisionEvent,
)
from .ranging import adaptive_final_range
from .execution import (
    AutoIAResult,
    AutoIAResultV11,
    rigorous_center_and_halfwidth,
    resolve_range_policy,
    autoia_single,
)
from .branching import AutoIAPathwiseResult, autoia_pathwise
autoia = autoia_pathwise
from .subdivision import (
    subdivide_parameter_box,
    AutoIAPiecewiseResult,
    PiecewiseResult,
    autoia_subdivide,
    AutoIAAdaptiveSubdivisionResult,
    AdaptiveSubdivisionResult,
    autoia_adaptive_subdivide,
)

__all__ = [
    "RangePolicy", "SubdivisionPolicy", "BranchPolicy", "ConfigurationPolicy", "AutoIAConfiguredResult",
    "configuration_candidates", "automatic_escalation_candidates", "interval_result_dominates",
    "select_dominating_result", "rigorous_intersection_bounds", "structural_summary",
    "adaptive_final_range",
    "FinalRangeDiagnostic", "ProofRangePolicyEvent", "BranchControllerMetrics",
    "SubdivisionProbeDiagnostic", "SubdivisionEvent",
    "AutoIAResult", "AutoIAResultV11", "rigorous_center_and_halfwidth",
    "resolve_range_policy", "autoia_single", "AutoIAPathwiseResult", "autoia", "subdivide_parameter_box",
    "AutoIAPiecewiseResult", "PiecewiseResult", "autoia_subdivide",
    "AutoIAAdaptiveSubdivisionResult", "AdaptiveSubdivisionResult",
    "autoia_adaptive_subdivide",
]
