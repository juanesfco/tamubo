"""AutoIA: automatic validated interval analysis for engineering code.

Step-1 package facade for the Version-1 architecture freeze.
"""

from .version import __version__
from .api import autoia, autoia_subdivide, autoia_adaptive_subdivide
from .session import (
    prepare, enclose_many, enclose_many_bounds, PreparedAutoIA,
    PreparedConfiguredAutoIA, PreparedManyBoxExecutor, PreparedSessionStats, PreparedConfiguredSessionStats,
    ManyBoxExecutionInfo, ManyBoxFailure, ManyBoxCertificationReport,
    ManyBoxBoundsItemReport, CertificateCache,
)
from .intervals import Interval
from .accelerators import (
    AcceleratorInfo, ProposalBatchResult, available_accelerators,
    resolve_proposal_device, proposal_map, proposal_box_centers,
)
from .backends import RigorousBinary64Backend
from .policies import RangePolicy, SubdivisionPolicy, BranchPolicy, ConfigurationPolicy
from .results import AutoIAResult, AdaptiveSubdivisionResult, PiecewiseResult, AutoIAPathwiseResult, PathwiseResult, AutoIAConfiguredResult
from .representations import (
    ValidatedValue,
    PolynomialEnclosure,
    IntervalEnclosure,
    PiecewiseEnclosure,
    SharedGeneratorPolynomialEnclosure,
)
from .semantic.spd_matrix import SPDMatrixCertificate, certify_spd_matrix, validated_logdet as logdet_spd, validated_quad_solve as quad_solve_spd
from .special import (
    erf, standard_normal_pdf, standard_normal_cdf, normal_pdf, normal_cdf, norm_pdf, norm_cdf,
)
from .application import (
    ApplicationBox, CertifiedInterval, CertificationContext, UpperBound, LowerBound,
    BoundPortfolioResult, LowerBoundPortfolioResult, CertifiedSPDMatrix, ApplicationTheorem, application_theorem, minimum_upper_bound, maximum_lower_bound,
)
from .exceptions import (
    AutoIAError,
    InvalidInputError,
    UnsupportedOperationError,
    CertificationError,
    ResourceLimitError,
    NumericalLimitError,
    ModelExecutionError,
    InternalAutoIAError,
)

__all__ = [
    "__version__",
    "autoia",
    "autoia_subdivide",
    "autoia_adaptive_subdivide",
    "prepare",
    "enclose_many",
    "enclose_many_bounds",
    "PreparedAutoIA",
    "PreparedConfiguredAutoIA",
    "PreparedManyBoxExecutor",
    "PreparedSessionStats",
    "PreparedConfiguredSessionStats",
    "ManyBoxExecutionInfo",
    "ManyBoxFailure",
    "ManyBoxCertificationReport",
    "ManyBoxBoundsItemReport",
    "CertificateCache",
    "AcceleratorInfo",
    "ProposalBatchResult",
    "available_accelerators",
    "resolve_proposal_device",
    "proposal_map",
    "proposal_box_centers",
    "Interval",
    "RigorousBinary64Backend",
    "RangePolicy",
    "SubdivisionPolicy",
    "BranchPolicy",
    "ConfigurationPolicy",
    "AutoIAResult",
    "AdaptiveSubdivisionResult",
    "PiecewiseResult",
    "AutoIAPathwiseResult",
    "PathwiseResult",
    "AutoIAConfiguredResult",
    "ValidatedValue",
    "PolynomialEnclosure",
    "IntervalEnclosure",
    "PiecewiseEnclosure",
    "SharedGeneratorPolynomialEnclosure",
    "erf",
    "standard_normal_pdf",
    "standard_normal_cdf",
    "normal_pdf",
    "normal_cdf",
    "norm_pdf",
    "norm_cdf",
    "SPDMatrixCertificate",
    "certify_spd_matrix",
    "logdet_spd",
    "quad_solve_spd",
    "AutoIAError",
    "InvalidInputError",
    "UnsupportedOperationError",
    "CertificationError",
    "ResourceLimitError",
    "NumericalLimitError",
    "ModelExecutionError",
    "InternalAutoIAError",
    "ApplicationBox",
    "CertifiedInterval",
    "CertificationContext",
    "UpperBound",
    "LowerBound",
    "BoundPortfolioResult",
    "LowerBoundPortfolioResult",
    "CertifiedSPDMatrix",
    "ApplicationTheorem",
    "application_theorem",
    "minimum_upper_bound",
    "maximum_lower_bound",
]
