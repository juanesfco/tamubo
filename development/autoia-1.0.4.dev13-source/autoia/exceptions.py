"""Stable public exception taxonomy for AutoIA.

The numerical reference core still raises several Python built-in exception
classes internally.  The public API translates those implementation details into
this small hierarchy so callers can react to *why* certification stopped without
depending on research-core exception choices.

No exception in this module weakens AutoIA's fail-closed rule: if certification
cannot be established, no result marked rigorous is returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class AutoIAError(Exception):
    """Base class for all failures intentionally exposed by the AutoIA API."""

    code = "autoia-error"

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, Any] | None = None,
        suggestions: Sequence[str] = (),
    ) -> None:
        super().__init__(str(message))
        self.message = str(message)
        self.context = dict(context or {})
        self.suggestions = tuple(str(s) for s in suggestions)

    def __str__(self) -> str:
        return self.message


class InvalidInputError(AutoIAError):
    """The public AutoIA call or configuration is malformed."""

    code = "invalid-input"


class UnsupportedOperationError(AutoIAError):
    """The engineering program uses semantics outside AutoIA's supported subset."""

    code = "unsupported-operation"


class CertificationError(AutoIAError):
    """The request is meaningful, but AutoIA could not prove a required condition."""

    code = "certification-failed"


class ResourceLimitError(AutoIAError):
    """Certification was stopped by an explicit computational/resource limit."""

    code = "resource-limit"


class NumericalLimitError(AutoIAError):
    """The trusted/reference numerical layer encountered an arithmetic limit."""

    code = "numerical-limit"


class ModelExecutionError(AutoIAError):
    """The user's model itself raised an exception unrelated to an AutoIA proof."""

    code = "model-execution-failed"


class InternalAutoIAError(AutoIAError):
    """An AutoIA internal invariant failed; this should be treated as a bug."""

    code = "internal-error"


__all__ = [
    "AutoIAError",
    "InvalidInputError",
    "UnsupportedOperationError",
    "CertificationError",
    "ResourceLimitError",
    "NumericalLimitError",
    "ModelExecutionError",
    "InternalAutoIAError",
]
