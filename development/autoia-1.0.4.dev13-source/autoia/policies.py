"""Public controller-policy objects with stable validation failures."""

from __future__ import annotations

from .controller import RangePolicy as _ControllerRangePolicy
from .controller import SubdivisionPolicy as _ControllerSubdivisionPolicy
from .controller import BranchPolicy as _ControllerBranchPolicy
from .controller import ConfigurationPolicy as _ControllerConfigurationPolicy
from .exceptions import InvalidInputError


class RangePolicy(_ControllerRangePolicy):
    """Public range policy; semantics match the extracted controller policy."""

    def __post_init__(self):
        try:
            super().__post_init__()
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidInputError(str(exc)) from exc


class SubdivisionPolicy(_ControllerSubdivisionPolicy):
    """Public adaptive-subdivision policy with typed validation failures."""

    def __post_init__(self):
        try:
            super().__post_init__()
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidInputError(str(exc)) from exc


class BranchPolicy(_ControllerBranchPolicy):
    """Public path-aware control-flow policy with typed validation failures."""

    def __post_init__(self):
        try:
            super().__post_init__()
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidInputError(str(exc)) from exc


class ConfigurationPolicy(_ControllerConfigurationPolicy):
    """Public automatic-configuration policy with typed validation failures."""

    def __post_init__(self):
        try:
            super().__post_init__()
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidInputError(str(exc)) from exc


__all__ = ["RangePolicy", "SubdivisionPolicy", "BranchPolicy", "ConfigurationPolicy"]
