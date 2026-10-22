"""Machine-readable interface contracts for AutoIA backends/representations.

The mathematical contract is documented in ``CONTRACT.md``.  These Protocols
are structural interfaces: they document the operations that higher layers may
use without depending on concrete representation classes.  They do not, by
themselves, establish numerical soundness.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
import numpy as np

from .trusted import Interval


@runtime_checkable
class TrustedBackend(Protocol):
    """Minimum trusted scalar-arithmetic interface used by proof code."""

    name: str
    machine_rigorous: bool

    def add(self, x: Interval, y: Interval) -> Interval: ...
    def sub(self, x: Interval, y: Interval) -> Interval: ...
    def mul(self, x: Interval, y: Interval) -> Interval: ...
    def div(self, x: Interval, y: Interval) -> Interval: ...


@runtime_checkable
class ScalarRepresentation(Protocol):
    """Common contract for a scalar enclosure representation.

    ``promotion_priority`` is a strategy convention, not a soundness theorem.
    When two different representation families interact, ``ValidatedValue``
    selects the higher-priority family and asks that representation to perform a
    sound conversion through ``convert_from``.
    """

    representation_name: str
    promotion_priority: int
    supports_value_arithmetic: bool

    h: np.ndarray
    m: int
    order: int
    backend: TrustedBackend
    range_method: str

    def copy(self) -> Any: ...
    def range(self, method: str | None = None) -> tuple[Interval, dict]: ...
    def constant_like(self, value: float) -> Any: ...
    def negated(self) -> Any: ...
    def binary_same(self, other: Any, op: str) -> Any: ...
    def convert_from(self, other: Any) -> Any: ...


@runtime_checkable
class ElementaryScalarRepresentation(ScalarRepresentation, Protocol):
    """Optional extension for validated unary elementary functions."""

    def unary(self, op: str) -> Any: ...


@runtime_checkable
class PiecewiseRepresentation(Protocol):
    """Range-capable finite-union representation used outside model arithmetic."""

    representation_name: str
    supports_value_arithmetic: bool
    backend: TrustedBackend

    def copy(self) -> Any: ...
    def range(self, method: str | None = None) -> tuple[Interval, dict]: ...


__all__ = [
    "TrustedBackend",
    "ScalarRepresentation",
    "ElementaryScalarRepresentation",
    "PiecewiseRepresentation",
]
