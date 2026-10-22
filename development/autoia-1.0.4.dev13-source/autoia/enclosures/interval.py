"""Dependency-blind scalar interval enclosure."""

from __future__ import annotations

import numpy as np

from ..trusted import Interval, point_interval, interval_neg, RigorousBinary64Backend


class IntervalEnclosure:
    """Scalar interval-only representation ``z in X``."""

    representation_name = "interval"
    promotion_priority = 0
    supports_value_arithmetic = True

    def __init__(self, interval, h, order, backend=None,
                 range_method="portfolio"):
        if not isinstance(interval, Interval):
            raise TypeError("IntervalEnclosure requires an Interval.")
        self.interval = interval
        self.h = np.asarray(h, dtype=float)
        self.m = len(self.h)
        self.order = int(order)
        self.backend = backend if backend is not None else RigorousBinary64Backend()
        self.range_method = range_method

    @staticmethod
    def constant(value, h, order, backend=None, range_method="portfolio"):
        return IntervalEnclosure(
            point_interval(float(value)), h, order, backend, range_method
        )

    def copy(self):
        return IntervalEnclosure(
            self.interval, self.h.copy(), self.order,
            self.backend, self.range_method,
        )

    def range(self, method=None):
        return self.interval, {"method": "interval"}

    # ------------------------------------------------------------------
    # Representation contract used by ValidatedValue
    # ------------------------------------------------------------------
    def constant_like(self, value):
        return IntervalEnclosure.constant(
            value, self.h, self.order, self.backend, self.range_method
        )

    def negated(self):
        return IntervalEnclosure(
            interval_neg(self.interval), self.h, self.order,
            self.backend, self.range_method,
        )

    def unary(self, op, *, domain=None):
        from .elementary import apply_interval_elementary
        return apply_interval_elementary(self, op, domain=domain)

    def power_constant(self, power, *, domain=None):
        from .power import power_representation
        return power_representation(self, power, domain=domain)

    def __pow__(self, power):
        return self.power_constant(power)

    def binary_same(self, other, op):
        if getattr(other, "representation_name", None) != "interval":
            raise TypeError("IntervalEnclosure.binary_same requires interval peer.")
        if op == "add":
            I = self.backend.add(self.interval, other.interval)
        elif op == "mul":
            I = self.backend.mul(self.interval, other.interval)
        elif op == "div":
            if other.interval.lo <= 0.0 <= other.interval.hi:
                from .domain import DomainRequirement, require_precomputed_domain
                require_precomputed_domain(
                    other,
                    DomainRequirement(
                        "reciprocal", "nonzero", 0.0,
                        "denominator separated from zero",
                    ),
                    other.interval,
                    methods_tried=("interval",),
                )
            I = self.backend.div(self.interval, other.interval)
        else:
            raise ValueError(op)
        return IntervalEnclosure(
            I, self.h, min(self.order, other.order),
            self.backend, self.range_method,
        )

    def convert_from(self, other):
        """Soundly convert any range-capable scalar representation to interval."""
        if getattr(other, "representation_name", None) == "interval":
            return other.copy()
        r, _ = other.range(getattr(other, "range_method", None))
        return IntervalEnclosure(
            r, other.h, other.order, other.backend, other.range_method
        )

    def __repr__(self):
        return (
            f"IntervalEnclosure([{self.interval.lo:.12g}, "
            f"{self.interval.hi:.12g}])"
        )


__all__ = ["IntervalEnclosure"]
