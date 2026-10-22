"""Runtime audit of the floating-point environment used by AutoIA's trusted kernel.

The fast trusted backend relies on a deliberately narrow platform contract:
Python ``float`` and NumPy ``float64`` must be IEEE-754 binary64, arithmetic must
currently use round-to-nearest for the operations exercised by the kernel, and
gradual underflow/subnormal arithmetic must be active.  This module checks those
assumptions before a rigorous backend is used.

The audit is intentionally small and deterministic.  It is not a proof of a
compiler or CPU implementation; it is a fail-closed runtime check of the
observable properties on which the reference backend relies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
import sys
import numpy as np


class Binary64EnvironmentError(FloatingPointError):
    """The current runtime does not satisfy AutoIA's binary64 trust contract."""


@dataclass(frozen=True)
class Binary64EnvironmentReport:
    """Result of the runtime binary64 environment audit."""

    supported: bool
    checks: tuple[tuple[str, bool, str], ...]

    @property
    def failed_checks(self) -> tuple[tuple[str, bool, str], ...]:
        return tuple(item for item in self.checks if not item[1])

    def summary(self) -> str:
        if self.supported:
            return "binary64 environment audit passed"
        return "; ".join(f"{name}: {detail}" for name, _, detail in self.failed_checks)


def nextafter_binary64(x: float, toward: float) -> float:
    """Return the adjacent NumPy float64 value without leaking FP warnings."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return float(np.nextafter(np.float64(x), np.float64(toward)))


def audit_binary64_environment() -> Binary64EnvironmentReport:
    """Audit the observable floating-point assumptions used by the reference kernel.

    This function intentionally performs the checks each time it is called.  A
    rigorous backend invokes it at construction, so a changed process floating-
    point environment is detected at the start of each AutoIA execution.
    """

    checks: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append((name, bool(ok), str(detail)))

    fi = sys.float_info
    add(
        "python-float-format",
        fi.radix == 2 and fi.mant_dig == 53 and fi.max_exp == 1024 and fi.min_exp == -1021,
        f"radix={fi.radix}, mant_dig={fi.mant_dig}, max_exp={fi.max_exp}, min_exp={fi.min_exp}",
    )
    add("c-double-size", struct.calcsize("d") == 8, f"sizeof(double)={struct.calcsize('d')}")
    add("numpy-float64-size", np.dtype(np.float64).itemsize == 8,
        f"itemsize={np.dtype(np.float64).itemsize}")

    # The public contract currently states round-to-nearest.  sys.float_info.rounds
    # reflects FLT_ROUNDS at interpreter startup; the arithmetic tie test below
    # checks the behavior actually observed now.
    add("startup-rounding-mode", fi.rounds == 1, f"sys.float_info.rounds={fi.rounds}")

    one = float(1.0)
    half_ulp_at_one = math.ldexp(1.0, -53)
    tie_pos = one + half_ulp_at_one
    tie_neg = -one - half_ulp_at_one
    tie_sub = one - math.ldexp(1.0, -54)
    add(
        "current-round-to-nearest-ties",
        tie_pos == one and tie_neg == -one and tie_sub == one,
        (
            f"1+2^-53={tie_pos.hex()}, -1-2^-53={tie_neg.hex()}, "
            f"1-2^-54={tie_sub.hex()}"
        ),
    )

    # Division is proof-critical in the scalar backend and in the elementary
    # kernels.  Check one ordinary inexact quotient whose correctly rounded
    # binary64 encoding is fixed by IEEE-754.
    one_tenth = 1.0 / 10.0
    add(
        "binary64-division-rounding",
        one_tenth.hex() == "0x1.999999999999ap-4",
        f"1/10={one_tenth.hex()}",
    )

    min_sub = float.fromhex("0x0.0000000000001p-1022")
    two_min_sub = float.fromhex("0x0.0000000000002p-1022")
    min_normal = float.fromhex("0x1.0000000000000p-1022")
    half_min_normal = float.fromhex("0x0.8000000000000p-1022")

    got_next = nextafter_binary64(0.0, math.inf)
    add(
        "nextafter-min-subnormal",
        got_next == min_sub,
        f"nextafter(0,+inf)={got_next.hex()}",
    )
    got_prev = nextafter_binary64(0.0, -math.inf)
    add(
        "nextafter-negative-min-subnormal",
        got_prev == -min_sub,
        f"nextafter(0,-inf)={got_prev.hex()}",
    )

    # These operations detect common flush-to-zero / denormals-are-zero modes.
    sub_add = min_sub + min_sub
    sub_mul = min_sub * 2.0
    normal_to_sub = min_normal * 0.5
    add(
        "gradual-underflow-add",
        sub_add == two_min_sub,
        f"min_sub+min_sub={sub_add.hex()}",
    )
    add(
        "gradual-underflow-mul",
        sub_mul == two_min_sub,
        f"min_sub*2={sub_mul.hex()}",
    )
    add(
        "gradual-underflow-normal-to-subnormal",
        normal_to_sub == half_min_normal,
        f"min_normal*0.5={normal_to_sub.hex()}",
    )

    # Probe round-to-nearest-even at the very bottom of the subnormal lattice.
    # These values are particularly sensitive to flush-to-zero and nonstandard
    # underflow handling.
    half_min_sub = min_sub / 2.0
    three_halves_min_sub = (3.0 * min_sub) / 2.0
    add(
        "subnormal-tie-to-even",
        half_min_sub == 0.0 and three_halves_min_sub == two_min_sub,
        (
            f"min_sub/2={half_min_sub.hex()}, "
            f"3*min_sub/2={three_halves_min_sub.hex()}"
        ),
    )

    max_finite = sys.float_info.max
    got_inf = nextafter_binary64(max_finite, math.inf)
    add(
        "nextafter-max-to-infinity",
        math.isinf(got_inf) and got_inf > 0.0,
        f"nextafter(max,+inf)={got_inf}",
    )

    # This is a useful guard against an execution model that retains wider
    # precision across ordinary Python float operations.
    big = float(2**53)
    extended_probe = (big + 1.0) - big
    add(
        "binary64-intermediate-rounding",
        extended_probe == 0.0,
        f"(2^53+1)-2^53={extended_probe!r}",
    )

    # Step 9 vectorized trusted reductions use NumPy elementwise float64 add,
    # multiply, and nextafter.  Audit those observable operations separately so
    # the vectorized path never silently assumes a wider/other dtype execution.
    va = np.asarray([1.0, min_sub, big], dtype=np.float64)
    vb = np.asarray([half_ulp_at_one, 2.0, 1.0], dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        vadd = np.add(va, vb, dtype=np.float64)
        vmul = np.multiply(va, vb, dtype=np.float64)
        vnext = np.nextafter(np.asarray([0.0, 1.0], dtype=np.float64), np.float64(np.inf))
    add(
        "numpy-vectorized-add-binary64",
        vadd.dtype == np.float64 and float(vadd[0]) == 1.0 and float(vadd[2]) == big,
        f"dtype={vadd.dtype}, values={[float(x).hex() for x in vadd]}",
    )
    add(
        "numpy-vectorized-mul-binary64",
        vmul.dtype == np.float64 and float(vmul[1]) == two_min_sub and float(vmul[2]) == big,
        f"dtype={vmul.dtype}, values={[float(x).hex() for x in vmul]}",
    )
    add(
        "numpy-vectorized-nextafter",
        vnext.dtype == np.float64 and float(vnext[0]) == min_sub and float(vnext[1]) == nextafter_binary64(1.0, math.inf),
        f"dtype={vnext.dtype}, values={[float(x).hex() for x in vnext]}",
    )

    return Binary64EnvironmentReport(
        supported=all(ok for _, ok, _ in checks),
        checks=tuple(checks),
    )


def require_binary64_environment() -> Binary64EnvironmentReport:
    """Return the audit report or fail closed if the environment is unsupported."""
    report = audit_binary64_environment()
    if not report.supported:
        raise Binary64EnvironmentError(
            "AutoIA's rigorous binary64 backend cannot be used in the current "
            f"floating-point environment: {report.summary()}"
        )
    return report


__all__ = [
    "Binary64EnvironmentError",
    "Binary64EnvironmentReport",
    "nextafter_binary64",
    "audit_binary64_environment",
    "require_binary64_environment",
]
