"""Independent exact-rational audit for AutoIA physical-input construction.

This script verifies, using Python's exact ``Fraction`` arithmetic, that
``Interval.enclosing`` never excludes the exact real interval requested by the
user.  It is intentionally independent of AutoIA's downstream range machinery:
this audit targets only the physical-input specification boundary introduced in
AutoIA 1.0.1.

Run from the source tree with::

    python validation/input_specification/audit_input_specification.py

or against an independently installed wheel by placing that installation first
on ``PYTHONPATH``.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import random
import sys

try:
    from autoia import Interval
except ModuleNotFoundError:
    # Allow direct execution from an unpacked source tree without requiring an
    # editable install.  When an installed AutoIA is already importable (the
    # clean-wheel audit), that installation is used instead.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from autoia import Interval


N_RANDOM = 10_000
SEED = 20260901


def _qfloat(x: float) -> Fraction:
    """Return the exact rational value of one stored binary64 number."""
    return Fraction.from_float(float(x))


def _check(qlo: Fraction, qhi: Fraction) -> None:
    if qlo > qhi:
        qlo, qhi = qhi, qlo
    I = Interval.enclosing(qlo, qhi)
    stored_lo = _qfloat(I.lo)
    stored_hi = _qfloat(I.hi)
    assert stored_lo <= qlo, (stored_lo, qlo, I)
    assert qhi <= stored_hi, (qhi, stored_hi, I)
    assert I.input_semantics == "exact-real-enclosing"
    assert I.input_specification["containment_verified"] is True


def main() -> None:
    checks = 0

    # 81 deterministic exact points spanning signs, zero, binary64-exact
    # rationals, and many nonrepresentable rational values.
    for k in range(-40, 41):
        _check(Fraction(k, 37), Fraction(k, 37))
        checks += 1

    rng = random.Random(SEED)
    for _ in range(N_RANDOM):
        # Keep values finite and moderate while varying numerator, denominator,
        # sign, interval width, and exact representability.
        a = Fraction(rng.randint(-10**9, 10**9), rng.randint(1, 10**9))
        b = Fraction(rng.randint(-10**9, 10**9), rng.randint(1, 10**9))
        _check(a, b)
        checks += 1

    assert checks == 10_081
    print(f"PASS: {checks:,} exact-rational input-enclosure checks")


if __name__ == "__main__":
    main()
