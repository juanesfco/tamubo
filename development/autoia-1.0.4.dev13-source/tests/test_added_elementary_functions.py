"""Coverage and high-precision audits for the 1.0.4.dev6 elementary additions."""
from __future__ import annotations

import math
import unittest
import numpy as np

from autoia import autoia, Interval, CertificationError
from autoia.trusted import (
    RigorousBinary64Backend,
    Interval as TrustedInterval,
    absolute_interval,
    sinh_point_interval, sinh_interval,
    cosh_point_interval, cosh_interval,
    tanh_point_interval, tanh_interval,
    arctan_point_interval, arctan_interval,
    log1p_point_interval, log1p_interval,
)

from numpy import tanh as direct_np_tanh
from numpy import abs as direct_np_abs
from math import atan as direct_math_atan

try:
    import mpmath as mp
except ImportError:  # pragma: no cover
    mp = None


def _mp_exact_float(x: float):
    n, d = float(x).as_integer_ratio()
    return mp.mpf(n) / mp.mpf(d)


def _contains_mp(I, y):
    return _mp_exact_float(I.lo) <= y <= _mp_exact_float(I.hi)


class TestAddedElementaryExecution(unittest.TestCase):
    def _check_numpy(self, name, box, exact, *, order=6, representation="polynomial"):
        def model(p):
            return getattr(np, name)(p[0])
        R = autoia(model, [box], order=order, representation=representation)
        for x in np.linspace(box.lo, box.hi, 801):
            y = exact(float(x))
            self.assertLessEqual(R.lower[0], y, (name, x, R))
            self.assertGreaterEqual(R.upper[0], y, (name, x, R))
        return R

    def test_numpy_polynomial_execution(self):
        cases = [
            ("tanh", Interval(-1.0, 1.0), math.tanh),
            ("sinh", Interval(-0.8, 0.9), math.sinh),
            ("arctan", Interval(-2.0, 2.0), math.atan),
            ("abs", Interval(-1.0, 2.0), abs),
            ("log1p", Interval(-0.4, 1.2), math.log1p),
            ("cosh", Interval(-0.8, 0.9), math.cosh),
        ]
        for name, box, exact in cases:
            with self.subTest(function=name):
                R = self._check_numpy(name, box, exact)
                self.assertTrue(R.machine_rigorous)

    def test_interval_representation_execution(self):
        cases = [
            ("tanh", Interval(-1.0, 1.0), math.tanh),
            ("sinh", Interval(-0.8, 0.9), math.sinh),
            ("arctan", Interval(-2.0, 2.0), math.atan),
            ("abs", Interval(-1.0, 2.0), abs),
            ("log1p", Interval(-0.4, 1.2), math.log1p),
            ("cosh", Interval(-0.8, 0.9), math.cosh),
        ]
        for name, box, exact in cases:
            with self.subTest(function=name):
                self._check_numpy(name, box, exact, order=4, representation="interval")

    def test_module_and_direct_alias_dispatch(self):
        def model(p):
            return np.array([
                direct_np_tanh(p[0]),
                direct_np_abs(p[0]),
                direct_math_atan(p[0]),
                math.sinh(p[0]),
                math.cosh(p[0]),
                math.tanh(p[0]),
                math.log1p(p[0]),
            ], dtype=object)
        box = Interval(-0.2, 0.3)
        R = autoia(model, [box], order=5)
        for x in np.linspace(box.lo, box.hi, 101):
            vals = [math.tanh(x), abs(x), math.atan(x), math.sinh(x), math.cosh(x), math.tanh(x), math.log1p(x)]
            for j, y in enumerate(vals):
                self.assertLessEqual(R.lower[j], y)
                self.assertGreaterEqual(R.upper[j], y)

    def test_abs_preserves_sign_definite_dependency_and_handles_crossing(self):
        # Direct range checks are important in addition to the cancellation test
        # below: a sign-definite positive interval must not spuriously widen down
        # to zero.  Exercise both the default structured route and interval-only
        # execution, plus Python's built-in abs dispatch.
        def positive_value(p): return np.abs(p[0])
        for representation in (None, "interval"):
            kwargs = {} if representation is None else {"representation": representation}
            Rv = autoia(positive_value, [Interval(0.2, 0.6)], order=5, **kwargs)
            self.assertGreaterEqual(Rv.lower[0], 0.2 - 2e-15)
            self.assertLessEqual(Rv.upper[0], 0.6 + 2e-15)

        def positive_builtin(p): return abs(p[0])
        Rb = autoia(positive_builtin, [Interval(0.2, 0.6)], order=5)
        self.assertGreaterEqual(Rb.lower[0], 0.2 - 2e-15)
        self.assertLessEqual(Rb.upper[0], 0.6 + 2e-15)

        def negative_value(p): return np.abs(p[0])
        Rn = autoia(negative_value, [Interval(-0.6, -0.2)], order=5)
        self.assertGreaterEqual(Rn.lower[0], 0.2 - 2e-15)
        self.assertLessEqual(Rn.upper[0], 0.6 + 2e-15)

        def positive(p): return np.abs(p[0]) - p[0]
        Rp = autoia(positive, [Interval(0.2, 0.7)], order=5)
        self.assertLessEqual(Rp.lower[0], 0.0)
        self.assertGreaterEqual(Rp.upper[0], 0.0)
        self.assertLess(Rp.width[0], 1e-12)

        def crossing(p): return np.abs(p[0])
        Rc = autoia(crossing, [Interval(-0.4, 0.7)], order=5)
        self.assertGreaterEqual(Rc.lower[0], 0.0)
        self.assertGreaterEqual(Rc.upper[0], 0.7)

    def test_log1p_domain_failure_is_fail_closed(self):
        def bad(p): return np.log1p(p[0])
        with self.assertRaises(CertificationError):
            autoia(bad, [Interval(-1.1, 0.2)], order=5)

    def test_object_array_dispatch(self):
        def model(p):
            x = np.array([p[0], p[0] + 0.1], dtype=object)
            return np.tanh(x) + np.abs(x)
        R = autoia(model, [Interval(-0.2, 0.2)], order=5)
        self.assertEqual(len(R.lower), 2)
        for x in np.linspace(-0.2, 0.2, 101):
            vals = [math.tanh(x)+abs(x), math.tanh(x+0.1)+abs(x+0.1)]
            for j, y in enumerate(vals):
                self.assertLessEqual(R.lower[j], y)
                self.assertGreaterEqual(R.upper[j], y)


@unittest.skipIf(mp is None, "mpmath is optional and used only for high-precision audits")
class TestAddedTrustedElementary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mp.mp.dps = 160
        cls.backend = RigorousBinary64Backend()

    def test_point_enclosures_against_high_precision(self):
        specs = [
            (sinh_point_interval, mp.sinh, [-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0]),
            (cosh_point_interval, mp.cosh, [-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0]),
            (tanh_point_interval, mp.tanh, [-100.0, -2.0, -0.1, 0.0, 0.1, 2.0, 100.0]),
            (arctan_point_interval, mp.atan, [-1e6, -2.0, -0.5, 0.0, 0.5, 2.0, 1e6]),
            (log1p_point_interval, mp.log1p, [-0.9, -0.2, 0.0, 0.2, 1.0, 10.0]),
        ]
        for fn, exact_fn, xs in specs:
            for x in xs:
                with self.subTest(function=fn.__name__, x=x):
                    I = fn(float(x), self.backend)
                    self.assertTrue(_contains_mp(I, exact_fn(_mp_exact_float(float(x)))), (fn.__name__, x, I))

    def test_interval_extensions_contain_high_precision_samples(self):
        specs = [
            (sinh_interval, mp.sinh, TrustedInterval(-1.2, 0.7)),
            (cosh_interval, mp.cosh, TrustedInterval(-1.2, 0.7)),
            (tanh_interval, mp.tanh, TrustedInterval(-2.0, 1.5)),
            (arctan_interval, mp.atan, TrustedInterval(-4.0, 3.0)),
            (log1p_interval, mp.log1p, TrustedInterval(-0.8, 2.0)),
        ]
        rng = np.random.default_rng(20260902)
        for fn, exact_fn, X in specs:
            Y = fn(X, self.backend)
            for x in rng.uniform(X.lo, X.hi, 100):
                self.assertTrue(_contains_mp(Y, exact_fn(_mp_exact_float(float(x)))))

    def test_absolute_interval_exact_shape(self):
        self.assertEqual(absolute_interval(TrustedInterval(-2.0, 3.0)), TrustedInterval(0.0, 3.0))
        self.assertEqual(absolute_interval(TrustedInterval(2.0, 3.0)), TrustedInterval(2.0, 3.0))
        self.assertEqual(absolute_interval(TrustedInterval(-3.0, -2.0)), TrustedInterval(2.0, 3.0))


if __name__ == "__main__":
    unittest.main()
