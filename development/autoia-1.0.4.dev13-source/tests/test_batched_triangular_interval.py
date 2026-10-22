import unittest
from fractions import Fraction

import numpy as np

from autoia.trusted import (
    Interval,
    RigorousBinary64Backend,
    point_interval,
    rigorous_triangular_solve_point_interval_many,
)


def scalar_reference(L, lo, hi, backend):
    L = np.asarray(L, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    out_lo = np.empty_like(lo)
    out_hi = np.empty_like(hi)
    for b in range(lo.shape[0]):
        vals = []
        for j in range(L.shape[0]):
            numer = Interval(float(lo[b, j]), float(hi[b, j]))
            for i in range(j):
                numer = backend.sub(
                    numer,
                    backend.mul(point_interval(float(L[j, i])), vals[i]),
                )
            vals.append(backend.div(numer, point_interval(float(L[j, j]))))
        for j, value in enumerate(vals):
            out_lo[b, j] = value.lo
            out_hi[b, j] = value.hi
    return out_lo, out_hi


class TestBatchedTriangularIntervalSolve(unittest.TestCase):
    def test_random_batch_contains_exact_rational_natural_interval(self):
        rng = np.random.default_rng(81723)
        n = 6
        batch = 12
        L = np.tril(rng.uniform(-0.4, 0.4, size=(n, n)))
        diag = rng.uniform(0.7, 1.5, size=n)
        np.fill_diagonal(L, diag)
        center = rng.normal(size=(batch, n))
        radius = rng.uniform(0.0, 0.05, size=(batch, n))
        rhs_lo = center - radius
        rhs_hi = center + radius

        got_lo, got_hi = rigorous_triangular_solve_point_interval_many(
            L, rhs_lo, rhs_hi, RigorousBinary64Backend(reduction_method="auto")
        )

        def f(x): return Fraction.from_float(float(x))
        for b in range(batch):
            exact = []
            for j in range(n):
                nlo, nhi = f(rhs_lo[b, j]), f(rhs_hi[b, j])
                for i in range(j):
                    a = f(L[j, i]); xlo, xhi = exact[i]
                    if a >= 0:
                        plo, phi = a*xlo, a*xhi
                    else:
                        plo, phi = a*xhi, a*xlo
                    nlo, nhi = nlo - phi, nhi - plo
                d = f(L[j, j])
                if d > 0:
                    exact.append((nlo/d, nhi/d))
                else:
                    exact.append((nhi/d, nlo/d))
            for j, (elo, ehi) in enumerate(exact):
                self.assertLessEqual(f(got_lo[b, j]), elo)
                self.assertGreaterEqual(f(got_hi[b, j]), ehi)

    def test_arbitrary_leading_batch_dimensions(self):
        L = np.array([[2.0, 0.0, 0.0], [0.25, 1.5, 0.0], [-0.5, 0.2, 0.75]])
        lo = np.arange(24.0).reshape(2, 4, 3) / 10.0
        hi = lo + 0.01
        backend = RigorousBinary64Backend()
        out_lo, out_hi = rigorous_triangular_solve_point_interval_many(L, lo, hi, backend)
        self.assertEqual(out_lo.shape, (2, 4, 3))
        self.assertEqual(out_hi.shape, (2, 4, 3))
        flat_lo, flat_hi = rigorous_triangular_solve_point_interval_many(
            L, lo.reshape(-1, 3), hi.reshape(-1, 3), RigorousBinary64Backend()
        )
        np.testing.assert_array_equal(out_lo.reshape(-1, 3), flat_lo)
        np.testing.assert_array_equal(out_hi.reshape(-1, 3), flat_hi)

    def test_point_rhs_encloses_exact_rational_solution(self):
        # Every decimal below is a binary64 rational.  Fraction.from_float gives
        # its exact real value, so this checks the proof result independently of
        # floating arithmetic.
        L = np.array([
            [1.25, 0.0, 0.0],
            [-0.375, 0.75, 0.0],
            [0.5, -0.125, 1.5],
        ])
        rhs = np.array([
            [0.2, -0.7, 1.1],
            [-1.25, 0.4, 0.05],
            [3.0, -2.0, 0.75],
        ])
        lo, hi = rigorous_triangular_solve_point_interval_many(
            L, rhs, rhs, RigorousBinary64Backend(reduction_method="vectorized")
        )
        for b in range(rhs.shape[0]):
            exact = []
            for j in range(L.shape[0]):
                numer = Fraction.from_float(float(rhs[b, j]))
                for i in range(j):
                    numer -= Fraction.from_float(float(L[j, i])) * exact[i]
                exact.append(numer / Fraction.from_float(float(L[j, j])))
            for j, value in enumerate(exact):
                self.assertLessEqual(Fraction.from_float(float(lo[b, j])), value)
                self.assertGreaterEqual(Fraction.from_float(float(hi[b, j])), value)

    def test_rejects_invalid_matrix(self):
        backend = RigorousBinary64Backend()
        with self.assertRaises(ValueError):
            rigorous_triangular_solve_point_interval_many(
                np.array([[1.0, 1.0], [0.0, 1.0]]),
                np.zeros((2, 2)), np.zeros((2, 2)), backend,
            )
        with self.assertRaises(ZeroDivisionError):
            rigorous_triangular_solve_point_interval_many(
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                np.zeros((2, 2)), np.zeros((2, 2)), backend,
            )


if __name__ == "__main__":
    unittest.main()
