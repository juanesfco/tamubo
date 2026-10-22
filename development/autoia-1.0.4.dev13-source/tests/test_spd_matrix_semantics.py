import unittest
import numpy as np

from autoia import (
    autoia, Interval, quad_solve_spd, logdet_spd,
    CertificationContext, CertificationError,
)


class TestSPDMatrixSemantics(unittest.TestCase):
    def test_slogdet_quad_solve_and_det_contain_samples(self):
        def model(p):
            x = p[0]
            A = np.array([
                [2.0 + 0.2*x, 0.15*x],
                [0.15*x, 1.5 - 0.1*x],
            ], dtype=object)
            sign, ld = np.linalg.slogdet(A)
            det = np.linalg.det(A)
            b = np.array([1.0 + x, 0.5 - 0.2*x], dtype=object)
            q = quad_solve_spd(A, b)
            return np.array([sign, ld, det, q, ld + q], dtype=object)

        result = autoia(model, [Interval(-0.2, 0.2)])
        self.assertTrue(result.machine_rigorous)
        for x in np.linspace(-0.2, 0.2, 81):
            A = np.array([[2.0 + 0.2*x, 0.15*x], [0.15*x, 1.5 - 0.1*x]])
            b = np.array([1.0 + x, 0.5 - 0.2*x])
            sign, ld = np.linalg.slogdet(A)
            det = np.linalg.det(A)
            q = float(b @ np.linalg.solve(A, b))
            vals = [sign, ld, det, q, ld + q]
            for i, value in enumerate(vals):
                self.assertLessEqual(result.lower[i], value)
                self.assertGreaterEqual(result.upper[i], value)

    def test_separately_constructed_symmetric_entries_are_recognized(self):
        def model(p):
            x = p[0]
            A = np.array([
                [2.0 + x*x, 0.1*x],
                [0.1*x, 1.4 + 0.5*x*x],
            ], dtype=object)
            return np.linalg.slogdet(A)[1]

        result = autoia(model, [Interval(-0.1, 0.1)])
        self.assertTrue(result.machine_rigorous)

    def test_non_symmetric_family_fails_closed(self):
        def model(p):
            x = p[0]
            A = np.array([[2.0, 0.1*x], [0.2*x, 1.5]], dtype=object)
            return np.linalg.slogdet(A)[1]

        with self.assertRaises(CertificationError):
            autoia(model, [Interval(-0.1, 0.1)])

    def test_application_context_spd_logdet_quad(self):
        cert = CertificationContext()
        a = cert.interval(1.96, 2.04)
        d = cert.interval(1.48, 1.52)
        off = cert.interval(-0.03, 0.03)
        A = np.empty((2, 2), dtype=object)
        A[0, 0] = a
        A[1, 1] = d
        A[0, 1] = off
        A[1, 0] = off
        proof = cert.spd(A)
        self.assertGreater(proof.eigenvalue_lower, 0.0)
        ld = cert.logdet(A)
        q = cert.quad_solve(A, [cert.interval(0.8, 1.2), cert.interval(0.46, 0.54)])
        self.assertLess(ld.lower, ld.upper)
        self.assertGreater(q.lower, 0.0)
        self.assertTrue(cert.machine_rigorous)

    def test_mle_like_family_certifies_on_moderate_box(self):
        X = np.array([0.0, 0.45, 1.0])
        y = np.array([0.2, -0.3, 0.5])

        def model(p):
            logell, loglam = p
            ell = np.exp(logell)
            lam = np.exp(loglam)
            K = np.empty((3, 3), dtype=object)
            for i in range(3):
                for j in range(3):
                    r = (X[i] - X[j]) / ell
                    K[i, j] = np.exp(-0.5*r*r)
                K[i, i] = K[i, i] + lam
            ld = np.linalg.slogdet(K)[1]
            q = quad_solve_spd(K, y)
            return 0.5*(ld + q)

        box = [Interval(np.log(0.28), np.log(0.38)), Interval(np.log(0.05), np.log(0.10))]
        result = autoia(model, box)
        self.assertTrue(result.machine_rigorous)
        for le in np.linspace(box[0].lo, box[0].hi, 9):
            for ln in np.linspace(box[1].lo, box[1].hi, 9):
                ell = np.exp(le); lam = np.exp(ln)
                D = (X[:, None] - X[None, :]) / ell
                K = np.exp(-0.5*D*D) + lam*np.eye(3)
                value = 0.5*(np.linalg.slogdet(K)[1] + y @ np.linalg.solve(K, y))
                self.assertLessEqual(result.lower[0], value)
                self.assertGreaterEqual(result.upper[0], value)

    def test_quad_solve_spd_does_not_require_general_solve_certificate(self):
        from unittest.mock import patch

        def model(p):
            x = p[0]
            A = np.array([
                [2.0 + 0.1*x, 0.12*x],
                [0.12*x, 1.6 - 0.05*x],
            ], dtype=object)
            b = np.array([1.0 + 0.2*x, 0.4 - 0.1*x], dtype=object)
            return quad_solve_spd(A, b)

        # The structured SPD inverse-series theorem is independent of the
        # older generic linear-solve contraction theorem.  Force that optional
        # tightening route to fail and confirm the SPD result still certifies.
        with patch(
            "autoia.semantic.spd_matrix.GenericSolveContext.traced_solve",
            side_effect=RuntimeError("forced optional solve-route failure"),
        ):
            result = autoia(model, [Interval(-0.2, 0.2)])
        self.assertTrue(result.machine_rigorous)
        for x in np.linspace(-0.2, 0.2, 41):
            A = np.array([[2.0 + 0.1*x, 0.12*x], [0.12*x, 1.6 - 0.05*x]])
            b = np.array([1.0 + 0.2*x, 0.4 - 0.1*x])
            q = float(b @ np.linalg.solve(A, b))
            self.assertLessEqual(result.lower[0], q)
            self.assertGreaterEqual(result.upper[0], q)


if __name__ == "__main__":
    unittest.main()
