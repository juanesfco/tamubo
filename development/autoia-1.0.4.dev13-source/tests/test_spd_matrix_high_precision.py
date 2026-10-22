import unittest
import numpy as np

from autoia import autoia, Interval, quad_solve_spd


class TestSPDMatrixHighPrecision(unittest.TestCase):
    def test_two_by_two_spd_logdet_and_quadratic_form_contain_80_digit_samples(self):
        try:
            import mpmath as mp
        except Exception:
            self.skipTest("optional mpmath not installed")
        mp.mp.dps = 80

        def model(p):
            x, y = p
            off = 0.10*x - 0.04*y
            A = np.array([
                [2.0 + 0.15*x*x + 0.05*y, off],
                [off, 1.5 + 0.08*y*y - 0.03*x],
            ], dtype=object)
            b = np.array([1.0 + 0.2*x, 0.5 - 0.1*y], dtype=object)
            ld = np.linalg.slogdet(A)[1]
            q = quad_solve_spd(A, b)
            return np.array([ld, q, 0.5*(ld+q)], dtype=object)

        box = [Interval(-0.25, 0.25), Interval(-0.20, 0.30)]
        result = autoia(model, box)
        self.assertTrue(result.machine_rigorous)

        for x0 in np.linspace(box[0].lo, box[0].hi, 9):
            for y0 in np.linspace(box[1].lo, box[1].hi, 9):
                x = mp.mpf(str(float(x0))); y = mp.mpf(str(float(y0)))
                off = mp.mpf('0.10')*x - mp.mpf('0.04')*y
                A = mp.matrix([
                    [mp.mpf('2.0') + mp.mpf('0.15')*x*x + mp.mpf('0.05')*y, off],
                    [off, mp.mpf('1.5') + mp.mpf('0.08')*y*y - mp.mpf('0.03')*x],
                ])
                b = mp.matrix([mp.mpf('1.0') + mp.mpf('0.2')*x,
                               mp.mpf('0.5') - mp.mpf('0.1')*y])
                ld = mp.log(mp.det(A))
                q = (b.T * (A**-1) * b)[0]
                vals = [ld, q, mp.mpf('0.5')*(ld+q)]
                for i, value in enumerate(vals):
                    self.assertLessEqual(mp.mpf(str(result.lower[i])), value)
                    self.assertGreaterEqual(mp.mpf(str(result.upper[i])), value)

    def test_rbf_like_spd_family_contains_high_precision_samples(self):
        try:
            import mpmath as mp
        except Exception:
            self.skipTest("optional mpmath not installed")
        mp.mp.dps = 80
        X = [mp.mpf('0.0'), mp.mpf('0.45'), mp.mpf('1.0')]
        y_np = np.array([0.2, -0.3, 0.5])

        def model(p):
            logell, loglam = p
            ell = np.exp(logell); lam = np.exp(loglam)
            Xn = np.array([0.0, 0.45, 1.0])
            K = np.empty((3,3), dtype=object)
            for i in range(3):
                for j in range(3):
                    r = (Xn[i]-Xn[j])/ell
                    K[i,j] = np.exp(-0.5*r*r)
                K[i,i] = K[i,i] + lam
            ld = np.linalg.slogdet(K)[1]
            q = quad_solve_spd(K, y_np)
            return np.array([ld, q], dtype=object)

        box = [Interval(np.log(0.28), np.log(0.38)),
               Interval(np.log(0.05), np.log(0.10))]
        result = autoia(model, box)
        self.assertTrue(result.machine_rigorous)

        y = mp.matrix([mp.mpf('0.2'), mp.mpf('-0.3'), mp.mpf('0.5')])
        for le0 in np.linspace(box[0].lo, box[0].hi, 7):
            for ln0 in np.linspace(box[1].lo, box[1].hi, 7):
                ell = mp.e**mp.mpf(str(float(le0)))
                lam = mp.e**mp.mpf(str(float(ln0)))
                K = mp.matrix(3, 3)
                for i in range(3):
                    for j in range(3):
                        r = (X[i]-X[j])/ell
                        K[i,j] = mp.e**(-mp.mpf('0.5')*r*r)
                    K[i,i] += lam
                vals = [mp.log(mp.det(K)), (y.T*(K**-1)*y)[0]]
                for i, value in enumerate(vals):
                    self.assertLessEqual(mp.mpf(str(result.lower[i])), value)
                    self.assertGreaterEqual(mp.mpf(str(result.upper[i])), value)

    def test_spd_enclosure_shrinks_with_physical_box(self):
        def model(p):
            x, y = p
            off = 0.08*x + 0.03*y
            A = np.array([[2.0+0.1*x, off], [off, 1.4+0.08*y]], dtype=object)
            b = np.array([1.0+0.1*x, 0.4-0.1*y], dtype=object)
            return np.array([np.linalg.slogdet(A)[1], quad_solve_spd(A,b)], dtype=object)

        center = np.array([0.05, -0.02])
        halfwidths = [0.20, 0.10, 0.05, 0.025]
        widths = []
        for h in halfwidths:
            box = [Interval(center[0]-h, center[0]+h),
                   Interval(center[1]-h, center[1]+h)]
            r = autoia(model, box, order=4, representation="polynomial")
            widths.append(np.asarray(r.upper-r.lower, dtype=float))
        widths = np.asarray(widths)
        self.assertTrue(np.all(widths[1:] <= widths[:-1] + 1e-14), widths)
        self.assertTrue(np.all(widths[-1] < 0.35*widths[0]), widths)


if __name__ == "__main__":
    unittest.main()
