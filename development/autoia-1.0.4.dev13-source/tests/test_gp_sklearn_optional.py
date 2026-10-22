"""Live optional scikit-learn exact-GP integration tests.

These tests execute when scikit-learn is installed and otherwise skip cleanly.
They exercise ordinary fit/predict code, not the expert freeze API alone.
"""
import math
import unittest
import numpy as np

from autoia import autoia, Interval
from autoia.integrations import (
    freeze_sklearn_gaussian_process_regressor,
    frozen_gp_posterior,
)


class TestLiveOptionalSklearnGPIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
        except ImportError:
            raise unittest.SkipTest("optional package scikit-learn is not installed")
        cls.GPR = GaussianProcessRegressor
        cls.ConstantKernel = ConstantKernel
        cls.RBF = RBF
        cls.Matern = Matern
        cls.WhiteKernel = WhiteKernel

    def _training_data(self):
        X = np.linspace(0.0, 1.0, 12).reshape(-1, 1)
        y = np.sin(2.0 * math.pi * X[:, 0])
        return X, y

    def test_fit_inside_engineering_model_mean_and_std(self):
        GaussianProcessRegressor = self.GPR
        ConstantKernel = self.ConstantKernel
        RBF = self.RBF

        def engineering_model(p):
            X = np.linspace(0.0, 1.0, 12).reshape(-1, 1)
            y = np.sin(2.0 * math.pi * X[:, 0])
            gp = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, constant_value_bounds="fixed")
                * RBF(0.25, length_scale_bounds="fixed"),
                alpha=1.0e-4,
                normalize_y=True,
                optimizer=None,
            )
            gp.fit(X, y)
            x = np.array([[p[0]]])
            mean, std = gp.predict(x, return_std=True)
            return mean[0], std[0]

        result = autoia(
            engineering_model,
            [Interval.enclosing("0.20", "0.30")],
        )
        self.assertTrue(result.machine_rigorous)

        # Dense sampling is not proof; it is a regression-quality check that the
        # rigorous result contains ordinary sklearn evaluations.
        X, y = self._training_data()
        gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, constant_value_bounds="fixed")
            * RBF(0.25, length_scale_bounds="fixed"),
            alpha=1.0e-4,
            normalize_y=True,
            optimizer=None,
        ).fit(X, y)
        q = np.linspace(0.20, 0.30, 101).reshape(-1, 1)
        m, s = gp.predict(q, return_std=True)
        self.assertLessEqual(float(result.lower[0]), float(np.min(m)))
        self.assertGreaterEqual(float(result.upper[0]), float(np.max(m)))
        self.assertLessEqual(float(result.lower[1]), float(np.min(s)))
        self.assertGreaterEqual(float(result.upper[1]), float(np.max(s)))

    def test_captured_fitted_model_mean(self):
        X, y = self._training_data()
        gp = self.GPR(
            kernel=self.RBF(0.3, length_scale_bounds="fixed"),
            alpha=2.0e-4,
            normalize_y=False,
            optimizer=None,
        ).fit(X, y)

        def engineering_model(p):
            x = np.array([[p[0]]])
            return gp.predict(x)[0]

        result = autoia(engineering_model, [Interval.enclosing("0.20", "0.30")])
        self.assertTrue(result.machine_rigorous)
        q = np.linspace(0.20, 0.30, 101).reshape(-1, 1)
        m = gp.predict(q)
        self.assertLessEqual(float(result.lower[0]), float(np.min(m)))
        self.assertGreaterEqual(float(result.upper[0]), float(np.max(m)))

    def test_frozen_state_matches_sklearn_pointwise_all_supported_profiles(self):
        X, y = self._training_data()
        profiles = [
            self.RBF(0.28, length_scale_bounds="fixed"),
            self.Matern(0.28, nu=0.5, length_scale_bounds="fixed"),
            self.Matern(0.28, nu=1.5, length_scale_bounds="fixed"),
            self.Matern(0.28, nu=2.5, length_scale_bounds="fixed"),
        ]
        q = np.linspace(0.05, 0.95, 41).reshape(-1, 1)
        for base in profiles:
            kernel = self.ConstantKernel(1.7, constant_value_bounds="fixed") * base
            gp = self.GPR(
                kernel=kernel,
                alpha=3.0e-4,
                normalize_y=True,
                optimizer=None,
            ).fit(X, y)
            state = freeze_sklearn_gaussian_process_regressor(gp)
            m_ref, s_ref = gp.predict(q, return_std=True)
            v_ref = s_ref * s_ref
            m, v = frozen_gp_posterior(q, state, outcome_space=True)
            m = np.asarray(m, dtype=float)
            v = np.asarray(v, dtype=float)
            self.assertLess(float(np.max(np.abs(m - m_ref))), 2.0e-12)
            self.assertLess(float(np.max(np.abs(v - v_ref))), 2.0e-12)

    def test_unsupported_white_kernel_fails_closed(self):
        X, y = self._training_data()
        gp = self.GPR(
            kernel=self.RBF(0.3, length_scale_bounds="fixed")
            + self.WhiteKernel(1.0e-3, noise_level_bounds="fixed"),
            alpha=1.0e-6,
            optimizer=None,
        ).fit(X, y)
        with self.assertRaises(NotImplementedError):
            freeze_sklearn_gaussian_process_regressor(gp)

    def test_return_cov_supported(self):
        GaussianProcessRegressor = self.GPR
        RBF = self.RBF

        def engineering_model(p):
            X = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
            y = np.sin(2.0 * math.pi * X[:, 0])
            gp = GaussianProcessRegressor(
                kernel=RBF(0.25, length_scale_bounds="fixed"),
                alpha=1.0e-4,
                optimizer=None,
            ).fit(X, y)
            x = np.array([[p[0]]])
            mean, cov = gp.predict(x, return_cov=True)
            return mean[0], cov[0, 0]

        result = autoia(
            engineering_model, [Interval.enclosing("0.20", "0.30")]
        )
        self.assertTrue(result.machine_rigorous)

        X = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
        y = np.sin(2.0 * math.pi * X[:, 0])
        gp = GaussianProcessRegressor(
            kernel=RBF(0.25, length_scale_bounds="fixed"),
            alpha=1.0e-4,
            optimizer=None,
        ).fit(X, y)
        q = np.linspace(0.20, 0.30, 101).reshape(-1, 1)
        means = []
        vars_ = []
        for qi in q:
            m, c = gp.predict(qi.reshape(1, 1), return_cov=True)
            means.append(float(m[0]))
            vars_.append(float(c[0, 0]))
        self.assertLessEqual(float(result.lower[0]), min(means))
        self.assertGreaterEqual(float(result.upper[0]), max(means))
        self.assertLessEqual(float(result.lower[1]), min(vars_))
        self.assertGreaterEqual(float(result.upper[1]), max(vars_))


if __name__ == "__main__":
    unittest.main()
