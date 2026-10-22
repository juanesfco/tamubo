import math
import unittest

import numpy as np

from autoia import Interval, autoia, UnsupportedOperationError

try:
    import scipy
    import scipy.stats as scipy_stats
    import scipy.special as scipy_special
    from scipy import stats as scipy_stats_from_scipy
    from scipy import special as scipy_special_from_scipy
    from scipy.stats import norm as scipy_norm
    from scipy.special import erf as scipy_erf
    from scipy.special import ndtr as scipy_ndtr
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    SCIPY_AVAILABLE = False

if SCIPY_AVAILABLE:
    SCIPY_PDF_ALIAS = scipy_norm.pdf
    SCIPY_CDF_ALIAS = scipy_norm.cdf
    SCIPY_FROZEN = scipy_norm(loc=0.4, scale=1.3)
    SCIPY_FROZEN_PDF_ALIAS = SCIPY_FROZEN.pdf
    SCIPY_FROZEN_CDF_ALIAS = SCIPY_FROZEN.cdf
    SCIPY_NORMAL_CLASS = getattr(scipy_stats, "Normal", None)
    SCIPY_NORMAL_INSTANCE = (
        SCIPY_NORMAL_CLASS(mu=-0.25, sigma=1.4)
        if SCIPY_NORMAL_CLASS is not None
        else None
    )


def _dense_truth(box, func, n=801):
    xs = np.linspace(box[0], box[1], n)
    vals = [float(func(float(x))) for x in xs]
    return min(vals), max(vals)


@unittest.skipUnless(SCIPY_AVAILABLE, "SciPy is optional and not installed")
class TestScipyNormalCompatibility(unittest.TestCase):
    def _assert_contains(self, result, index, lo, hi):
        self.assertLessEqual(float(result.lower[index]), float(lo))
        self.assertGreaterEqual(float(result.upper[index]), float(hi))
        self.assertTrue(result.machine_rigorous)

    def test_top_level_scipy_stats_and_special_access(self):
        def model(p):
            x = p[0]
            return np.array([
                scipy.stats.norm.pdf(x),
                scipy.stats.norm.cdf(x),
                scipy.special.erf(x),
                scipy.special.ndtr(x),
            ])

        box = (-1.2, 0.9)
        result = autoia(model, [Interval(*box)], order=8)
        for i, truth in enumerate([
            scipy_stats.norm.pdf,
            scipy_stats.norm.cdf,
            scipy_special.erf,
            scipy_special.ndtr,
        ]):
            lo, hi = _dense_truth(box, truth)
            self._assert_contains(result, i, lo, hi)

    def test_imported_stats_and_special_modules(self):
        def model(p):
            x = p[0]
            return np.array([
                scipy_stats.norm.pdf(x, loc=0.5, scale=1.7),
                scipy_stats_from_scipy.norm.cdf(x, loc=-0.2, scale=0.8),
                scipy_special.erf(x),
                scipy_special_from_scipy.ndtr(x),
            ])

        box = (-0.7, 1.1)
        result = autoia(model, [Interval(*box)], order=8)
        truths = [
            lambda x: scipy_stats.norm.pdf(x, loc=0.5, scale=1.7),
            lambda x: scipy_stats.norm.cdf(x, loc=-0.2, scale=0.8),
            scipy_special.erf,
            scipy_special.ndtr,
        ]
        for i, truth in enumerate(truths):
            lo, hi = _dense_truth(box, truth)
            self._assert_contains(result, i, lo, hi)

    def test_direct_import_aliases_and_bound_method_aliases(self):
        def model(p):
            x = p[0]
            return np.array([
                scipy_erf(x),
                scipy_ndtr(x),
                SCIPY_PDF_ALIAS(x, 0.3, 1.2),
                SCIPY_CDF_ALIAS(x, loc=-0.4, scale=1.1),
            ])

        box = (-1.0, 0.6)
        result = autoia(model, [Interval(*box)], order=8)
        truths = [
            scipy_special.erf,
            scipy_special.ndtr,
            lambda x: scipy_stats.norm.pdf(x, 0.3, 1.2),
            lambda x: scipy_stats.norm.cdf(x, loc=-0.4, scale=1.1),
        ]
        for i, truth in enumerate(truths):
            lo, hi = _dense_truth(box, truth)
            self._assert_contains(result, i, lo, hi)

    def test_frozen_norm_object_and_method_aliases(self):
        def model(p):
            x = p[0]
            return np.array([
                SCIPY_FROZEN.pdf(x),
                SCIPY_FROZEN.cdf(x),
                SCIPY_FROZEN_PDF_ALIAS(x),
                SCIPY_FROZEN_CDF_ALIAS(x),
            ])

        box = (-0.8, 1.0)
        result = autoia(model, [Interval(*box)], order=8)
        truths = [SCIPY_FROZEN.pdf, SCIPY_FROZEN.cdf] * 2
        for i, truth in enumerate(truths):
            lo, hi = _dense_truth(box, truth)
            self._assert_contains(result, i, lo, hi)

    def test_frozen_norm_created_inside_model_through_global_norm_proxy(self):
        def model(p):
            x = p[0]
            rv = scipy_norm(loc=0.2, scale=0.9)
            return np.array([rv.pdf(x), rv.cdf(x)])

        box = (-0.5, 0.7)
        result = autoia(model, [Interval(*box)], order=8)
        rv = scipy_stats.norm(loc=0.2, scale=0.9)
        for i, truth in enumerate([rv.pdf, rv.cdf]):
            lo, hi = _dense_truth(box, truth)
            self._assert_contains(result, i, lo, hi)

    @unittest.skipUnless(
        SCIPY_AVAILABLE and SCIPY_NORMAL_CLASS is not None,
        "scipy.stats.Normal is unavailable in this SciPy version",
    )
    def test_new_normal_distribution_object_api(self):
        def model(p):
            x = p[0]
            X = SCIPY_NORMAL_CLASS(mu=0.3, sigma=1.6)
            return np.array([X.pdf(x), X.cdf(x)])

        box = (-0.9, 1.2)
        result = autoia(model, [Interval(*box)], order=8)
        X = SCIPY_NORMAL_CLASS(mu=0.3, sigma=1.6)
        for i, truth in enumerate([X.pdf, X.cdf]):
            lo, hi = _dense_truth(box, truth)
            self._assert_contains(result, i, lo, hi)

    @unittest.skipUnless(
        SCIPY_AVAILABLE and SCIPY_NORMAL_INSTANCE is not None,
        "scipy.stats.Normal is unavailable in this SciPy version",
    )
    def test_preexisting_new_normal_instance_and_two_argument_cdf(self):
        def model(p):
            x = p[0]
            return np.array([
                SCIPY_NORMAL_INSTANCE.pdf(x),
                SCIPY_NORMAL_INSTANCE.cdf(x),
                SCIPY_NORMAL_INSTANCE.cdf(x, x + 0.5),
            ])

        box = (-0.6, 0.6)
        result = autoia(model, [Interval(*box)], order=8)
        X = SCIPY_NORMAL_INSTANCE
        truths = [
            X.pdf,
            X.cdf,
            lambda x: X.cdf(x, x + 0.5),
        ]
        for i, truth in enumerate(truths):
            lo, hi = _dense_truth(box, truth)
            self._assert_contains(result, i, lo, hi)

    def test_vectorized_scipy_special_and_norm_calls(self):
        def model(p):
            x = p[0]
            v = np.array([x, x + 0.25])
            return np.concatenate([
                scipy_special.ndtr(v),
                scipy_stats.norm.pdf(v, loc=0.1, scale=1.2),
            ])

        box = (-0.4, 0.5)
        result = autoia(model, [Interval(*box)], order=8)
        self.assertEqual(result.output_shape, (4,))
        samples = np.linspace(*box, 601)
        vals = np.array([
            np.concatenate([
                np.asarray(scipy_special.ndtr([x, x + 0.25]), dtype=float),
                np.asarray(scipy_stats.norm.pdf([x, x + 0.25], loc=0.1, scale=1.2), dtype=float),
            ])
            for x in samples
        ])
        for j in range(4):
            self._assert_contains(result, j, vals[:, j].min(), vals[:, j].max())

    def test_real_scipy_objects_are_never_mutated(self):
        original_erf = scipy_special.erf
        original_ndtr = scipy_special.ndtr
        original_norm = scipy_stats.norm
        original_pdf_func = scipy_stats.norm.pdf.__func__
        original_cdf_func = scipy_stats.norm.cdf.__func__

        def model(
            p,
            real_special=scipy_special,
            real_stats=scipy_stats,
            real_erf=original_erf,
            real_ndtr=original_ndtr,
            real_norm=original_norm,
            real_pdf_func=original_pdf_func,
            real_cdf_func=original_cdf_func,
        ):
            if real_special.erf is not real_erf:
                raise RuntimeError("real scipy.special.erf was mutated")
            if real_special.ndtr is not real_ndtr:
                raise RuntimeError("real scipy.special.ndtr was mutated")
            if real_stats.norm is not real_norm:
                raise RuntimeError("real scipy.stats.norm was mutated")
            if real_stats.norm.pdf.__func__ is not real_pdf_func:
                raise RuntimeError("real scipy.stats.norm.pdf was mutated")
            if real_stats.norm.cdf.__func__ is not real_cdf_func:
                raise RuntimeError("real scipy.stats.norm.cdf was mutated")
            x = p[0]
            return np.array([scipy_special.ndtr(x), scipy_stats.norm.pdf(x)])

        result = autoia(model, [Interval(-0.3, 0.4)], order=8)
        self.assertTrue(result.machine_rigorous)
        self.assertIs(scipy_special.erf, original_erf)
        self.assertIs(scipy_special.ndtr, original_ndtr)
        self.assertIs(scipy_stats.norm, original_norm)
        self.assertIs(scipy_stats.norm.pdf.__func__, original_pdf_func)
        self.assertIs(scipy_stats.norm.cdf.__func__, original_cdf_func)


    def test_scipy_entry_points_match_direct_autoia_kernels_bitwise(self):
        from autoia import erf, standard_normal_pdf, standard_normal_cdf, normal_pdf, normal_cdf

        def scipy_model(p):
            x = p[0]
            return np.array([
                scipy_special.erf(x),
                scipy_special.ndtr(x),
                scipy_stats.norm.pdf(x),
                scipy_stats.norm.cdf(x),
                scipy_stats.norm.pdf(x, loc=0.3, scale=1.4),
                scipy_stats.norm.cdf(x, loc=-0.2, scale=0.7),
            ])

        def direct_model(p):
            x = p[0]
            return np.array([
                erf(x),
                standard_normal_cdf(x),
                standard_normal_pdf(x),
                standard_normal_cdf(x),
                normal_pdf(x, mean=0.3, std=1.4),
                normal_cdf(x, mean=-0.2, std=0.7),
            ])

        box = [Interval(-0.8, 0.9)]
        a = autoia(scipy_model, box, order=8)
        b = autoia(direct_model, box, order=8)
        self.assertTrue(np.array_equal(a.lower.view(np.uint64), b.lower.view(np.uint64)))
        self.assertTrue(np.array_equal(a.upper.view(np.uint64), b.upper.view(np.uint64)))

    def test_uncertain_distribution_parameter_fails_typed(self):
        def model(p):
            x, mu = p
            return scipy_stats.norm.cdf(x, loc=mu, scale=1.0)

        with self.assertRaises(UnsupportedOperationError):
            autoia(model, [Interval(-1.0, 1.0), Interval(-0.2, 0.2)], order=6)


if __name__ == "__main__":
    unittest.main()
