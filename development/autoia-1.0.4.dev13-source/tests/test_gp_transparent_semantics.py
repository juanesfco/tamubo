import unittest
import numpy as np

from autoia import autoia, Interval
from autoia.exceptions import UnsupportedOperationError
from autoia.integrations import freeze_gpytorch_exact_gp


ZeroMean = type("ZeroMean", (), {})
ConstantMean = type("ConstantMean", (), {})
RBFKernel = type("RBFKernel", (), {})
MaternKernel = type("MaternKernel", (), {})
ScaleKernel = type("ScaleKernel", (), {})
GaussianLikelihood = type("GaussianLikelihood", (), {})


class _FakeTorch:
    __name__ = "torch"
    def tensor(self, data, *args, **kwargs):
        return np.asarray(data, dtype=float)
    def stack(self, xs, dim=0, *args, **kwargs):
        return np.stack(xs, axis=dim)


class _StructuralExactGP:
    __autoia_supported_exact_gp__ = True
    def __init__(self, kernel="rbf"):
        self.train_inputs = (np.array([[0.0], [0.5], [1.0]]),)
        self.train_targets = np.array([0.0, 0.8, 0.2])
        self.mean_module = ZeroMean()
        if kernel == "rbf":
            k = RBFKernel(); k.lengthscale = np.array([0.35])
        elif kernel == "matern32":
            k = MaternKernel(); k.lengthscale = np.array([0.35]); k.nu = 1.5
        else:
            k = type("UnsupportedKernel", (), {})(); k.lengthscale = np.array([0.35])
        self.covar_module = k
        like = GaussianLikelihood(); like.noise = np.array([1e-3])
        self.likelihood = like
    def __call__(self, x):
        raise AssertionError("validated execution should be intercepted")




_FakeExactGPBase = type("ExactGP", (), {})
_FakeExactGPBase.__module__ = "gpytorch.models.exact_gp"


class _UserDefinedExactGP(_FakeExactGPBase):
    """Mimic the class identity pattern of an ordinary user ExactGP subclass."""
    def __init__(self):
        self.training = False
        self.train_inputs = (np.array([[0.0], [0.5], [1.0]]),)
        self.train_targets = np.array([0.0, 0.8, 0.2])
        self.mean_module = ZeroMean()
        k = RBFKernel(); k.lengthscale = np.array([0.35])
        self.covar_module = k
        like = GaussianLikelihood(); like.noise = np.array([1e-3])
        self.likelihood = like
    def __call__(self, x):
        raise AssertionError("user-defined ExactGP subclass should be intercepted")




class _FakeExactGPConstructibleBase:
    def __init__(self, x, y, likelihood):
        self.training = False
        self.train_inputs = (np.asarray(x, dtype=float),)
        self.train_targets = np.asarray(y, dtype=float)
        self.likelihood = likelihood
    def __call__(self, x):
        raise AssertionError("locally constructed ExactGP should be intercepted")

_FakeExactGPConstructibleBase.__name__ = "ExactGP"
_FakeExactGPConstructibleBase.__module__ = "gpytorch.models.exact_gp"


class _FakeGPModelsNamespace:
    ExactGP = _FakeExactGPConstructibleBase


class _FakeGPyTorchModule:
    __name__ = "gpytorch"
    models = _FakeGPModelsNamespace()


class TestTransparentGPSemantics(unittest.TestCase):
    def test_model_defined_and_constructed_inside_engineering_function(self):
        global gpytorch, torch
        gpytorch = _FakeGPyTorchModule()
        torch = _FakeTorch()

        def engineering_model(p):
            train_x = np.array([[0.0], [0.5], [1.0]])
            train_y = np.array([0.0, 0.8, 0.2])
            like = GaussianLikelihood(); like.noise = np.array([1e-3])

            class ExactGPModel(gpytorch.models.ExactGP):
                def __init__(self, x, y, likelihood):
                    super().__init__(x, y, likelihood)
                    self.mean_module = ZeroMean()
                    k = RBFKernel(); k.lengthscale = np.array([0.35])
                    self.covar_module = k

            model = ExactGPModel(train_x, train_y, like)
            model.training = False
            x = torch.tensor([[p[0]]])
            return model(x).mean[0]

        result = autoia(engineering_model, [Interval(0.2, 0.3)])
        self.assertTrue(result.machine_rigorous)
        self.assertLess(result.lower[0], result.upper[0])

    def test_user_defined_exactgp_subclass_is_recognized_by_mro(self):
        global model, torch
        model = _UserDefinedExactGP()
        torch = _FakeTorch()
        def engineering_model(p):
            x = torch.tensor([[p[0]]])
            return model(x).mean[0]
        result = autoia(engineering_model, [Interval(0.2, 0.3)])
        self.assertTrue(result.machine_rigorous)

    def test_global_model_and_torch_tensor_are_transparent(self):
        global model, torch
        model = _StructuralExactGP("rbf")
        torch = _FakeTorch()
        def engineering_model(p):
            x = torch.tensor([[p[0]]])
            posterior = model(x)
            return posterior.mean[0]
        result = autoia(engineering_model, [Interval(0.2, 0.3)])
        self.assertTrue(result.machine_rigorous)
        self.assertLess(result.lower[0], result.upper[0])

    def test_closure_model_is_transparent(self):
        torch = _FakeTorch()
        fitted = _StructuralExactGP("matern32")
        def engineering_model(p):
            x = torch.stack([p[0]]).unsqueeze(0)
            return fitted(x).variance[0]
        result = autoia(engineering_model, [Interval(0.2, 0.3)], representation="interval")
        self.assertTrue(result.machine_rigorous)
        self.assertLessEqual(result.lower[0], result.upper[0])

    def test_expert_freeze_api_remains_available(self):
        state = freeze_gpytorch_exact_gp(_StructuralExactGP("rbf"))
        self.assertEqual(state.kernel_kind, "RBFKernel")
        self.assertEqual(state.dimension, 1)

    def test_unsupported_kernel_fails_closed(self):
        global model, torch
        model = _StructuralExactGP("unsupported")
        torch = _FakeTorch()
        def engineering_model(p):
            return model(torch.tensor([[p[0]]])).mean[0]
        with self.assertRaises(UnsupportedOperationError):
            autoia(engineering_model, [Interval(0.2, 0.3)])


if __name__ == "__main__":
    unittest.main()

# ---------------------------------------------------------------------------
# Development regressions for independent multi-output BoTorch posterior and
# validated tensor algebra used by analytic third-party acquisitions.
# ---------------------------------------------------------------------------

_FakeSingleTaskGPBase = type("SingleTaskGP", (), {})
_FakeSingleTaskGPBase.__module__ = "botorch.models.gp_regression"

Standardize = type("Standardize", (), {})


class _FakeMultiOutputSingleTaskGP(_FakeSingleTaskGPBase):
    def __init__(self):
        self.training = False
        self.num_outputs = 2
        self.batch_shape = ()
        X = np.array([[0.0], [0.5], [1.0]], dtype=float)
        self.train_inputs = (np.stack([X, X], axis=0),)
        self.train_targets = np.array([
            [0.0, 0.8, 0.2],
            [0.9, 0.4, 0.1],
        ], dtype=float)
        self.mean_module = ConstantMean()
        self.mean_module.constant = np.array([0.0, 0.0])
        k = RBFKernel()
        k.lengthscale = np.array([[[0.35]], [[0.55]]], dtype=float)
        self.covar_module = k
        like = GaussianLikelihood()
        like.noise = np.array([[1e-3], [2e-3]], dtype=float)
        self.likelihood = like
        ot = Standardize()
        ot.means = np.array([[0.2, -0.1]], dtype=float)
        ot.stdvs = np.array([[1.2, 0.8]], dtype=float)
        self.outcome_transform = ot

    def posterior(self, x, *args, **kwargs):
        raise AssertionError("validated multi-output posterior should be intercepted")


def _test_multioutput_posterior_none_keyword_and_shape(self):
    global model, torch
    model = _FakeMultiOutputSingleTaskGP()
    torch = _FakeTorch()

    def engineering_model(p):
        x = torch.tensor([[[p[0]]]])
        posterior = model.posterior(x, posterior_transform=None)
        return posterior.mean[0, 0, 0] + posterior.mean[0, 0, 1]

    result = autoia(engineering_model, [Interval(0.2, 0.3)])
    self.assertTrue(result.machine_rigorous)
    self.assertLess(result.lower[0], result.upper[0])


# Attach without disturbing the historical class body above.
TestTransparentGPSemantics.test_multioutput_posterior_none_keyword_and_shape = \
    _test_multioutput_posterior_none_keyword_and_shape
