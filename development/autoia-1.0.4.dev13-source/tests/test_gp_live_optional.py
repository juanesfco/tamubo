import math
import unittest
import numpy as np

from autoia import autoia, Interval


class TestLiveOptionalGPIntegration(unittest.TestCase):
    def test_live_gpytorch_exact_gp_transparent(self):
        try:
            import torch
            import gpytorch
        except ImportError:
            self.skipTest("optional torch/gpytorch packages are not installed")

        train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double)
        train_y = torch.sin(2.0 * math.pi * train_x)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().double()

        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, x, y, likelihood):
                super().__init__(x, y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x), self.covar_module(x)
                )

        model = ExactGPModel(train_x, train_y, likelihood).double()
        model.train(); likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.10)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(8):
            optimizer.zero_grad(); loss = -mll(model(train_x), train_y)
            loss.backward(); optimizer.step()
        model.eval(); likelihood.eval()

        def engineering_model(p):
            x = torch.tensor([[p[0]]], dtype=torch.double)
            return model(x).mean[0]

        result = autoia(engineering_model, [Interval(0.2, 0.3)])
        self.assertTrue(result.machine_rigorous)

    def test_live_gpytorch_fit_inside_engineering_model(self):
        try:
            import torch
            import gpytorch
        except ImportError:
            self.skipTest("optional torch/gpytorch packages are not installed")

        def engineering_model(p):
            train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double)
            train_y = torch.sin(2.0 * math.pi * train_x)
            likelihood = gpytorch.likelihoods.GaussianLikelihood().double()

            class ExactGPModel(gpytorch.models.ExactGP):
                def __init__(self, x, y, likelihood):
                    super().__init__(x, y, likelihood)
                    self.mean_module = gpytorch.means.ConstantMean()
                    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                def forward(self, x):
                    return gpytorch.distributions.MultivariateNormal(
                        self.mean_module(x), self.covar_module(x)
                    )

            model = ExactGPModel(train_x, train_y, likelihood).double()
            model.train(); likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.10)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            for _ in range(4):
                optimizer.zero_grad(); loss = -mll(model(train_x), train_y)
                loss.backward(); optimizer.step()
            model.eval(); likelihood.eval()

            x = torch.tensor([[p[0]]], dtype=torch.double)
            return model(x).mean[0]

        result = autoia(engineering_model, [Interval(0.2, 0.3)])
        self.assertTrue(result.machine_rigorous)

    def test_live_botorch_single_task_gp_transparent(self):
        try:
            import torch
            from botorch.models import SingleTaskGP
            from botorch.models.transforms.input import Normalize
            from botorch.models.transforms.outcome import Standardize
            from gpytorch.mlls import ExactMarginalLogLikelihood
            from botorch.fit import fit_gpytorch_mll
        except ImportError:
            self.skipTest("optional torch/gpytorch/botorch packages are not installed")

        train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
        train_y = torch.sin(2.0 * math.pi * train_x)
        model = SingleTaskGP(
            train_x, train_y,
            input_transform=Normalize(d=1),
            outcome_transform=Standardize(m=1),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        model.eval()

        def engineering_model(p):
            x = torch.tensor([[p[0]]], dtype=torch.double)
            post = model.posterior(x)
            return post.mean[0, 0] + 0.1 * post.variance[0, 0]

        result = autoia(engineering_model, [Interval(0.2, 0.3)])
        self.assertTrue(result.machine_rigorous)


if __name__ == "__main__":
    unittest.main()
