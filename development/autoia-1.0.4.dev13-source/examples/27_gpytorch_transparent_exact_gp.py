"""Example 27: ordinary GPyTorch fit *inside* engineering code.

Requires optional torch and gpytorch.  No AutoIA freeze adapter is used.
Training/autograd are ordinary floating-point GPyTorch operations and are not
certified; AutoIA certifies the uncertain posterior evaluation conditional on
the fitted stored binary64 state.  The fitting stage must not depend on the
uncertain parameter ``p``.
"""
import math
import torch
try:
    import gpytorch
except ImportError:
    print("SKIP: example 27 requires optional package gpytorch.")
    raise SystemExit(0)

from autoia import autoia, Interval


def engineering_model(p):
    train_x = torch.linspace(0.0, 1.0, 12, dtype=torch.double)
    train_y = torch.sin(2.0 * math.pi * train_x)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, x, y, likelihood):
            super().__init__(x, y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    model = ExactGPModel(train_x, train_y, likelihood).double()
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.10)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(35):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
    model.eval(); likelihood.eval()

    x = torch.tensor([[p[0]]], dtype=torch.double)
    posterior = model(x)
    return posterior.mean[0]


result = autoia(engineering_model, [Interval.enclosing("0.20", "0.30")])
print(result)
assert result.machine_rigorous
