"""Transparent BoTorch SingleTaskGP posterior semantics.

Requires optional torch, gpytorch, and botorch.  The ordinary BoTorch
``model.posterior(x)`` call is used inside engineering code; no manual freeze
adapter appears in the normal workflow.
"""
import math
import torch
try:
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.fit import fit_gpytorch_mll
except ImportError:
    print("SKIP: example 28 requires optional packages gpytorch and botorch.")
    raise SystemExit(0)

from autoia import autoia, Interval

train_x = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
train_y = torch.sin(2.0 * math.pi * train_x)
model = SingleTaskGP(
    train_x, train_y,
    input_transform=Normalize(d=1),
    outcome_transform=Standardize(m=1),
)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)
model.eval()

def engineering_model(p):
    x = torch.tensor([[p[0]]], dtype=torch.double)
    posterior = model.posterior(x)
    return posterior.mean[0, 0] + 0.1 * posterior.variance[0, 0]

result = autoia(engineering_model, [Interval.enclosing("0.20", "0.30")])
print(result)
assert result.machine_rigorous
