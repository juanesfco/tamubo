"""34 - Ordinary BoTorch analytic EHVI inside an engineering function.

This example demonstrates transparent execution of a common Bayesian-
optimization workflow.  The user's function:

1. constructs a two-output BoTorch ``SingleTaskGP``;
2. fits it with BoTorch's standard ``fit_gpytorch_mll`` routine;
3. constructs the non-dominated Pareto partition;
4. constructs BoTorch's analytic ``ExpectedHypervolumeImprovement`` object;
5. evaluates EHVI at an uncertain candidate supplied by AutoIA.

No EHVI formula is implemented in AutoIA and there is no acquisition-specific
adapter in this example.  BoTorch supplies the acquisition computation.  AutoIA
provides rigorous semantics for the uncertain exact-GP posterior and for the
supported tensor / Gaussian operations reached by BoTorch's original analytic
EHVI code.

The fitting calculation is ordinary floating-point BoTorch computation.  The
AutoIA certificate is conditional on the fixed fitted binary64 model state used
when the uncertain query is evaluated.

The resulting generic EHVI enclosure on this deliberately troublesome box is
expected to be rigorous but loose.  Example 35 shows how a user-known Gaussian
cell theorem can be added without changing AutoIA core or changing this box.
"""

try:
    import torch
    import gpytorch
    import botorch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from botorch.acquisition.multi_objective.analytic import (
        ExpectedHypervolumeImprovement,
    )
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
    )
    from gpytorch.mlls import ExactMarginalLogLikelihood
except ImportError as exc:
    print("SKIP: Example 34 requires torch, gpytorch, and botorch.")
    print("Install those optional packages in the AutoIA environment and rerun.")
    raise SystemExit(0) from exc

from autoia import Interval, autoia


torch.set_default_dtype(torch.double)

TRAIN_X = torch.tensor(
    [
        [0.10, 0.10],
        [0.20, 0.80],
        [0.80, 0.20],
        [0.85, 0.85],
        [0.50, 0.50],
    ],
    dtype=torch.double,
)
TRAIN_Y = torch.tensor(
    [
        [0.25, 1.00],
        [0.40, 0.85],
        [1.00, 0.35],
        [0.75, 0.65],
        [0.65, 0.75],
    ],
    dtype=torch.double,
)
REF_POINT = [0.0, 0.0]


def engineering_model(p):
    """Normal BoTorch code; only the final candidate depends on uncertain p."""
    model = SingleTaskGP(
        TRAIN_X,
        TRAIN_Y,
        input_transform=Normalize(d=2),
        outcome_transform=Standardize(m=2),
    ).double()

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    model.likelihood.eval()

    ref = torch.tensor(REF_POINT, dtype=torch.double)
    partitioning = FastNondominatedPartitioning(ref_point=ref, Y=TRAIN_Y)
    ehvi = ExpectedHypervolumeImprovement(
        model=model,
        ref_point=REF_POINT,
        partitioning=partitioning,
    )

    # Analytic EHVI has q=1, hence candidate shape 1 x 1 x d.
    x = torch.tensor([[[p[0], p[1]]]], dtype=torch.double)
    return ehvi(x)[0]


if __name__ == "__main__":
    point = engineering_model([0.30, 0.70])
    print("ordinary BoTorch point EHVI at [0.30, 0.70] =", float(point.detach().cpu()))

    box = [
        Interval.enclosing("0.265", "0.335"),
        Interval.enclosing("0.665", "0.735"),
    ]
    result = autoia(engineering_model, box)
    print(result)
    print("This is the generic certificate.  Example 35 adds user mathematics.")
    assert result.machine_rigorous
