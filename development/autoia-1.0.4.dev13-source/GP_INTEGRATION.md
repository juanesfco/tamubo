# Transparent exact-GP posterior semantics

**Development status:** AutoIA `1.0.4.dev2`

AutoIA recognizes a deliberately narrow mathematical class of fitted scalar
exact Gaussian-process (GP) regressors across multiple ordinary Python
interfaces.  The user does not manually freeze the model or rewrite it in an
AutoIA-specific GP language.

The supported frontends in this development build are:

- GPyTorch `gpytorch.models.ExactGP`;
- BoTorch `botorch.models.SingleTaskGP`;
- scikit-learn `sklearn.gaussian_process.GaussianProcessRegressor`;
- hand-coded GP algebra composed from ordinary AutoIA-supported numerical
  operations remains usable independently of these frontend adapters.

All recognized library frontends map to one common `FrozenExactGPState` and one
shared validated posterior implementation.

## Common mathematical subset

The first broad coverage contract intentionally remains fail-closed:

- one scalar exact GP output;
- RBF covariance;
- Matern covariance with `nu = 1/2`, `3/2`, or `5/2`;
- one positive constant/output scale;
- zero or scalar constant prior mean when the frontend exposes one;
- Gaussian observation noise as represented by the fitted exact-GP state;
- common affine input normalization;
- common affine output normalization/standardization;
- posterior mean and marginal posterior variance;
- frontend standard-deviation forms where they can be represented soundly.

Unsupported kernels, additive/composite kernels outside the listed subset,
non-Gaussian likelihoods, classification, sparse/inducing approximations,
variational GPs, deep GPs, multitask/multi-output correlated models, posterior
hyperparameter integration, and arbitrary framework execution fail explicitly.

## Ordinary scikit-learn workflow

A model may be constructed and fitted inside the engineering function.  Fitting
is ordinary floating-point scikit-learn computation; AutoIA certifies posterior
evaluation conditional on the stored fitted binary64 state.

```python
import math
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from autoia import autoia, Interval


def engineering_model(p):
    X = np.linspace(0.0, 1.0, 12).reshape(-1, 1)
    y = np.sin(2.0 * math.pi * X[:, 0])

    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(0.2),
        alpha=1.0e-4,
        normalize_y=True,
    )
    gp.fit(X, y)

    x = np.array([[p[0]]])
    mean, std = gp.predict(x, return_std=True)
    return mean[0], std[0]


result = autoia(
    engineering_model,
    [Interval.enclosing("0.20", "0.30")],
)
```

No AutoIA GP adapter appears in the normal workflow.  The complete runnable
example is `examples/29_sklearn_transparent_gpr.py`.

In the current development subset, sklearn supports:

- `RBF`;
- `Matern` with `nu = 0.5`, `1.5`, or `2.5`;
- either kernel optionally multiplied by one `ConstantKernel`;
- scalar-output fitted GPR;
- `normalize_y=False` or the ordinary scalar-output affine normalization used by
  `normalize_y=True`;
- `predict(X)`;
- `predict(X, return_std=True)`;
- `predict(X, return_cov=True)` for the supported scalar-output exact-GP subset.

## Shared posterior semantics

For an uncertain query point `x`, every supported frontend is mapped to fixed
stored state and evaluated through

```text
mu_int(x) = m + k_x^T alpha,
L v(x)    = k_x,
s2_int(x) = k(x,x) - v(x)^T v(x).
```

Here:

- `x` is the uncertain query point;
- `m` is the frozen scalar prior mean in internal output coordinates;
- `k_x` is the kernel vector between `x` and frozen training inputs;
- `alpha` is the frozen posterior coefficient vector;
- `L` is the frozen lower-triangular Cholesky factor;
- `s2_int(x)` is the marginal latent posterior variance.

With affine output offset `a` and positive scale `b`, AutoIA uses

```text
mu(x) = a + b * mu_int(x),
s2(x) = b^2 * s2_int(x).
```

The current variance formula is rigorous but can be very loose on a wide query
box because it subtracts strongly dependent terms.  AutoIA does not silently
subdivide the user's physical input box to repair that looseness.  Box division
remains under the user's or an outer algorithm's control.

## Fitting and trust boundary

Hyperparameter fitting, optimizer convergence, autodiff, framework arithmetic,
and accelerator arithmetic are not certified.  Fitted numerical state is treated
as fixed stored data.  Proof-critical posterior evaluation over uncertain query
inputs is executed through AutoIA's validated arithmetic.

For scikit-learn the extractor uses fitted `X_train_`, `alpha_`, `L_`, `kernel_`,
and scalar output-normalization statistics.  For GPyTorch/BoTorch the equivalent
supported fitted state is extracted from their model and likelihood objects.

## Expert API

The expert/audit API includes:

```python
from autoia.integrations import (
    FrozenExactGPState,
    freeze_gpytorch_exact_gp,
    freeze_botorch_single_task_gp,
    freeze_sklearn_gaussian_process_regressor,
    frozen_gp_posterior,
    frozen_gp_posterior_mean,
    frozen_gp_posterior_variance,
    frozen_gp_posterior_covariance,
)
```

These functions are not required by ordinary user code.
## Acquisition functions remain application examples

The supported GP frontends provide posterior semantics only. Expected Improvement
(EI) and Expected Hypervolume Improvement (EHVI) are not GP-library semantics and
are not built into the AutoIA core. Examples 30--33 demonstrate the intended
advanced pattern: ordinary acquisition code can be certified directly, while a
user may optionally supply independent mathematical theorems and combine them
with the generic certificate through the public application-theorem portfolio API.



## 1.0.4.dev4 independent multi-output BoTorch compatibility

The transparent BoTorch boundary now also recognizes the standard multi-output
``SingleTaskGP`` representation in which BoTorch stores independent outputs as an
output-augmented GPyTorch batch.  This is deliberately not correlated multi-task
GP support.  The supported surface has no additional model batch dimensions and
freezes each independent output into the same scalar ``FrozenExactGPState`` used
by the existing posterior backend.

BoTorch analytic acquisition code calls ``model.posterior`` with
``posterior_transform=None`` explicitly; that keyword is now accepted as exactly
equivalent to omitting the transform.  A non-``None`` posterior transform still
fails closed.

The same development pass adds a narrow validated tensor carrier for deterministic
post-posterior tensor algebra (broadcast arithmetic, pointwise clamp/sqrt/exp/erf,
stack, gather, product, and sum) and a fixed Normal-distribution proxy for validated
arguments.  These operations are application-independent compatibility semantics;
no EI or EHVI acquisition formula is encoded in AutoIA core.

The first motivating live acceptance case is ordinary BoTorch analytic EHVI built
inside an engineering function.  Live package acceptance remains required before
this development checkpoint is promoted.
