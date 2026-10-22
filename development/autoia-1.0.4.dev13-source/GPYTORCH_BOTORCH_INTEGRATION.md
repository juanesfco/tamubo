# Transparent GPyTorch and BoTorch posterior semantics

**Applies to:** AutoIA `1.0.2`

AutoIA 1.0.2 can recognize a deliberately narrow class of fitted scalar exact
Gaussian-process (GP) models while executing ordinary user source. The normal
workflow does **not** require the user to freeze the GP manually or rewrite the
model in an AutoIA-specific language.

## Normal GPyTorch workflow

The user may fit an ordinary `gpytorch.models.ExactGP`, place it in evaluation
mode, and query it inside ordinary engineering code:

```python
import torch
import gpytorch
from autoia import Interval, autoia

# train_x, train_y, likelihood, and ExactGPModel are created/fitted normally.
model.eval()
likelihood.eval()

def engineering_model(p):
    x = torch.tensor([[p[0]]], dtype=torch.float64)
    posterior = model(x)
    return posterior.mean[0]

result = autoia(engineering_model, [Interval(0.20, 0.30)])
print(result)
```

The complete runnable example, including model definition and fitting, is
`examples/27_gpytorch_transparent_exact_gp.py`.

## Normal BoTorch workflow

For a fitted `botorch.models.SingleTaskGP`, the ordinary public posterior call is
recognized:

```python
import torch
from autoia import Interval, autoia

model.eval()

def engineering_model(p):
    x = torch.tensor([[p[0]]], dtype=torch.float64)
    posterior = model.posterior(x)
    return posterior.mean[0, 0] + posterior.variance[0, 0]

result = autoia(engineering_model, [Interval(0.20, 0.30)])
print(result)
```

The complete runnable example, including fitting and affine input/outcome
transforms, is `examples/28_botorch_transparent_single_task_gp.py`.

## What AutoIA recognizes

The initial 1.0.2 subset is intentionally narrow and fail-closed:

- one scalar exact GP output;
- `RBFKernel`;
- `MaternKernel` with `nu = 1/2`, `3/2`, or `5/2`;
- optional `ScaleKernel`;
- `ZeroMean` or scalar `ConstantMean`;
- Gaussian observation noise, scalar or one stored value per training point;
- no input transform, or an affine transform compatible with BoTorch
  `Normalize` or `InputStandardize`;
- no outcome transform, or affine standardization compatible with BoTorch
  `Standardize`;
- source fitted state residing on CPU, CUDA, or MPS, provided it can be copied
  to CPU binary64 for the trusted proof state.

Unsupported kernels, non-affine transforms, multitask/multi-output structures,
variational GPs, batched model states, arbitrary Torch programs, and unsupported
posterior options fail explicitly.

## Transparent semantic boundary

When `autoia(...)` locally executes the user's function and encounters a
supported fitted GP object, it substitutes a semantic proxy only inside that
local execution. The original model object is not mutated.

For an uncertain query point `x`, AutoIA freezes fixed numerical GP state and
uses the posterior formulas

```text
mu_int(x) = m + k_x^T alpha,
L v(x)    = k_x,
s2_int(x) = k(x,x) - v(x)^T v(x).
```

Here:

- `x` is the uncertain query point;
- `m` is the frozen scalar mean;
- `k_x` is the kernel vector between `x` and the frozen training inputs;
- `alpha` is the frozen posterior coefficient vector;
- `L` is the frozen lower-triangular Cholesky factor;
- `s2_int(x)` is the latent posterior variance before any affine outcome
  untransform.

If an affine outcome standardization is present, with stored offset `a` and
positive scale `b`, AutoIA returns

```text
mu(x) = a + b * mu_int(x),
s2(x) = b^2 * s2_int(x).
```

All arithmetic involving the uncertain query is performed through AutoIA's
validated arithmetic and therefore receives the same proof-critical treatment as
other supported ordinary source semantics.

## Torch is not a proof backend

This feature is **not** arbitrary PyTorch interval execution. AutoIA recognizes
only the narrow tensor construction/shape operations required to form supported
GP query points and the supported fitted-GP posterior calls. Other Torch
operations retain ordinary behavior for numeric inputs and are not silently
claimed to be certified for uncertain AutoIA values.

## Device and trust boundary

The fitted model may have been trained on an accelerator. Freezing performs the
conceptual transition

```text
fitted tensor state -> detach -> CPU -> binary64 -> immutable NumPy data.
```

The statistical fitting procedure, autograd, optimizer convergence, training
arithmetic, and accelerator arithmetic are **not certified**. The copied
binary64 fitted parameters are treated as fixed stored data. Proof-critical
posterior evaluation over uncertain query inputs is performed by AutoIA's trusted
CPU binary64 validated backend.

The 1.0.2 independent posterior audit checks 504 high-precision containment
witnesses across RBF and all three supported Matern profiles.

## Expert API

The lower-level freeze functions remain available for diagnostics, testing, and
advanced integration work:

```python
from autoia.integrations import (
    FrozenExactGPState,
    freeze_gpytorch_exact_gp,
    freeze_botorch_single_task_gp,
    frozen_gp_posterior,
    frozen_gp_posterior_mean,
    frozen_gp_posterior_variance,
)
```

They are not required by the normal transparent examples above.
