# AutoIA 1.0.4.dev13

AutoIA provides automatic, machine-rigorous validated interval analysis for
ordinary engineering Python code. A user writes ordinary numerical source and
calls approximately

```python
from autoia import Interval, autoia

result = autoia(model, [Interval(0.1, 0.2), Interval(1.0, 1.1)])
```

AutoIA locally reinterprets supported numerical semantics using validated values.
It does not require a wholesale rewrite into an interval-specific language.

## Release status

Version `1.0.4.dev13` is a V1 development checkpoint. In addition to the
existing exact-input, GP, elementary-function, implicit-solve, and array
semantics, it adds certified optimal-value semantics for ordinary
`scipy.optimize.linprog` calls with inequality/equality constraints and constant variable
bounds. It also extends local semantic dispatch through ordinary Python closure
variables and captured source-level user objects.  dev11 additionally covers
function-valued attributes on those captured objects and adds an explicit
persistent many-box executor for dynamic branch-and-bound workloads. dev12 adds
per-box failure reporting for those batches. dev13 strengthens parametric
`linprog` lower bounds with automatic shared McCormick/RLT lifting and
cost-capped certified optimization-based bound tightening (OBBT).

The trusted proof backend remains CPU binary64 with outward rounding. CUDA and
Apple MPS may be used only as explicitly untrusted proposal assistance where an
API supports it; they are never proof backends.

## Important capabilities

- outward-rounded binary64 interval arithmetic;
- polynomial enclosures and shared-generator enclosures;
- Bernstein and monotonicity ranging;
- automatic representation/configuration selection;
- path-aware execution of ordinary Python source;
- local NumPy and `math` semantic dispatch;
- selected SciPy semantics;
- validated `numpy.linalg.solve`;
- validated scalar root solving through ordinary `scipy.optimize.root_scalar`;
- validated square nonlinear systems through ordinary `scipy.optimize.fsolve`;
- certified LP optimal values through ordinary `scipy.optimize.linprog` (`res.fun`);
- lifted McCormick/RLT lower-bound tightening for polynomial-parametric LP coefficients;
- fixed-data linear `RegularGridInterpolator`;
- supported SPD matrix-family operations;
- prepared execution and many-box execution;
- trusted multicore CPU proof execution;
- transparent supported exact-GP posterior evaluation from fitted GPyTorch,
  BoTorch, and scikit-learn models.

## Input endpoint semantics

`Interval(0.1, 0.3)` means the exact real interval whose endpoints are the
stored binary64 values of those Python floats.

For intended exact decimal/rational endpoints, use:

```python
Interval.enclosing("0.1", "0.3")
```

This constructs finite binary64 proof endpoints enclosing the mathematical
interval `[1/10, 3/10]`. Exact endpoint inputs may be decimal strings,
`decimal.Decimal`, `fractions.Fraction`, or integers.

`Interval.outward(0.1, 0.3)` explicitly pads already-stored floats by one ULP in
each outward direction. It is not a provenance proof for an arbitrary computed
floating-point quantity.

Ordinary constants inside engineering source retain stored-binary64 semantics.
An uncertain physical coefficient should ordinarily be exposed as an uncertain
input rather than wrapped as an exact source constant.

See `INPUT_SPECIFICATION.md` for the complete contract.

## Transparent fitted-GP semantics

AutoIA recognizes supported fitted GP objects during local semantic execution.
Normal user code can therefore remain ordinary GPyTorch, BoTorch, or scikit-learn
source:

```python
import torch
import gpytorch
from autoia import Interval, autoia

# Define and fit an ordinary gpytorch.models.ExactGP first.
model.eval()

def engineering_model(p):
    x = torch.tensor([[p[0]]], dtype=torch.float64)
    posterior = model(x)
    return posterior.mean[0]

result = autoia(engineering_model, [Interval(0.20, 0.30)])
```

The normal workflow does not require a manual freeze adapter. A lower-level
freeze API remains available for expert diagnostics and testing.

The initial supported subset is intentionally narrow: scalar exact GP, RBF or
Matern `nu=1/2,3/2,5/2`, optional `ScaleKernel`, zero/scalar constant mean,
Gaussian noise, and common affine input/outcome transforms. Unsupported GP
structures or unsupported Torch use fail explicitly.

Training, autograd, hyperparameter optimization, and accelerator arithmetic are
not certified. The fitted binary64 numerical state is treated as fixed stored
data; uncertain-query posterior evaluation is reconstructed using AutoIA's
validated CPU arithmetic.

See `GP_INTEGRATION.md` and the complete examples:

- `examples/27_gpytorch_transparent_exact_gp.py`
- `examples/28_botorch_transparent_single_task_gp.py`
- `examples/29_sklearn_transparent_gpr.py`

## Advanced acquisition-function examples

EI and EHVI remain application mathematics rather than AutoIA core semantics.
The numbered examples deliberately show both levels of use:

- `examples/30_handcoded_gp_expected_improvement.py` — ordinary hand-coded exact GP plus analytic EI passed directly to `autoia(...)`;
- `examples/31_expected_improvement_user_theorems.py` — the same application style augmented by user-supplied EI monotonicity, GP nonnegativity, RBF box geometry, and a rigorous bound portfolio;
- `examples/32_handcoded_analytic_ehvi.py` — ordinary analytic two-objective `q=1` EHVI built from GP posterior quantities and normal PDF/CDF;
- `examples/33_ehvi_user_gaussian_theorem.py` — the same EHVI application augmented by user-supplied Gaussian positive-part and finite-band monotonicity theorems.

These examples are intentionally not acquisition-function primitives.  They demonstrate
how a domain expert can combine a generic whole-computation certificate with
independently valid application theorems through the public theorem API.


## Certified linear-program optimal values

AutoIA recognizes ordinary `scipy.optimize.linprog(...)` calls during validated
execution. SciPy/HiGHS proposes primal and dual witnesses; AutoIA independently
certifies `res.fun`. Weak duality supplies the rigorous lower side. A fixed
primal witness supplies the upper side only after AutoIA proves it feasible for
every physical input in the box. If no universal primal witness is proved,
AutoIA preserves the valid lower certificate and returns `[L,+inf]` with
`value_upper_certified=False` instead of discarding the lower bound. LP
coefficients may depend on the same uncertain physical inputs, and the
certificate arithmetic retains that shared dependency.

Version 1 supports `A_ub`, `b_ub`, `A_eq`, `b_eq`, and constant variable bounds.
Equalities are reduced exactly to paired inequalities inside the certified semantic.
`res.x` and mixed-integer semantics fail closed. See
`LINEAR_PROGRAM_SEMANTICS.md`, `examples/38_scipy_linprog_optimal_value.py`, and
`examples/39_scipy_linprog_lower_only.py`.

## Prepared and many-box execution

For repeated boxes, prepare the engineering program once:

```python
from autoia import Interval, prepare

prepared = prepare(model)
boxes = [
    [Interval(0.0, 0.1)],
    [Interval(0.1, 0.2)],
    [Interval(0.2, 0.3)],
]
lower, upper = prepared.enclose_many_bounds(
    boxes, workers=1, parallel="serial"
)
```

For a branch-and-bound queue that submits many successive batches, dev11 also
provides a persistent executor.  In process mode the worker processes and their
prepared AutoIA sessions remain alive across calls, avoiding repeated macOS/spawn
and dispatch-preparation cost:

```python
prepared = prepare(model)
with prepared.manybox_executor(
    workers="auto",
    parallel="auto",
    chunk_size=1,
    use_cache=False,
) as pool:
    lo1, hi1 = pool.enclose_many_bounds(batch1)
    lo2, hi2 = pool.enclose_many_bounds(batch2)
```

This changes scheduling only.  Every node remains an independent CPU-binary64
AutoIA proof.  The one-shot safe-auto scheduler deliberately stays serial for
small finite batches; use the persistent executor when repeated expensive batches
make process startup amortization worthwhile.

## Failure semantics

AutoIA is fail-closed. A recognized but unsupported operation raises an explicit
unsupported/certification error rather than silently falling back to ordinary
floating-point arithmetic and labeling the result rigorous.

## Installation

From the verified wheel in the current directory:

```bash
python -m pip install --upgrade --force-reinstall --no-deps ./autoia-1.0.4.dev11-py3-none-any.whl
```

GPyTorch, BoTorch, and scikit-learn are optional integrations. They are deliberately not AutoIA
installation dependencies. Live integration tests and examples run when those
packages are available and otherwise skip explicitly.

## Release validation

The final 1.0.2 release verification records the exact named-test, pytest-subtest,
exact-input audit, high-precision GP audit, numbered-example, clean-wheel, and
release-content audit counts. See `V1_0_2_FINAL_VERIFICATION.md`.

## Documentation

The synchronized V1 documentation set is:

- `AUTOIA_MATHEMATICAL_FOUNDATIONS_V1.pdf`
- `AUTOIA_TUTORIAL_MANUAL_V1.pdf`
- `AUTOIA_USER_DEVELOPER_MANUAL_V1.pdf`
- `AUTOIA_TEST_VALIDATION_STRATEGY_V1.pdf`
- `AUTOIA_V2_PATH_FORWARD.pdf`

Source versions live under `docs/`.

## Package boundary

AutoIA is a general validated-computation package. Application-specific search,
objective-selection, or optimization algorithms belong outside this package and
may depend on AutoIA through its public API. AutoIA does not depend on such
packages and does not embed their algorithms.

### 1.0.4.dev12 many-box failure reporting

Adaptive searches can use `on_error="report"` so one uncertifiable box does not
cancel successful siblings. For compact bounds,
`enclose_many_bounds(..., on_error="report")` returns `(lower, upper, ok)`;
failed rows are `NaN` and `ok=False`. Persistent process executors support the
same mode. Closure/local models in `parallel="process"` use the optional
`autoia[process]` (`cloudpickle`) extra.
