# AutoIA new-user tutorial — Version-1 candidate API

This tutorial starts from the interface intended for an engineer who has not
studied interval analysis or AutoIA's internal representations.

## 0. Start with the numbered examples

The source bundle contains one progressive sequence under `examples/`. Run these
one at a time when learning AutoIA:

| Example | Main idea |
|---|---|
| `00_smoke_test.py` | smallest zero-knob rigorous call |
| `01_scalar_nonlinear.py` | ordinary nonlinear NumPy code |
| `02_vector_numpy.py` | vector outputs, arrays, reductions, dot products |
| `03_linear_solve.py` | semantic `np.linalg.solve` certificate |
| `04_branching.py` | uncertain Python `if` and pathwise proof |
| `05_subdivision.py` | whole-box, fixed, and adaptive subdivision |
| `06_prepared_many_box.py` | prepared sessions, cache reuse, many boxes |
| `07_generator_cancellation.py` | automatic shared-error retry plus expert comparison |
| `08_normal_functions.py` | normal PDF/CDF and `log10` |
| `09_failure_handling.py` | typed fail-closed behavior and domain proofs |
| `10_ei_like.py` | generic EI-like acquisition calculation |
| `11_application_monotonicity.py` | simplest user-supplied theorem |
| `12_application_compose.py` | certify intermediates, then apply a theorem |
| `13_application_lipschitz.py` | a user-known Lipschitz bound |
| `14_application_portfolio.py` | minimum of independent rigorous upper bounds |
| `15_downstream_gp_application_rbf.py` | downstream GP application RBF application theorem |
| `16_downstream_gp_application_matern.py` | downstream GP application Matérn 1/2, 3/2, and 5/2 |
| `17_botorch_downstream_gp_application.py` | optional fitted BoTorch `SingleTaskGP` -> downstream GP application freeze adapter |
| `18_botorch_mean_portfolio.py` | optional RBF radial-vs-joint posterior-mean bound study |

Run the whole sequence with:

```bash
python examples/run_all_examples.py
```

The first ten examples teach generic AutoIA before application-specific theorem
code appears. This ordering is intentional.


## 1. The normal call has two ideas

Write an ordinary Python/NumPy model and describe the uncertain physical input
box with `Interval` objects.

```python
from autoia import Interval, autoia


def model(p):
    x, y = p
    return x*x + 2.0*y


result = autoia(
    model,
    [Interval(1.0, 2.0), Interval(3.0, 4.0)],
)

print(result)
print(result.lower)
print(result.upper)
```

`autoia(model, box)` is now the novice default. AutoIA begins with its release
baseline, currently a plain physical-polynomial representation of order 4, and
uses cheap structural diagnostics to decide whether one bounded retry is worth
the cost. The user does **not** need to choose polynomial order, shared-error
generators, Bernstein ranging, or subdivision before the first run.

A successful result with `machine_rigorous=True` means that every exact-real
model output produced by every physical input in the supplied box is contained
in the reported interval, subject to the documented Version-1 language and
platform contract.

## 2. What the automatic controller may do

The release-oriented controller is deliberately small. It may retry at one
higher polynomial order when the unstructured rigorous remainder is at least as
wide as the retained physical-polynomial variation. It may retry with the
shared-generator representation when AutoIA detects a strong signature that a
single approximation error was reused and later recombined after the retained
physical dependence cancelled.

Every retry is independently rigorous. If two successful runs give intervals
`Y1` and `Y2`, AutoIA returns their componentwise intersection. Therefore a bad
controller decision can waste time or miss a possible tightening, but cannot
turn a valid certificate into an invalid one.

The ordinary printed result is intentionally compact. If you want to know why
an optional retry occurred, use:

```python
print(result.explain_configuration())
```

Full research diagnostics remain available as:

```python
result.configuration_diagnostics
```

Most users should not need that dictionary.

## 3. Expert overrides are explicit

If you explicitly supply `order=` or `representation=` and do not also supply a
configuration policy, AutoIA interprets that as a request for one fixed
configuration. This is useful for reproducibility and research comparisons.

```python
fixed_order5 = autoia(model, box, order=5)
fixed_generator = autoia(model, box, representation="generator")
```

If an expert wants an automatic controller with a nonstandard baseline, state
that explicitly:

```python
from autoia import ConfigurationPolicy

result = autoia(
    model,
    box,
    order=5,
    configuration_policy=ConfigurationPolicy(mode="auto"),
)
```

The deliberately expensive research oracle is also explicit:

```python
result = autoia(
    model,
    box,
    configuration_policy=ConfigurationPolicy(mode="reference"),
)
```

## 4. Repeated boxes

For repeated certification of the same program, use `prepare`:

```python
from autoia import prepare

session = prepare(model)

r1 = session.certify([Interval(1.0, 1.5), Interval(3.0, 3.5)])
r2 = session.certify([Interval(1.5, 2.0), Interval(3.5, 4.0)])
```

`prepare(model)` uses the same automatic novice semantics. Alternate
configuration sessions are created lazily only if a queried box justifies them,
and then reused. Explicit `order=` or `representation=` again requests a fixed
prepared configuration unless a configuration policy is supplied explicitly.

## 5. What a beginner should tune

For the first run: nothing beyond the physical input intervals.

If AutoIA returns a certificate that is too wide for the engineering task, the
next useful information is the task requirement itself, for example a desired
output width or a safety threshold. Representation names, polynomial order,
Bernstein coefficient limits, and generator settings are expert diagnostics and
overrides, not prerequisites for using AutoIA.

If AutoIA fails closed, read the typed exception first. Do not rewrite the model
merely to force a certificate until the diagnostic identifies whether the issue
is an unsupported operation, a true mathematical-domain problem, or a proof
budget/sharpness limitation.

## 6. Adding an application-specific theorem

AutoIA can certify ordinary code automatically, but an application expert may
know a stronger mathematical theorem that cannot reasonably be inferred from
arbitrary source code.  The public application-theorem interface lets you use
that theorem without learning AutoIA's polynomial, generator, Bernstein, or
trusted-backend internals.

The central separation is:

- **you supply the mathematical implication** that makes the bound valid;
- **AutoIA rigorously evaluates the numerical bound** using outward-rounded
  binary64 arithmetic;
- checked assumptions are recorded in the returned theorem result;
- application code never receives an internal AutoIA representation.

A minimal theorem looks like this:

```python
import numpy as np
from autoia.application import ApplicationBox, application_theorem


@application_theorem(name="exp is increasing", target="f")
def exp_upper(box, cert):
    # User theorem: exp(x) is increasing, hence max exp(x) = exp(u).
    x_upper = cert.point(box.upper[0])
    return cert.exp(x_upper)


box = ApplicationBox(np.array([-0.7]), np.array([1.2]))
bound = exp_upper(box)
print(bound.value)
```

`cert.point(x)` explicitly declares a stored binary64 number to be a fixed exact
real constant for the theorem calculation.  Arithmetic on the returned
`CertifiedInterval` object is automatically outward rounded.  For example,
`+`, `-`, `*`, `/`, powers, `abs`, `cert.exp`, `cert.sqrt`, `cert.log`,
`cert.sin`, `cert.cos`, normal PDF/CDF, rigorous sums, point-by-interval dot
products, and lower-triangular solves are public certified operations.

Do **not** perform proof-critical arithmetic in ordinary floats and then wrap the
answer afterward.  Once a value participates in theorem-bound arithmetic, keep
it as a `CertifiedInterval`.

## 7. Let AutoIA certify an intermediate quantity

A theorem can ask ordinary AutoIA to enclose a subcomputation:

```python
import numpy as np
from autoia.application import ApplicationBox, application_theorem


def a_model(p):
    return p[0] + p[1]*p[1]


def b_model(p):
    return np.exp(-p[0]*p[1])


@application_theorem(name="monotone outer composition", target="score")
def score_upper(box, cert):
    a = cert.enclose(a_model, box)
    b = cert.enclose(b_model, box)

    cert.require(b.lower >= 0.0, "b is nonnegative on the box")

    # Suppose the application theorem says G(a,b)=exp(a)+sqrt(b)
    # is nondecreasing in both arguments.
    return cert.exp(cert.point(a.upper)) + cert.sqrt(cert.point(b.upper))
```

`cert.enclose` uses AutoIA's normal automatic configuration internally.  The
application theorem sees only certified lower and upper endpoints.  It does not
know or need to know whether AutoIA used a polynomial, shared generators,
Bernstein ranging, monotonicity reduction, or a configuration retry.

## 8. Checked theorem assumptions

Use `cert.require(condition, description)` for assumptions that can be checked
at runtime:

```python
@application_theorem(name="known Lipschitz theorem", target="f")
def bound(box, cert, L):
    cert.require(L >= 0.0, "Lipschitz constant L is nonnegative")
    ...
```

If a required condition is false, the theorem fails closed with a typed
`CertificationError`.  Successful checks are stored in
`bound.assumptions_checked`.

This is not a symbolic theorem prover.  For a theorem marked
`theorem_status="user_asserted"`, AutoIA does not prove that the theorem itself
is mathematically correct.  It proves the numerical enclosure produced by the
theorem code, conditional on the supplied theorem and its checked assumptions.
For a theorem established elsewhere, use for example
`theorem_status="externally_verified"` to make that provenance explicit.

## 9. Combining several valid theorems

If several independent theorems bound the same quantity from above, their
minimum is still a valid upper bound:

```python
from autoia.application import minimum_upper_bound

b1 = theorem_one(box)
b2 = theorem_two(box)
best = minimum_upper_bound(b1, b2)

print(best.value)
print(best.winner.theorem)
```

This is a **proof portfolio**, not a heuristic replacement of one proof by
another.  Every admitted member must independently be valid.  A weak theorem
therefore costs time but cannot weaken the returned upper bound.

## 10. downstream GP application as the acceptance example

The downstream GP application validation application is the nontrivial reference example for this
interface.  Its kernel-specific fact is more general than RBF: suppose

```text
k(x,xi) = sigma_f^2 rho(r_i(x)),
```

where `r_i(x)` is scaled Euclidean distance to training point `xi` and `rho` is
nonincreasing.  Exact nearest/farthest box geometry gives `r_min` and `r_max`,
so the application theorem immediately gives

```text
sigma_f^2 rho(r_max) <= k(x,xi) <= sigma_f^2 rho(r_min).
```

RBF is one such profile.  The same theorem applies to the common Matern kernels

```text
nu = 1/2:  rho(r) = exp(-r)
nu = 3/2:  rho(r) = (1 + sqrt(3) r) exp(-sqrt(3) r)
nu = 5/2:  rho(r) = (1 + sqrt(5) r + 5 r^2/3) exp(-sqrt(5) r).
```

All four profiles are nonincreasing for `r>=0`.  After their kernel-component
bounds are formed, downstream GP application reuses exactly the same certified GP-posterior
calculation.  Expected Improvement then uses the kernel-independent theorem that
it is nondecreasing in the improvement numerator and in nonnegative posterior
standard deviation.

For Matern `3/2` and `5/2`, `r` occurs several times in the formula.  The public
bridge therefore does **not** interval-evaluate the whole formula over
`[r_min,r_max]`.  It uses the nonincreasing-profile theorem and rigorously
evaluates only `rho(r_max)` and `rho(r_min)`.  This is a concrete example of an
application theorem removing avoidable dependency inflation.

The constants `sqrt(3)` and `sqrt(5)` in the worked demo are the stored
binary64 constants used by the ordinary GP implementation.  AutoIA certifies
the subsequent theorem-bound arithmetic conditional on those fixed constants.

Arbitrary Matern `nu` is not included here because its usual expression contains
the modified Bessel function `K_nu`, for which this checkpoint does not expose a
rigorous public primitive.

`validation/downstream_gp_application/public_application_bridge.py` implements all of these scalar
downstream GP application theorems through `autoia.application` only.  It imports no
`autoia.trusted`, `autoia.enclosures`, `autoia.polynomial`, `autoia.semantic`, or
`autoia.controller` module.  The historical RBF frozen-box upper endpoints still
match the original scalar rigorous bridge exactly.

The progression of runnable source-tree examples is:

```text
examples/application_theorems/01_monotone_endpoint.py
examples/application_theorems/02_certify_then_compose.py
examples/application_theorems/03_lipschitz_bound.py
examples/application_theorems/04_bound_portfolio.py
examples/application_theorems/05_downstream_gp_application_public.py
examples/application_theorems/06_downstream_gp_application_radial_kernels.py
```

The first four show the public concepts in isolation.  Example 5 gives the
historical RBF downstream GP application acceptance case.  Example 6 shows that an application
user can change RBF to Matern `1/2`, `3/2`, or `5/2` by changing the radial
profile theorem while reusing the posterior and EI proof unchanged.


## 11. Freezing a fitted BoTorch / GPyTorch exact GP

A common workflow is to fit the surrogate normally in BoTorch and certify only
the acquisition optimization stage. AutoIA does not need to trace the optimizer
or autograd graph. For the supported scalar exact-GP subset:

```python
from autoia.integrations import freeze_botorch_single_task_gp
from autoia.applications.downstream_gp_application import rigorous_downstream_gp_application_frozen_gp_upper_one_box

frozen = freeze_botorch_single_task_gp(
    model,
    objective="maximize",
    best_f=float(train_Y.max()),
)

bound = rigorous_downstream_gp_application_frozen_gp_upper_one_box(lower, upper, frozen)
print(bound.value)
```

`model` may originally reside on CPU, CUDA, or Apple MPS. The adapter detaches
its fitted tensors, moves them to CPU, and only then promotes them to binary64.
This keeps AutoIA's Version-1 proof backend unchanged. If the source model used
BoTorch `Normalize`, the frozen length scales are converted back to physical
input coordinates, so `lower` and `upper` above are the original engineering
coordinates rather than normalized coordinates. If the source model used
`Standardize`, EI is converted back to the original output units.

The current adapter deliberately rejects unsupported structure rather than
guessing. The initial release subset is single-output exact GP regression with
RBF or Matern `nu=1/2,3/2,5/2`, scalar constant/zero mean, Gaussian or fixed
Gaussian noise, and full-dimensional affine input transforms. See
`examples/17_botorch_downstream_gp_application.py` and `GP_INTEGRATION.md`.

### Dependency-preserving RBF posterior-mean portfolio

For a frozen RBF GP, the cheap radial theorem bounds every kernel component
independently and then combines those component intervals. This is rigorous, but
it can lose the fact that all kernel components depend on the same physical
design point. Step 20g therefore adds a second RBF mean route that asks ordinary
AutoIA to enclose the complete posterior mean as one function

```text
mu(x) = mean_constant + sum_i alpha_i * sigma_f2 * exp(-r_i(x)^2/2).
```

The radial and joint mean intervals are intersected. The intersection is still a
rigorous enclosure because both members independently contain the same posterior
mean. The existing radial posterior-variance / sigma upper bound is unchanged,
so this step isolates the mean-dependency bottleneck.

`rigorous_downstream_gp_application_frozen_gp_upper_one_box(...)` uses this stronger mean portfolio
for RBF frozen states. The previous radial-only behavior remains available as
`rigorous_downstream_gp_application_frozen_gp_radial_upper_one_box(...)`. Matérn frozen states keep
the radial route in this checkpoint; a generic AutoIA trace through the Euclidean
radius needs a separate performance/smoothness study before becoming a default.

See `examples/18_botorch_mean_portfolio.py`.

