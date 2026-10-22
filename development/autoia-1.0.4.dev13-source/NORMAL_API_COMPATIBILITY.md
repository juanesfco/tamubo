# SciPy normal-distribution API compatibility

Status: implemented in AutoIA 0.15.0.dev2 (Step 4b).

AutoIA already owns the validated mathematics for the real error function,
standard-normal PDF, and standard-normal CDF.  This compatibility layer makes
common SciPy syntax route to those same AutoIA kernels while leaving SciPy
itself unmodified and untrusted for the certificate.

## Supported user-facing forms

The local shadow dispatcher recognizes:

```python
import scipy
scipy.special.erf(x)
scipy.special.ndtr(x)
scipy.stats.norm.pdf(x)
scipy.stats.norm.cdf(x)
scipy.stats.norm.pdf(x, loc=mu, scale=sigma)
scipy.stats.norm.cdf(x, loc=mu, scale=sigma)
```

It also recognizes submodule imports:

```python
import scipy.stats as stats
import scipy.special as special
stats.norm.pdf(x, loc=mu, scale=sigma)
stats.norm.cdf(x, loc=mu, scale=sigma)
special.erf(x)
special.ndtr(x)
```

and direct imports/aliases such as:

```python
from scipy.stats import norm
from scipy.special import erf, ndtr

norm.pdf(x)
norm.cdf(x)
erf(x)
ndtr(x)
```

Frozen legacy distributions are supported:

```python
rv = scipy.stats.norm(loc=mu, scale=sigma)
rv.pdf(x)
rv.cdf(x)
```

including a frozen object or its `pdf`/`cdf` bound method stored in a module-level
variable before AutoIA execution.

When available in the installed SciPy version, the newer distribution-object API
is also supported:

```python
from scipy.stats import Normal
X = Normal(mu=mu, sigma=sigma)
X.pdf(x)
X.cdf(x)
```

AutoIA also supports the two-argument continuous-normal CDF form
`X.cdf(x, y)` by evaluating the mathematically exact difference
`F(y) - F(x)` with validated arithmetic.

## Distribution-parameter scope

For the compatibility layer, `loc`/`mu` and `scale`/`sigma` must currently be
constant real scalars, with positive finite scale.  Uncertain normal-distribution
parameters remain unsupported.  The uncertain argument `x` may be a validated
scalar or an object array containing validated values.

## What is not yet claimed

This Step-4b layer does not yet validate:

- `scipy.stats.norm.logpdf`, `logcdf`, `sf`, `ppf`, or `isf`;
- `scipy.special.log_ndtr`, `ndtri`, or inverse-error-function APIs;
- uncertain `loc`, `scale`, `mu`, or `sigma`;
- non-default numerical `method=` selection on validated `scipy.stats.Normal`
  `pdf`/`cdf` calls;
- SciPy `out=`/general ufunc keyword semantics for validated `erf`/`ndtr`;
- runtime-local SciPy imports inside a model function body;
- SciPy calls hidden inside helpers imported from another module.

These omissions fail closed rather than silently delegating validated values to
SciPy's ordinary numerical implementations.

## No SciPy proof dependency

SciPy is not listed as an AutoIA package dependency.  `autoia.semantic.scipy_dispatch`
has no hard `import scipy` statement.  The compatibility code activates only if
the user's code has already imported SciPy.  If only top-level `scipy` was
imported, AutoIA may lazily import the public `scipy.stats` and `scipy.special`
submodules inside the private execution preparation step.

The real SciPy module, `stats.norm` object, `special.erf`/`ndtr` ufuncs, frozen
objects, and `Normal` classes/instances are never modified.

For validated inputs the dispatch routes to AutoIA's existing rigorous kernels:

- `scipy.special.erf(x)` -> AutoIA `erf(x)`;
- `scipy.special.ndtr(x)` -> AutoIA standard normal CDF;
- `scipy.stats.norm.pdf/cdf` -> AutoIA normal PDF/CDF after validated
  standardization by constant parameters;
- frozen/new normal objects -> the same AutoIA normal PDF/CDF kernels.

Numeric-only calls continue to delegate to the user's real SciPy implementation.

## NumPy and scikit-learn

NumPy exposes normal random sampling, but it does not define a standard public
normal PDF/CDF API analogous to `scipy.stats.norm`, so no separate NumPy normal
PDF/CDF compatibility syntax is required.

AutoIA does not target private scikit-learn normal-distribution helper routines.
Scikit-learn's public modeling APIs build on NumPy/SciPy rather than exposing a
general-purpose public normal PDF/CDF surface equivalent to `scipy.stats.norm`.
Supporting the public SciPy interfaces is therefore the appropriate compatibility
boundary for Version 1.
