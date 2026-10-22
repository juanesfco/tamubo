# AutoIA candidate Version-1 mathematical and software contract

**Status:** evolving Version-1 contract through validated implicit solves, branch/domain hardening, fixed-data interpolation, structured SPD matrix families, and parallel many-box execution.  
**Package version:** `1.0.2`.  
**Purpose:** define the contract that later Version-1 implementation work must preserve.

---

## 1. The problem AutoIA certifies

Let the uncertain real input vector be

\[
p=(p_1,\ldots,p_m)^T,
\]

where `T` denotes transpose and `m` is the number of uncertain input scalars.
For each index `i`, the user supplies a finite interval

\[
p_i\in[\underline p_i,\overline p_i].
\]

The Cartesian input box is

\[
\mathcal P=\prod_{i=1}^m[\underline p_i,\overline p_i].
\]

The engineering model is a deterministic program.  Under the supported AutoIA
semantics, its exact real-valued output is denoted

\[
y=\mathcal P_{\rm code}(p).
\]

A successful AutoIA call returns output intervals

\[
Y_j=[\underline y_j,\overline y_j],
\]

one for each scalar output component `j`.

The primary correctness contract is

\[
\boxed{
  \mathcal P_{{\rm code},j}(p)\in Y_j
  \quad\text{for every }p\in\mathcal P.
}
\]

AutoIA does **not** promise that `Y_j` is the exact interval hull.  Sharpness is
a controller/representation quality issue; inclusion is the correctness issue.

---

## 2. Binary64-constant semantics

A Python source literal such as `0.2` has already been converted by Python to a
binary64 floating-point value before AutoIA sees it.  AutoIA currently treats
that stored binary64 value as an exact real constant.

Thus the certified mathematical semantics are:

> exact real arithmetic applied to the stored binary64 constants and the
> uncertain real inputs, through the supported AutoIA operations.

AutoIA Version 1.0 is **not currently intended to certify every floating-point
rounding event that the original uninstrumented engineering program would
perform**.  It certifies the corresponding real-valued mathematical computation
with binary64 constants unless a semantic primitive has a more specific stated
contract.

This distinction must remain explicit in user documentation.

### 2.1 Physical input endpoint specification

The constants inside the engineering program and the endpoints that define the
physical input set are related but distinct specification boundaries.  AutoIA
1.0.2 preserves the original direct-constructor semantics while adding an
explicit rigorous bridge for exact decimal/rational physical endpoints.

For a user-intended exact real interval

\[
I_i^\star=[a_i,b_i],
\]

AutoIA may certify on a finite binary64 proof interval

\[
I_i=[\underline p_i,\overline p_i]
\]

provided

\[
\boxed{I_i^\star\subseteq I_i.}
\]

The public constructors have the following meanings.

- `Interval(lo, hi)` preserves the Version-1 default: `lo` and `hi` are
  canonicalized to binary64 and those stored values are interpreted as exact
  real endpoints.  Thus $I_i^\star=I_i$.
- `Interval.enclosing(lo, hi)` accepts exact decimal/rational specifications
  (decimal strings, `Decimal`, `Fraction`, or integers), verifies an outward
  binary64 enclosure by exact rational comparison, and uses that enclosing
  binary64 interval as the proof domain.  Floating-point arguments are rejected
  because their original decimal/rational intent is no longer recoverable.
- `Interval.outward(lo, hi)` pads already stored binary64 endpoints by one
  adjacent binary64 number on each side.  This proves containment of the supplied
  stored-float interval in the padded proof interval, but it is not a theorem
  about how the supplied floats were produced before AutoIA was called.

A certificate over the proof box therefore also applies to the intended exact
physical box whenever the constructor records and verifies the inclusion above.
The result exposes descriptive `input_semantics`, `input_specification`, and
`physical_input_box` metadata so this boundary is visible to the user.

This addition does **not** reinterpret numerical constants throughout ordinary
engineering code.  Stored binary64 constants retain the semantics in Section 2.
If a numerical coefficient is physically uncertain, the intended treatment is
to expose it as an uncertain physical input rather than to silently reinterpret
its source literal.

See `INPUT_SPECIFICATION.md` for the user-facing decision rule and examples.

---

## 3. Centered uncertain coordinates

For each uncertain input, AutoIA chooses a stored binary64 center `p_c,i` and a
rigorously enlarged half-width `h_i` such that

\[
[\underline p_i,\overline p_i]
\subseteq
[p_{c,i}-h_i,p_{c,i}+h_i].
\]

Define the centered uncertain coordinate

\[
q_i=p_i-p_{c,i}.
\]

Then

\[
q_i\in[-h_i,h_i].
\]

The symmetric centered proof box is

\[
\mathcal Q=\prod_{i=1}^m[-h_i,h_i].
\]

The *actual centered physical set* is

\[
\mathcal Q_{\rm phys}
=\{p-p_c:p\in\mathcal P\}.
\]

By construction,

\[
\boxed{\mathcal Q_{\rm phys}\subseteq\mathcal Q.}
\]

The distinction matters at exact mathematical-domain boundaries.  The symmetric
proof box `Q` may extend a few binary64 ulps beyond the actual centered physical
set even though it remains a valid over-domain for polynomial ranging.  AutoIA's
semantic certification obligation concerns the original physical inputs, hence
`Q_phys`; proof kernels are free to range over the larger `Q` whenever doing so
is sound and convenient.

The `q_i` symbols retain provenance: every occurrence of the same original
uncertain input refers to the same `q_i`.

---

## 4. Universal `ValidatedValue` contract

The universal AutoIA object is **not** a polynomial.  It is a validated value.

Let `E` be a scalar validated enclosure object.  Its mathematical interpretation
is a set-valued map

\[
\Gamma_E(q)\subseteq\mathbb R,
\]

where `R` denotes the real numbers and `q` is an admissible original-input
coordinate vector.

If an exact program quantity is `z(q)`, the representation invariant is

\[
\boxed{
 z(q)\in\Gamma_E(q)
 \quad\text{for every }q\in\mathcal Q_{\rm phys}.
}
\]

This is the representation-independent semantic invariant that all future
enclosure families must satisfy.  A concrete representation or proof kernel may
establish the stronger property on all of `Q`; AutoIA does not require that
stronger synthetic-domain property when `Q` strictly over-encloses the physical
set.

### 4.1 Range contract

A call

```python
I, diagnostics = E.range(...)
```

must return an interval `I` satisfying at least

\[
\boxed{
 \bigcup_{q\in\mathcal Q_{\rm phys}}\Gamma_E(q)\subseteq I.
}
\]

Current polynomial range kernels usually prove the stronger enclosure obtained
by ranging on all of the symmetric proof box `Q`.  A rigorously justified
physical-domain proof may instead exploit information valid on `Q_phys` and
intersect it with such structural enclosures.  Diagnostics may influence
controller decisions but may never weaken the inclusion contract.

### 4.2 Operation contract

Let `op` be an exact supported scalar operation and let `op#` denote the AutoIA
enclosure implementation.  If

\[
E_{\rm out}=\operatorname{op}^{\#}(E_1,\ldots,E_s),
\]

then soundness requires

\[
\boxed{
 \operatorname{op}(z_1,\ldots,z_s)
 \in\Gamma_{E_{\rm out}}(q)
}
\]

for every admissible physical `q` in `Q_phys` and every choice

\[
z_k\in\Gamma_{E_k}(q).
\]

### 4.3 Conversion contract

A representation conversion

\[
E_A\longrightarrow E_B
\]

is sound only if

\[
\boxed{
 \Gamma_{E_A}(q)\subseteq\Gamma_{E_B}(q)
 \quad\text{for every }q\in\mathcal Q_{\rm phys}.
}
\]

Conversions may lose dependency information.  They may never invent precision
that was previously lost.

### 4.4 Software representation contract

The package-level `ValidatedValue` wrapper now dispatches through a structural
representation interface rather than testing specifically for the polynomial
and interval classes.  A scalar representation used inside engineering-model
arithmetic provides, at minimum:

* a stable `representation_name`;
* domain/backend metadata `h`, `m`, `order`, `backend`, and `range_method`;
* `copy()` and rigorous `range()`;
* `constant_like(value)`;
* `negated()`;
* optionally `unary(op)` when the representation supports validated elementary
  functions;
* `binary_same(other, op)` for same-family supported arithmetic; and
* `convert_from(other)` implementing a **sound** conversion into its own family.

For a mixed-representation operation, the current controller-free dispatch
chooses the representation with larger `promotion_priority`, converts both
operands to that family, and applies the same-family operation.  The priority is
a **strategy convention only**.  Soundness depends on `convert_from` satisfying
the conversion contract above, not on which priority number is chosen.

The current priorities preserve Version-0.11 behavior: polynomial enclosures
are promoted above interval enclosures, so mixed interval/polynomial arithmetic
converts the interval value to a constant polynomial plus a rigorous remainder.
A future representation can participate without modifying every arithmetic
operator in `ValidatedValue` provided it implements this interface and a sound
conversion policy.

Piecewise enclosures are deliberately marked as range-capable but not enabled
for ordinary `ValidatedValue` arithmetic until branch-domain/path alignment has
explicit semantics.

---

## 4.5 Semantic implicit-solve contract

For recognized common nonlinear solve syntax, the exact mathematical state is
defined by a residual equation rather than by SciPy's floating-point iteration.
For a scalar root, AutoIA certifies an equation `F(y,p)=0`; for a square system,
`F(y,p)=0` with vector `y`. SciPy may propose a nominal root, predictor, or
preconditioner, but proposal convergence is never itself a certificate.

A successful semantic implicit solve returns a `ValidatedValue` (or vector of
validated values) whose set-valued interpretation contains an exact solution for
every physical input in the current box. The implementation retains an implicit
dependency-bearing predictor in the original AutoIA input variables and adds only
a theorem-certified correction enclosure.

For supported bracketed `scipy.optimize.root_scalar`, the Version-1 contract also
proves a strict endpoint sign change and one nonzero derivative sign across the
supplied bracket, so the certified solution is the unique bracket root for every
physical input. For supported `scipy.optimize.fsolve`, the contract is local: a
contraction/Krawczyk-style certificate proves one unique solution inside the
returned local tube, but does not exclude remote roots outside that tube.

See `IMPLICIT_SOLVE_SEMANTICS.md` for the exact supported syntax and proof
conditions.

---

## 5. Current concrete representations

### 5.1 Polynomial enclosure

The mature default representation is

\[
\boxed{z(q)\in P(q)+R,}
\]

where

* `P(q)` is a stored point-coefficient polynomial in the **original centered
  uncertain inputs** `q_1,...,q_m`, and
* `R=[r_lo,r_hi]` is a rigorous interval remainder.

Stored polynomial coefficients are binary64 point values interpreted as exact
real numbers.  All coefficient-rounding and truncation errors created by
supported generic arithmetic must be added rigorously to `R`.

This representation is the default for smooth parametric engineering code, but
it is not the universal AutoIA contract.

### 5.2 Interval enclosure

An interval representation stores

\[
\Gamma_E(q)=X
\]

for every `q`, where `X` is an interval.  This is universal and cheap but loses
original-input dependency.

### 5.3 Piecewise enclosure

A piecewise result represents a finite union of separately certified branches.
If branch domains satisfy

\[
\mathcal Q=\bigcup_{s=1}^S\mathcal Q_s
\]

and branch `s` certifies `Y_s`, then

\[
\boxed{
 y(\mathcal Q)\subseteq\bigcup_{s=1}^S Y_s.
}
\]

Piecewise values are top-level finite-union results.  For uncertain Python
control flow, downstream arithmetic is propagated path-by-path by re-executing
the remainder of the engineering model on every feasible path; AutoIA does not
prematurely replace those paths by their interval hull.  `PiecewiseEnclosure`
itself remains a top-level union object rather than a generic arithmetic value.
See `PATH_AWARE_EXECUTION.md`.

---

## 6. Machine-rigorous trusted-kernel contract

`RigorousBinary64Backend` is conditioned on an explicitly audited binary64
runtime contract.  At backend construction AutoIA checks that Python `float` and
NumPy `float64` have the expected binary64 format, current arithmetic exhibits
round-to-nearest behavior, `numpy.nextafter` reaches the adjacent binary64 values,
and gradual underflow/subnormal arithmetic is active.  If that audit fails, the
rigorous backend fails closed.  For every proof-critical scalar operation, AutoIA
then expands the computed result outward using adjacent binary64 values.

The current contract assumes external code does not change the process
floating-point environment during an AutoIA proof after this audit has passed.
AutoIA itself does not intentionally change that environment.

Higher-level proof calculations must be built from trusted scalar operations or
from a separately proved rigorous algorithm.  In particular, ordinary BLAS or
NumPy matrix multiplication is **not** proof-critical merely because one
`nextafter` is applied to the final result.

The smallest proof-critical arithmetic layer is physically isolated in
`autoia.trusted`.  As of `0.13.0.dev1`, that layer is also independently audited
against exact rational arithmetic and performs a fail-closed runtime binary64
environment audit before rigorous backend use.  The trusted layer contains:

- the closed binary64-endpoint `Interval` data type and point/integer helpers;
- runtime validation of the binary64 format, round-to-nearest behavior,
  `nextafter`, and gradual underflow/subnormal support;
- outward-rounded scalar binary64 `+`, `-`, `*`, and `/`;
- explicit trusted dot products, matrix-vector products, matrix-matrix products,
  and nonnegative upper reductions used by certificates;
- trusted real `sqrt`, `exp`, `log`, `log10`, `sin`, `cos`, and `erf` interval kernels, plus rigorous standard-normal PDF/CDF kernels built from
  the audited arithmetic plus analytic series/certificate bounds.

The higher-level reference implementation imports these operations rather than
defining duplicate arithmetic internally.  Polynomial coefficient bounds,
residual bounds, contraction bounds, polynomial ranges, and final endpoints are
then built from this extracted kernel.

A result may report

```python
machine_rigorous == True
```

only if every calculation needed to establish its inclusion certificate follows
this trusted path and no proof obligation was silently delegated to unchecked
floating-point computation.

Proposal computations are different.  Nominal solves, approximate inverses,
polynomial predictors, controller scores, and similar quantities may use
ordinary floating point **provided the returned result is subsequently certified
against the actual stored proposal**.

---

## 7. Failure contract

AutoIA follows a fail-closed rule:

\[
\boxed{
\text{no successful certificate} \;\Longrightarrow\; \text{no rigorous result}.
}
\]

A supported computation may fail because:

1. the source operation is outside the supported language;
2. a required mathematical condition cannot be proven on the current box, for
   example a denominator may contain zero;
3. a semantic certificate such as a contraction test is inconclusive;
4. a configured resource/controller budget is insufficient;
5. a trusted numerical operation overflows or encounters an unsupported
   extended-arithmetic case;
6. an internal invariant is violated.

The package-facing API maps reference-core built-in failures to the following
stable typed hierarchy:

```text
AutoIAError
├── InvalidInputError
├── UnsupportedOperationError
├── CertificationError
├── ResourceLimitError
├── NumericalLimitError
├── ModelExecutionError
└── InternalAutoIAError
```

The distinction is semantic rather than cosmetic.  Unsupported source semantics,
a supported computation that cannot currently be certified, a user's own model
exception, and violation of an AutoIA soundness invariant are different failure
modes and must not be conflated.  The private reference core may continue to use
built-in exceptions during migration, but those exception classes are no longer
part of the public execution API.  Original exceptions are retained through
Python exception chaining where possible.  See `FAILURE_SEMANTICS.md`.

---

## 8. Supported execution subset at this architecture freeze

The current `0.24.0.dev14` package supports the following conservative core subset:

* finite binary64 interval endpoints defining a Cartesian input box;
* scalar `+`, `-`, `*`, `/`;
* constant real powers: all integer exponents (including negative integers when the base excludes zero) and finite non-integer exponents on the documented positive real branch;
* rigorously validated real `sqrt`, `exp`, `log`, `log10`, `sin`, `cos`, and `erf`, plus standard-normal PDF/CDF, through common
  module/global NumPy and standard-library `math` call forms;
* NumPy object-array construction/indexing sufficient for the existing examples;
* local semantic interception of `numpy.linalg.solve`;
* polynomial semantic `numpy.linalg.solve` for affine or higher-order retained
  polynomial matrix/right-hand-side dependence in the original centered inputs,
  with rigorous additive remainders, rigorous predictor/preconditioner portfolios,
  and vector or matrix right-hand sides;
* interval semantic solve as a dependency-blind alternative;
* scalar validated comparisons `<`, `<=`, `>`, `>=`, `==`, and `!=`, with
  path-aware ordinary Python `if` / `elif` / `else`, truth testing, and
  short-circuit Boolean execution under `BranchPolicy`;
* generic hand-coded algebraic solutions when written entirely with supported
  scalar operations;
* termwise, monotonicity, Bernstein, monotonicity-reduced Bernstein, and
  portfolio range algorithms;
* separate internal and final range policies;
* proof-driven escalation for reciprocal nonzero checks;
* supported bracketed `scipy.optimize.root_scalar` as a proposal-plus-independent-proof semantic with uniform bracket/uniqueness certification and a dependency-bearing implicit predictor;
* supported square-system `scipy.optimize.fsolve` as a proposal-plus-independent-proof semantic for one locally unique solution branch in a certified tube;
* fixed finite numeric `scipy.interpolate.RegularGridInterpolator(..., method="linear")` data with uncertain query coordinates, path-aware cell selection, multilinear dependence preservation, descending axes, scalar/vector-valued tables, finite fixed fill values, and `fill_value=None` linear extrapolation;
* parameter-dependent symmetric positive-definite matrix-family certification for structurally symmetric AutoIA-valued matrices, including rigorous positive eigenvalue enclosures, validated `np.linalg.slogdet`/`np.linalg.det` SPD semantics, dependency-preserving matrix-log series, and `quad_solve_spd(A,b)` for `b.T @ solve(A,b)` with spectral/series/optional-solve proof portfolios;
* prepared many-box CPU certification through serial, threaded, or spawn-process execution, including `workers="auto"`, `parallel="auto"`, optional `cloudpickle` model serialization, streaming box iteration, compact bound arrays, and explicit non-proof `ManyBoxExecutionInfo`;
* optional PyTorch-backed Apple MPS and NVIDIA CUDA proposal/assist evaluation through `proposal_map` / `proposal_box_centers`; these accelerator values are explicitly non-rigorous and final certificates still require the trusted CPU binary64 backend;
* local/lazy preparation of reached source-backed pure-Python imported helpers, including helpers that own supported implicit-solve or interpolation calls;
* fixed and structure-aware adaptive subdivision at the top level;
* adaptive split selection through a cheap structural score by default, with the historical one-step lookahead retained as an explicit research-oracle policy;
* Step-4 NumPy object-array semantics for the documented constructor/mutation,
  broadcasting, structural, reduction, dot/matmul/inner/outer, trace, and
  vector/Frobenius-norm subset described in `ARRAY_SEMANTICS.md`.

This list is intentionally conservative.  If a feature is not listed here, its
support must not be inferred from incidental Python behavior.

### 8.1 Path-aware scalar control flow

For a scalar validated comparison, AutoIA rigorously ranges the difference of
the two operands and returns a Python Boolean only when the complete active box
proves one outcome.  If the predicate is unresolved, the path controller may
subdivide the input box and, when necessary, enumerate both Boolean outcomes.
Each forced branch formula is independently certified over a box that contains
its feasible path domain, so the final finite union contains the original program
semantics.  Exact affine one-variable comparison boundaries are partitioned
directly when recognized.  See `PATH_AWARE_EXECUTION.md` for the proof argument,
branch budgets, and limitations.

---

## 9. Explicitly unsupported or not-yet-frozen behavior

The current candidate Version-1 contract does **not yet** include:

* elementary functions beyond the currently certified real `sqrt`, `exp`, `log`, `log10`,
  `sin`, `cos`, and `erf` subset (for example `tan`, inverse trig, hyperbolic functions,
  inverse trig, and hyperbolic functions);
* NumPy array behavior beyond the explicitly frozen Step-4 subset (for example
  uncertain max/min/comparison semantics, arbitrary `einsum`/`tensordot`,
  validated `out=`/masked reductions, and matrix spectral or general p-norms);
* uncertain array-valued Boolean masks, `np.where` masks, and uncertain indices;
* a general frozen contract for uncertainty-dependent loop termination;
* mutation/aliasing semantics beyond the currently exercised object-array cases;
* arbitrary compiled C/C++/Fortran routines;
* automatic recognition of an arbitrary hand-coded algorithm as a semantic
  primitive;
* nonlinear root-solver families beyond the documented bracketed `root_scalar` and square-system `fsolve` subsets, general ODE/DAE solvers, or arbitrary optimization routines as semantic primitives;
* validated uncertain eigenvalues/eigenvectors: `scipy.linalg.eigh` with AutoIA-dependent matrices is recognized and deliberately raises `UnsupportedOperationError` in this V1 build;
* `RegularGridInterpolator` methods other than `linear`, uncertain grid/table data, and derivative `nu=` query semantics;
* non-box/correlated input sets;
* full preservation of generator correlation through every semantic vector operation (the current linear solve soundly collapses and re-lifts generator inputs);
* generic validated-operation interception through a local/dynamic `import numpy`, `import math`, or `import scipy` executed inside a function body, through opaque/non-source-backed imported helpers, or through compiled extension code; source-backed pure-Python module helpers reached through the prepared namespace are supported for the documented semantic subset;
* semantic interception through unsupported runtime-import/compiled-code paths; AutoIA does not claim to control side effects or thread safety of arbitrary user model code itself.
* trusted GPU proof arithmetic: V1 MPS/CUDA helpers are proposal/assist only, and no GPU result is accepted as a rigorous certificate without independent CPU proof;

The active implementation does not monkeypatch `numpy.linalg.solve` or elementary
functions.  User Python functions are executed against a private shadow namespace
containing read-only NumPy/linalg/math proxies; the process-wide modules are not
mutated.  Common module-level NumPy/math/SciPy aliases, direct supported-operation aliases, same-global-namespace Python helpers, and reached source-backed pure-Python imported helpers are supported.  The remaining tracing boundary
is documented in `LOCAL_SEMANTIC_DISPATCH.md`.

The AutoIA-owned proof-policy/event runtime is now context-local via Python
`contextvars`; together with local semantic dispatch, the active AutoIA machinery
no longer relies on the previously known process-global mutable stacks or NumPy
monkeypatch.  This does not make arbitrary user engineering code thread-safe: user
models may still contain their own global mutation, non-thread-safe compiled
libraries, or other side effects outside AutoIA's contract.

The scalar-remainder correlation limitation is now addressed by an opt-in
shared-generator representation

    z(q) in P(q) + G epsilon + R,    epsilon in [-1,1]^s.

Here `q` contains the original centered physical uncertain inputs, while each
component of `epsilon` is an auxiliary nonphysical error variable with a stable
provenance id.  Equal ids denote the same error variable; unrelated errors are
never merged merely because their interval bounds coincide.  The vector matrix
`G` is obtained by stacking the scalar generator coefficients.  Operations that
leave the affine-generator family (for example `epsilon_i*epsilon_j` or
`q_i*epsilon_j`) are rigorously ranged and moved to a fresh generator or the
residual interval `R`.  See `SHARED_GENERATOR_REPRESENTATION.md`.

As of `0.18.0.dev2`, the semantic linear solve certifies generator-valued
matrix and right-hand-side inputs directly.  If the retained family is

    A(q, epsilon) in Ahat(q) + sum_s A_s epsilon_s + R_A,
    b(q, epsilon) in bhat(q) + sum_s b_s epsilon_s + R_b,

the solver may return a predictor

    X(q, epsilon) = P(q) + sum_s u_s epsilon_s

using the same incoming generator ids.  The actual stored residual `b-A*X` and
the defect `I-C*A` are rigorously bounded over the full joint physical/generator
domain before acceptance.  The remaining componentwise correction is still
introduced with fresh independent generator ids, because the current theorem
proves a correction box only.  See `STEP7B_GENERATOR_AWARE_SOLVE.md`.

---

## 10. Validated elementary-function contract

For a supported real elementary function `f` and a validated scalar argument `E`,
the returned validated value must enclose `f(z(q))` for every actual program value
`z(q)` represented by `E` on the input domain.  Domain conditions are proof
obligations: `log` requires a strictly positive certified argument range and `sqrt`
requires a nonnegative certified argument range.  Failure to prove the real domain
is a `CertificationError`, not silent clipping.

The polynomial representation uses a generic Taylor composition about a stored
binary64 center `c`.  Rigorous intervals contain each coefficient
`f^(n)(c)/n!`; coefficient uncertainty is propagated through the existing
point-polynomial-plus-remainder arithmetic, and an analytic Taylor remainder is
added.  The interval representation applies the trusted scalar interval kernel
directly.

The trusted scalar kernels do not rely on platform transcendental results being
correctly rounded.  See `ELEMENTARY_FUNCTIONS.md` for the exact constructions.

As of `0.22.0.dev1`, domain admissibility is a separate proof obligation rather
than a side effect of whichever structural range happened to be selected.  A
`ValidatedValue` may carry an optional *natural physical-box interval shadow*
`N_z`, defined by

\[
z(q)\in N_z\qquad\text{for every }q\in\mathcal Q_{\rm phys}.
\]

`N_z` is independent proof metadata.  It is neither a physical uncertain
coordinate `q_i` nor an auxiliary shared-error generator `epsilon_j`, and it does
not alter the stored `P(q)+R` or `P(q)+G epsilon+R` representation.

For a domain-sensitive operation AutoIA constructs a proof portfolio from
independently rigorous enclosures, including `N_z` when available and structural
termwise/monotonicity/Bernstein ranges.  Their intersection is rigorous on the
actual physical set.  Thus `sqrt(x)` for a physical input `x in [0,4]` can prove
`x >= 0` from the exact physical-box shadow even when the symmetric centered
polynomial proof box produces a tiny negative structural lower bound.  This is
**not clipping**: both operands of the intersection are independently valid on
`Q_phys`.

If a universal domain predicate is neither proved nor proved wholly invalid on
the current physical box, path-aware execution may subdivide the physical box
and retry.  Every child must certify.  Exhaustion of the domain-refinement budget
fails closed.  `sqrt` requires `z >= 0`; `log` and negative noninteger real powers
require `z > 0`; positive noninteger powers require `z >= 0`.

---

## 11. Semantic primitives versus generic execution

A semantic primitive is an implementation strategy, not a prerequisite for
correctness when equivalent code is expressible entirely with supported generic
operations.

For example, these two programs may both be certifiable:

```python
x = np.linalg.solve(A, b)
```

and a hand-coded 2-by-2 formula containing only `+`, `-`, `*`, and `/`.

The semantic primitive may provide substantial computational compression and
better numerical robustness.  Generic execution provides implementation-style
independence within the supported language.

AutoIA must not silently replace an approximate algorithm by a different exact
mathematical problem merely because it guesses the programmer's intent.  A
semantic replacement requires an explicit recognized contract.

### Current linear-solve semantic contract

The current primitive lives in `autoia.semantic.linear_solve`.  For the
polynomial path it certifies a parameter-dependent system `A(q)x(q)=b(q)` from
the residual of an actually stored polynomial predictor and a rigorous
contraction bound.  The retained `A(q)` and `b(q)` may now be general polynomials,
not only affine expressions.  Predictor and preconditioner proposals are allowed
to be ordinary floating-point heuristics because every candidate is independently
validated through trusted bounds on `|Cr|` and `|I-CA|`.  The established affine
path remains unchanged.  For the interval path AutoIA certifies the independently
boxed matrix/right-hand-side family.  See `STEP5_GENERAL_LINEAR_SOLVE.md` and
`SEMANTIC_LINEAR_SOLVE.md`.

---

## 12. Controller contract

Representation choice, range method, polynomial order, semantic-versus-generic
execution, and subdivision are **strategy decisions**.  Their quality affects
cost and sharpness, not validity.

Any controller action is admissible only if the resulting enclosure remains
independently certified.

Current theorem-backed range logic includes:

1. cheap termwise range;
2. rigorous monotonicity analysis;
3. if all active directions are rigorously monotone, no Bernstein calculation is
   needed for the polynomial range;
4. otherwise, monotone directions may be fixed at certified extremizing
   endpoints and Bernstein applied only to unresolved directions;
5. if the reduced Bernstein calculation exceeds budget, subdivision may split
   unresolved directions and repeat the same analysis locally.

Heuristic or learned controller decisions may be added later, but they must
remain outside the trusted correctness boundary.

---

## 13. Public package API proposed for Version 1

The candidate stable user-facing imports are:

```python
from autoia import (
    Interval,
    autoia,
    autoia_subdivide,
    autoia_adaptive_subdivide,
    RangePolicy,
    SubdivisionPolicy,
    RigorousBinary64Backend,
)
```

The primary novice call is

```python
result = autoia(model, parameter_intervals)
```

With no expert configuration keywords, this call uses the bounded automatic
configuration controller with a plain-polynomial/order-4 baseline. Strategy
selection is outside the trusted correctness boundary: every successful
configuration is independently certified and the returned numeric bounds are
the componentwise intersection of successful attempts.

Explicit `order=` or `representation=` without an explicit
`configuration_policy=` requests one fixed expert configuration. An expert can
request automatic configuration around a chosen baseline or the expensive
reference portfolio by passing an explicit `ConfigurationPolicy`. The package
facade keeps all strategy parameters keyword-only; the two positional arguments
are the model and uncertain input box.

A successful result must expose at least:

* `lower`: one lower endpoint per flattened scalar output;
* `upper`: one upper endpoint per flattened scalar output;
* `width`;
* `machine_rigorous`;
* the requested/returned representation information;
* semantic certificates where applicable;
* final range diagnostics;
* controller events needed to explain escalation decisions.


### 13.1 Public application-theorem boundary

Version `0.24.0.dev4` additionally exposes `autoia.application` for application
experts who possess a stronger mathematical bound than generic validated
execution can infer from source.  The stable concepts are `ApplicationBox`,
`CertifiedInterval`, `CertificationContext`, `application_theorem`,
`UpperBound`, `LowerBound`, `minimum_upper_bound`, and `maximum_lower_bound`.

The contract deliberately separates theorem truth from numerical certification.
For a theorem with `theorem_status="user_asserted"`, AutoIA does **not** claim to
prove the user's mathematical implication.  AutoIA does guarantee that
proof-critical numerical operations performed through `CertifiedInterval` and
`CertificationContext` use its machine-rigorous arithmetic contract.  Runtime-
checkable assumptions stated through `cert.require(...)` fail closed when false
and are recorded in the theorem result.

Application theorem code must not depend on internal polynomial, generator,
Bernstein, semantic-controller, or trusted-backend objects.  Ordinary numeric
arithmetic performed outside certified values is outside the numerical proof
contract; `cert.point(x)` is the explicit operation for interpreting a stored
binary64 constant as an exact real point.

If several independently valid upper bounds target the same mathematical
quantity, `minimum_upper_bound` returns their minimum. If several valid lower
bounds target the same quantity, `maximum_lower_bound` returns their maximum.
`CertifiedInterval.intersect(...)` / `cert.intersect(...)` likewise computes the
intersection of independent rigorous enclosures and fails closed if they are
disjoint. These are general proof-composition operations, not application-specific
semantics.

### 13.2 Historical fitted-GP integration boundary (extracted at `0.24.0.dev10`)

Versions `0.24.0.dev7` through `0.24.0.dev9` temporarily carried fitted-GP,
BoTorch/GPyTorch, radial-kernel, and downstream GP application-specific application code while the
public application boundary was being exercised. Step 20i (`0.24.0.dev10`)
removed those semantics from AutoIA and moved them to the independent downstream GP application
package. They are **not** part of the AutoIA Version-1 API or installed wheel.

The architectural lesson retained by AutoIA is general: application packages may
combine specialized analytic theorems with dependency-preserving
`cert.enclose(...)` routes and rigorously intersect independent enclosures, while
application-specific model semantics remain outside AutoIA. The dependency
direction is application -> AutoIA, never AutoIA -> application.

Historical migration modules are not part of the production package or stable API.

---

## 14. Reproducibility contract

Before Version 1.0, every public release candidate must pass a regression suite
that includes at minimum:

1. analytically solvable spring enclosure containment;
2. coupled affine-parametric linear-system regression;
3. nonlinear retained-polynomial linear-system regression, including a case in
   which the full-family midpoint preconditioner succeeds while the center
   preconditioner fails the contraction test;
4. semantic versus hand-coded 2-by-2 solution comparison;
5. interval-representation fallback behavior;
6. reciprocal/division zero-crossing refusal;
7. range-policy structural cases;
8. subdivision union containment;
9. validated elementary-function scalar high-precision audits and end-to-end
   polynomial/interval containment tests;
10. randomized diagnostic containment tests in addition to, but never replacing,
   proof-based guarantees.

Random sampling is diagnostic evidence only.  It is never a certificate.

---

## 15. Version-1 release meaning

Version 1.0 will **not** mean that AutoIA handles arbitrary Python or that its
controller/enclosures cannot later be improved.

It will mean:

> AutoIA is an installable, documented, fail-closed validated-execution system
> for a clearly specified and useful subset of engineering Python/NumPy.  For
> supported computations that return successfully, the documented machine-level
> inclusion guarantee is enforced by a small auditable trusted kernel.  The tool
> has useful dependency preservation, semantic acceleration for important
> operations, practical controller fallbacks, reproducible tests, and explicit
> diagnostics when it cannot certify a calculation.

This is the contract subsequent Steps 2 through 10 should preserve.


### 10.1 Error function and normal-distribution contract

`0.14.0.dev3` adds the real error function `erf` and the standard-normal
density/CDF as first-class validated unary operations.  The trusted definitions are

```text
erf(x) = 2/sqrt(pi) * integral_0^x exp(-t^2) dt,
phi(x) = exp(-x^2/2)/sqrt(2*pi),
Phi(x) = (1 + erf(x/sqrt(2)))/2.
```

The package additionally exposes `normal_pdf(x, mean, std)` and
`normal_cdf(x, mean, std)` for finite constant binary64 `mean` and strictly
positive constant binary64 `std`.  Uncertain distribution parameters remain
unsupported.  The documented `scipy.stats.norm` / `scipy.stats.Normal` PDF/CDF
and `scipy.special.erf` / `ndtr` compatibility surfaces are part of the supported
language when SciPy is already available to the user model.

## Optional SciPy compatibility surface

When user code has imported SciPy, AutoIA may locally recognize the supported normal-distribution/error-function APIs documented in `NORMAL_API_COMPATIBILITY.md`, the documented bracketed `root_scalar` and square-system `fsolve` semantics, and fixed-data linear `RegularGridInterpolator`. SciPy is not trusted proof code and is not an AutoIA dependency. For implicit solves SciPy supplies only nominal proposals; AutoIA independently verifies the root/system theorem. For interpolation AutoIA evaluates the exact fixed-data multilinear formula on path-selected cells. `scipy.linalg.eigh` is recognized as an explicit unsupported V1 boundary when matrices depend on AutoIA inputs. Unsupported SciPy forms fail closed rather than being evaluated numerically on validated objects.


## Step-9 performance contract

Performance implementations may change the order in which already-rigorous
interval operations are composed, so certified endpoints need not be bitwise
identical to the scalar oracle. They must independently enclose the same exact-real
quantity. The scalar trusted reduction remains selectable as the reference oracle.
Proposal-only batching/caching may alter floating predictor coefficients, but every
semantic certificate remains responsible for validating the actual stored proposal.
No performance cache may bypass construction of a fresh audited rigorous backend.

`0.20.0.dev2` extends this contract to point×interval and interval×interval
reductions. For finite binary64 endpoints, the vectorized path must first construct
componentwise outward product intervals and then combine them only by outward
pairwise interval addition. Batched matrix kernels are merely simultaneous
execution of independent rigorous dot trees; batching does not authorize a BLAS
result or a single post-hoc error bound. Unsupported extended-endpoint cases fall
back to the scalar trusted semantics. Short reductions retain the scalar reference
path under `reduction_method="auto"`.


## Prepared and repeated-query execution (`0.21.0.dev1`)

`prepare(model)` creates a lazy automatic prepared session with the same novice
configuration semantics as `autoia(model, box)`. Explicit `order=` or
`representation=` without an explicit configuration policy creates one fixed
`PreparedAutoIA` expert session. Automatic prepared sessions create alternate
configuration sessions lazily, only after a queried box triggers them, and then
reuse those sessions and their certificate caches. Calling `session.certify(box)`
has the same inclusion contract as calling the corresponding one-shot API.
Preparation, configuration selection, caching, chunking, and parallel scheduling
are performance/sharpness mechanisms only and do not weaken the certificate
requirement.

If a cached parent box `D` contains a new query box `Q`, AutoIA may answer `Q`
without re-executing the engineering model only when it can derive a sound
certificate from the stored parent certificate. For polynomial and
shared-generator outputs, the stored polynomial is rigorously recentered from
the parent binary64 center to the query binary64 center; all coordinate-transform
rounding error is enclosed by trusted arithmetic and added to the new remainder.
The symmetric centered query domain must itself be rigorously contained in the
parent physical box. If that proof fails, AutoIA performs a fresh certification.

`certify_many`, `enclose_many`, and their streaming/compact-bound variants certify each query box independently. `workers="auto"` resolves an available CPU worker count, and `parallel="auto"` chooses serial execution for one worker and otherwise prefers process execution when the model can be serialized. Optional `cloudpickle` may serialize local functions/closures for spawn workers; serialization is not proof logic. Thread/process parallelism changes scheduling only. In process mode each worker constructs its own prepared model and independently audited trusted backend. `ManyBoxExecutionInfo` records resolved scheduling metadata but is never used to establish inclusion. No certificate is accepted merely because another worker or earlier query returned a similar result.

Optional PyTorch MPS/CUDA helpers may evaluate non-rigorous proposal values at many points or box centers. Their result objects carry `machine_rigorous=False`. The Version-1 machine-rigorous contract remains tied to independent CPU binary64 verification. AutoIA does not claim to automatically translate arbitrary engineering Python into a Torch GPU graph.


## Transparent fitted-GP semantic contract added in 1.0.2

A supported fitted scalar exact-GP object may be recognized during AutoIA local
semantic execution. The normal user workflow remains ordinary GPyTorch/BoTorch
source; a manual freeze adapter is not required. Recognition is limited to the
documented RBF/Matern exact-GP subset, common affine transforms, and the narrow
query/posterior construction surface. Unsupported structures fail closed.

Fitting, autograd, training convergence, and accelerator arithmetic are outside
the proof claim. The frozen CPU binary64 fitted state is treated as fixed stored
data. Proof-critical arithmetic involving uncertain query inputs is performed by
AutoIA's validated backend.
