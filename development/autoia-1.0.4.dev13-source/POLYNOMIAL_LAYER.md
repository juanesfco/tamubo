# AutoIA polynomial layer — Step 1 extraction (`0.12.0.dev4`)

## Purpose

The polynomial layer is the first proof-critical layer above `autoia.trusted`.
It implements the point-polynomial dependency representation used by the mature
AutoIA polynomial enclosure backend, but it does **not** define the universal
`ValidatedValue` contract.  It is an internal implementation layer and is not
yet part of the frozen AutoIA 1.0 public API.

A stored point polynomial has the form

\[
P(q)=\sum_{\alpha} c_\alpha q^\alpha,
\]

where:

- `q = (q_1,...,q_m)` are the original centered uncertain inputs;
- `alpha = (alpha_1,...,alpha_m)` is a multi-index of nonnegative integers;
- `q^alpha = prod_i q_i^(alpha_i)`;
- each stored coefficient `c_alpha` is a binary64 value interpreted as an exact
  real point value.

Errors caused by computing a new stored binary64 coefficient are **not** silently
ignored.  They are bounded rigorously using `autoia.trusted` and moved into the
scalar interval remainder owned by the enclosing representation.

## Module boundary

```
autoia.trusted
      ^
      |
autoia.polynomial.algebra
      ^
      |
autoia.polynomial.ranging
```

`autoia.polynomial` must not import:

- the legacy/reference core;
- enclosure representation classes;
- semantic primitives;
- range/subdivision controllers;
- the public execution API.

This dependency direction is enforced by tests.

## `algebra.py`

`autoia.polynomial.algebra` contains:

- multi-index helpers;
- polynomial cleaning;
- rigorous monomial magnitude/range helpers used by coefficient-error bounds;
- rigorous point-polynomial addition;
- rigorous truncated point-polynomial multiplication;
- proposal-only reciprocal-polynomial construction.

For addition, if the exact coefficient is enclosed by `C_alpha` and the stored
binary64 proposal coefficient is `c_hat_alpha`, the coefficient error

\[
E_\alpha = C_\alpha - [\hat c_\alpha,\hat c_\alpha]
\]

is rigorously ranged over the input box and added to the enclosure remainder.

For truncated multiplication, retained coefficient rounding error and the full
range of discarded high-order terms are both returned as an interval remainder.

The reciprocal predictor remains proposal-only.  Its rigor is established later
by the residual certificate in the enclosure layer, not by `algebra.py`.

## `ranging.py`

`autoia.polynomial.ranging` contains the rigorous polynomial-only range
algorithms:

- termwise range;
- rigorous derivative termwise range;
- monotonicity range;
- full tensor Bernstein range;
- endpoint-fixed Bernstein range;
- monotonicity-reduced Bernstein range;
- Bernstein coefficient-count estimators;
- the legacy flat `range_polynomial(..., method=...)` dispatcher.

The range-policy/controller is deliberately **not** in this module.  The
polynomial layer answers questions such as "what is the rigorous Bernstein
range?"; a higher layer decides whether Bernstein is worth paying for.

## Regression policy for this migration

`tests/support/oracle_v11.py` retains the verbatim pre-refactor oracle for
development-time regression only.

For the `dev3 -> dev4` structural move we require, on the tested operations:

1. identical stored coefficient dictionaries;
2. identical binary64 interval endpoint bits for coefficient/truncation errors;
3. identical binary64 endpoint bits for polynomial range methods;
4. identical monotonicity signs and Bernstein coefficient-count metadata;
5. unchanged end-to-end Version 0.11 model regressions.

These bit-for-bit requirements are migration safeguards, not permanent
requirements on future improved algorithms.

## Current non-goals

This extraction intentionally does not:

- change polynomial order selection;
- introduce sparse-basis optimizations;
- change Bernstein algorithms;
- change monotonicity logic;
- add new range methods;
- change reciprocal prediction;
- change representation classes or controller policies.

Those are algorithmic changes and should be reviewed separately from this
behavior-preserving refactor.


## Physical set versus symmetric proof box (`0.22.0.dev1`)

The centered polynomial range routines accept half-widths `h` and rigorously
range over the symmetric box

\[
Q=\prod_i[-h_i,h_i].
\]

For binary64 centering this can be a strict over-domain of the actual centered
physical set

\[
Q_{\rm phys}=\{p-p_c:p\in P\}\subseteq Q.
\]

That stronger structural range remains useful and unchanged.  The Step-10a
domain prover may combine it with an independent enclosure valid directly on
`Q_phys` when proving a mathematical-domain predicate.  No physical-domain
metadata or auxiliary `epsilon` generator is added to this low-level polynomial
module; the distinction is maintained by the enclosure/runtime layers above it.
