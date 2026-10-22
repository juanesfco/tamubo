# AutoIA 1.0.4.dev13 notes

## Lifted McCormick/RLT lower bounds for parametric `linprog`

Version `1.0.4.dev13` adds a second rigorous lower-bound portfolio member for
ordinary `scipy.optimize.linprog(...).fun` semantics when LP coefficients retain
polynomial dependence on the current physical inputs.

For the lower endpoint, AutoIA views the physical inputs and LP decision variables
jointly.  Products between physical monomials and LP decision variables are lifted
into shared auxiliary variables with rigorous McCormick envelopes.  Higher-order
physical monomials are represented by a shared recursive lift rather than by
independent coefficient intervals.  Nonzero unstructured polynomial remainders
are outer-relaxed by bounded auxiliary variables.

The enlarged constant-data LP is still not trusted.  HiGHS supplies proposal dual
marginals only; AutoIA independently recomputes a weak-duality lower certificate
with the trusted CPU binary64 backend.  The new result is intersected as a theorem
portfolio with the existing fixed-dual lower certificate, so the new mechanism
cannot weaken an already established lower bound.

## Cost-capped certified OBBT

When the initial lifted relaxation already improves the fixed-dual lower bound,
dev13 can perform up to two optimization-based bound-tightening rounds on original
LP decision variables that participate in lifted products.  Every tightened bound
is itself certified over the current relaxation before the McCormick envelopes are
rebuilt.  Automatic OBBT is skipped when more than eight decision variables require
product tightening.  A zero decision-variable lower bound is left unchanged as a
cost heuristic; this can only reduce optional tightening work, not proof validity.

The lift requires finite bounds for factors in McCormick products.  When a required
LP decision-variable bound is infinite, AutoIA records the reason and falls back to
the pre-existing fixed-dual theorem.

## T3 active-set acceptance

For

```text
min -z
s.t. x*z <= 1
     z <= x
     0 <= z <= 10
x in [0.5,2]
```

with exact value `V(x)=-min(1/x,x)` and exact range `[-1,-0.5]`, dev13 gives:

```text
fixed-dual lower            about -2.000000
initial McCormick/RLT       about -1.750000
OBBT/rebuild round 1        about -1.200000
OBBT/rebuild round 2        about -1.062500
final AutoIA enclosure      about [-1.062500,-0.500000]
```

No hidden physical-domain subdivision is used to obtain this improvement.
Caller-controlled subdivision remains available and continues converging through
the active-set switch.

## Diagnostics

`scipy.optimize.linprog` certificates now include:

```text
fixed_dual_lower_bound
rlt_lower_bound
selected_lower_bound_method
rlt_status
rlt_reason
rlt_initial_lower
rlt_final_lower
rlt_obbt_rounds
rlt_lifted_variable_count
rlt_lifted_row_count
rlt_monomial_count
rlt_product_count
rlt_remainder_variable_count
rlt_relevant_z_count
```

## Validation

The dev13 focused LP suite contains 16 tests, including analytic T3 improvement,
a nonlinear polynomial constraint-matrix coefficient, infinite-factor fallback,
and the automatic OBBT cost gate.

The complete regression was run in deterministic file groups:

- 431 named tests passed;
- 3 optional live-GP tests skipped;
- 0 failures;
- 2,451 / 2,451 parameterized subtests passed.

A separate seeded 40-case affine-parametric LP sweep also checked dense pointwise
SciPy values against the dev13 certified enclosures; all sampled values were
contained.  The dense sweep is diagnostic only and is not part of the proof claim.
