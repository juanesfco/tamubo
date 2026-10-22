# AutoIA 1.0.4.dev8 notes

## New semantic primitive: `scipy.optimize.linprog`

AutoIA now recognizes ordinary `scipy.optimize.linprog(...)` calls during
validated execution and certifies the LP optimal value returned as `res.fun`.

The supported Version-1 LP is

`min c(q)^T z` subject to `A_ub(q) z <= b_ub(q)` and constant variable bounds.

SciPy/HiGHS is proposal-only. The lower endpoint is independently certified by
weak-duality witnesses evaluated with AutoIA's dependency-carrying arithmetic.
The upper endpoint is independently certified from fixed primal candidates
whose feasibility is proved for every physical input in the current box.

The primal candidate portfolio includes a conservative flat-interval robust LP
proposal plus ordinary numerical scenario proposals and simple anchors. None of
those proposal computations is trusted for proof.

## Scope

- certified: `res.fun`;
- deliberately unsupported: `res.x`;
- supported constraints: `A_ub`, `b_ub`, constant variable bounds;
- common default bounds `0 <= z < +inf` are supported when reduced-cost signs
  permit a finite weak-duality certificate;
- deferred/fail-closed: `A_eq`, `b_eq`, mixed-integer `integrality`, callbacks,
  parametric optimizer values.

A box containing an LP active-set transition may produce a looser certificate.
Version 1 deliberately does not hide physical-domain subdivision inside the LP
semantic merely to improve tightness.

See `LINEAR_PROGRAM_SEMANTICS.md` and
`examples/38_scipy_linprog_optimal_value.py`.

## Validation

The new focused LP suite contains 10 tests covering:

- constant-data optimal-value certification;
- uncertain objective coefficients;
- shared-dependency cancellation through `c`, `A`, and `b`;
- a sound active-set-switch regression;
- infeasible-for-part-of-box fail-closed behavior;
- unbounded fail-closed behavior;
- `res.x` fail-closed behavior;
- equality-constraint deferral;
- module-level `scipy.optimize.linprog` dispatch;
- SciPy's default nonnegative, unbounded-above variable bounds.

Existing named tests continue to pass. The complete test set is 413 passed,
3 optional live-GP skips, 0 failures, with 2,451 pytest subtests passed.
