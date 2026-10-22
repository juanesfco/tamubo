# AutoIA 1.0.4.dev10 notes

## Closure-variable semantic dispatch

`1.0.4.dev10` fixes a usability defect in the local shadow dispatcher. A supported
SciPy semantic reached through a pure-Python function captured in a closure cell,
or through a method of a captured source-level user object, is now prepared locally.
This covers common factory/object patterns such as `fn(x)` and `prob.g(x)` without
publishing the helper as a module global. The source object and NumPy/SciPy modules
remain unmodified. Regression tests cover captured functions, captured objects,
nested `self` method calls, `fsolve`, and `linprog`.

## `linprog` equality constraints

`A_eq` / `b_eq` are now supported for the certified `res.fun` semantic. AutoIA
reduces each equality exactly to the paired inequalities `A_eq z <= b_eq` and
`-A_eq z <= -b_eq`; the original dependency-bearing coefficient objects are used
in both copies. This lets the existing weak-duality and universal-primal-witness
proofs remain authoritative. Parametric equalities with no fixed primal witness may
return a rigorous lower-only `[L,+inf]` result, exactly as in dev9.

## Input endpoint note

`Interval.enclosing(...)` intentionally remains an exact-decimal/rational constructor.
NumPy floating scalars are therefore not accepted as exact specifications. If a
NumPy scalar is intended as an exact stored binary64 endpoint, use `Interval(float(v),
float(v))`; if the shortest decimal spelling is intended as an exact decimal, use
`Interval.enclosing(repr(float(v)), repr(float(v)))`. This preserves the established
input-provenance contract rather than silently changing it.

## Validation

The complete named regression suite was run in independent chunks:

- 418 passed;
- 3 optional live-GP tests skipped;
- 0 failures;
- 2,451 / 2,451 parameterized subtests passed.
