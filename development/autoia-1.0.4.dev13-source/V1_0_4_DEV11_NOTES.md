# AutoIA 1.0.4.dev11 notes

## Captured-object callable attributes

dev10 prepared ordinary functions captured in closure cells and class-defined
methods of captured source-level Python objects.  One normal Python pattern still
bypassed the shadow namespace: a pure-Python function stored directly as an
instance attribute, for example `problem.g = response` followed by
`problem.g(x)`.  Python returns such an attribute as a plain function rather than
a bound method.

dev11 recognizes that case locally.  The function-valued attribute receives the
same lazy private semantic preparation as a directly captured function.  The
source object, NumPy, and SciPy are not monkey-patched.  Regression tests cover
both nested `fsolve` and `linprog` calls through this path.

## Persistent prepared many-box execution

`PreparedAutoIA.certify_many(..., parallel="process")` intentionally owns a
finite process pool.  Repeating that call inside branch-and-bound therefore
repeats process spawn and worker model preparation, which can dominate runtime on
spawn-based platforms such as macOS.

dev11 adds the explicit context-managed `PreparedManyBoxExecutor`, constructed
with `prepared.manybox_executor(...)`.  In process mode:

- the model is serialized once when the executor starts;
- each worker creates one prepared AutoIA session and audited CPU proof backend;
- the same processes persist across successive node batches;
- each submitted box is still independently certified;
- caller-visible ordering is preserved.

The persistent executor is execution infrastructure, not a new proof theorem or
global optimizer.  It does not change certified endpoints, LP weak-duality
semantics, one-sided `[L,+inf]` results, `A_eq` handling, or the trusted backend.

For dynamic B&B, a typical pattern is:

```python
prepared = prepare(node_model)
with prepared.manybox_executor(
    workers=4,
    parallel="process",
    chunk_size=1,
    use_cache=False,
) as pool:
    lower, upper = pool.enclose_many_bounds(current_node_batch)
```

`workers="auto"` is also available.  Applications using internally threaded
native libraries should benchmark a few worker counts to avoid oversubscription.

## LP sharpness is unchanged

The active-set-switch LP remains deliberately governed by fixed primal/dual
witnesses in this checkpoint.  dev11 does **not** add a parametric LP witness.
Subdivision by the outer optimizer continues to tighten such boxes while keeping
physical-domain refinement under caller control.

## Validation

The dev11 regression suite adds two captured-attribute dispatch tests and two
persistent-executor tests.  Full validation was run in independent test groups:

- 422 named tests passed;
- 3 optional live-GP tests skipped;
- 0 failures;
- 2,451 / 2,451 parameterized subtests passed.

An installed-wheel acceptance run also verified direct `obj.g(x)` dispatch and
two successive persistent process batches against a bit-for-bit identical
prepared-serial reference.
