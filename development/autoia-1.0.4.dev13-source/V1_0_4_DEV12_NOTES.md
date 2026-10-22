# AutoIA 1.0.4.dev12 notes

## Per-box failure reporting for prepared many-box execution

Version 1.0.4.dev12 adds an opt-in `on_error="report"` mode to prepared
many-box APIs. The default remains `on_error="raise"`, so existing code keeps
the previous all-or-nothing exception behavior.

The new mode is intended for branch-and-bound and other adaptive search methods
where failure to certify one box is a normal refinement signal and must not
discard successful sibling certificates.

For compact bounds:

```python
lower, upper, ok = prepared.enclose_many_bounds(
    boxes,
    on_error="report",
)
```

`ok` is a Boolean vector in input order. Rows with `ok[i] == False` contain
`NaN` placeholders in `lower` and `upper`; they are not certificates and must
not be used as bounds.

For detailed failure information:

```python
items = prepared.certify_many(boxes, on_error="report")
for item in items:
    if item.ok:
        result = item.result
    else:
        print(item.failure.error_code, item.failure.message)
```

The detailed records are `ManyBoxCertificationReport` objects. Failures include
the public AutoIA exception type, stable error code, message, context, and
suggestions. `InternalAutoIAError` is never converted to a report; it still
raises immediately because it denotes a software invariant failure rather than
an ordinary box-certification outcome.

The same behavior is available in `certify_many_bounds`, `enclose_many`,
`enclose_many_bounds`, `iter_certify_many`, `iter_bounds_many`, and the
persistent `PreparedManyBoxExecutor`. Process workers catch/report failures per
box, not per chunk, so one failed box does not cancel successful boxes assigned
to the same worker chunk.

Example persistent B&B-style use:

```python
prepared = prepare(node_model)
with prepared.manybox_executor(
    workers=2,
    parallel="process",
    chunk_size=1,
    use_cache=False,
    on_error="report",
) as pool:
    lower, upper, ok = pool.enclose_many_bounds(children)
```

## Optional cloudpickle process extra

Spawn-based process execution can use ordinary pickle for module-level models.
Closure/local-function models require optional `cloudpickle`. dev12 exposes this
as an install extra:

```bash
python -m pip install "autoia[process]"
```

`cloudpickle` remains optional and is not part of the core proof dependency set.
It is transport infrastructure only; every worker still constructs its own
trusted CPU proof backend.

## Validation

The regression suite was run in independent groups:

- 427 named tests passed;
- 3 optional live-GP tests skipped;
- 0 failures;
- 2,451 / 2,451 parameterized subtests passed.
