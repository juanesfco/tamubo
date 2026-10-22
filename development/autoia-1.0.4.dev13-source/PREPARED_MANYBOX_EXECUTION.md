# Prepared and many-box execution

This document defines the Step-9c repeated-query architecture introduced in
AutoIA `0.21.0.dev1`.

## 1. Motivation

Let an engineering program be denoted by

\[
y = \mathcal P(p),
\]

where:

- \(\mathcal P\) is the fixed deterministic engineering program;
- \(p=(p_1,\ldots,p_m)\) is the vector of \(m\) uncertain physical inputs;
- each query supplies a Cartesian interval box
  \[
  D = D_1\times\cdots\times D_m,
  \qquad
  D_i=[\underline p_i,\overline p_i].
  \]

The one-box API `autoia(model, box)` remains correct, but a user may need to
certify the *same* program on thousands or millions of different boxes.  Step 9c
separates static program preparation from dynamic box certification and adds
sound reuse of earlier certificates.

## 2. Prepared session API

```python
from autoia import prepare, Interval

session = prepare(
    model,
    order=4,
    representation="polynomial",
)

r1 = session.certify([
    Interval(-1.0, 1.0),
    Interval(0.0, 2.0),
])

r2 = session.certify([
    Interval(-0.2, 0.3),
    Interval(0.8, 1.1),
])
```

A `PreparedAutoIA` object retains:

1. one locally prepared NumPy/SciPy semantic-dispatch clone of the engineering
   callable;
2. one audited `RigorousBinary64Backend`;
3. the fixed order, representation, range policy, and branch policy;
4. the process-level polynomial combinatorial caches already present from
   Steps 8--9;
5. a bounded session-local certificate cache.

No global NumPy, SciPy, or Python function is monkeypatched.

## 3. Prepared semantic dispatch is context-local

A prepared model cannot bind `np.linalg.solve` permanently to one solve context,
because every certification creates a new semantic solve context.  Instead the
one-time shadow clone contains a stable solve hook.  During each execution a
`ContextVar` points that hook to the active `GenericSolveContext`.

Therefore:

- the engineering function is cloned once;
- the semantic solve context remains dynamic per certification;
- nested and concurrent thread executions do not communicate through global
  mutable state.

This preparation is an efficiency mechanism only.  It is not part of the proof.

## 4. Exact certificate cache

For a query box \(D\), define the exact binary64 endpoint key

\[
K(D)
=
\bigl((\underline p_1,\overline p_1),\ldots,
      (\underline p_m,\overline p_m)\bigr).
\]

If the same endpoint key has already been certified in the session, the previous
certificate is returned directly.  The engineering program is not executed.

Correctness is immediate because the mathematical query is identical.

## 5. Restricting a previously certified polynomial certificate

The more important reuse case is a new box \(Q\) contained in an already
certified parent box \(D\):

\[
Q\subseteq D.
\]

Suppose an output was certified on \(D\) as

\[
z(q)\in P(q)+R,
\]

where:

- \(q=p-c_D\) is the old centered physical coordinate;
- \(c_D\in\mathbb R^m\) is the stored binary64 center of the parent box;
- \(P(q)\) is the stored point-coefficient polynomial;
- \(R\) is its rigorous interval remainder.

For the new query box, let

\[
r=p-c_Q,
\]

where \(c_Q\) is the new stored binary64 center.  Then, as exact real numbers,

\[
q = r + s,
\qquad
s = c_Q-c_D.
\]

Therefore

\[
P(q)=P(r+s).
\]

AutoIA algebraically expands this polynomial about the new center.  For a
multi-index \(\alpha=(\alpha_1,\ldots,\alpha_m)\), define

\[
r^\alpha=\prod_{i=1}^m r_i^{\alpha_i}.
\]

Each old monomial obeys

\[
(r+s)^\alpha
=
\prod_{i=1}^m
\left(
\sum_{\beta_i=0}^{\alpha_i}
\binom{\alpha_i}{\beta_i}
 s_i^{\alpha_i-\beta_i} r_i^{\beta_i}
\right).
\]

This transform is performed with an ordinary binary64 point-coefficient
proposal, but every exact-real transformed coefficient is independently
interval-enclosed with the trusted backend.  The difference between that
rigorous coefficient interval and the stored proposal coefficient is ranged on
the new box and added to the certified remainder.

Thus the recentering itself is machine-rigorous.

### 5.1 The center difference is also certified

The stored binary64 subtraction

```python
shift = new_center - old_center
```

is only a proposal for the exact real difference of the two stored center
values.  AutoIA therefore also computes

\[
S_i
\supseteq
c_{Q,i}-c_{D,i}
\]

with trusted outward subtraction and uses \(S_i\), rather than treating the
rounded `shift` as exact, when proving the transformed coefficients.

## 6. Boundary safety

AutoIA's polynomial representation uses a symmetric centered domain

\[
r_i\in[-h_i,h_i],
\]

where \(h_i\) is an outward-computed half-width.  This symmetric box can extend
a few ulps beyond the exact input endpoints.

A cached parent certificate is reused by polynomial recentering only if the new
symmetric physical envelope

\[
c_{Q,i}+[-h_i,h_i]
\]

is itself rigorously contained in the certified parent physical interval
\(D_i\) for every input coordinate \(i\).

If this cannot be proved, the sharp recentering reuse is refused and the normal
fresh certification path is used.  AutoIA never clips the new domain to make a
cache hit succeed.

The interval-only representation does not need this restriction because its
cached output interval is already valid for every subset of the parent input
box.

## 7. Shared-generator certificates

For the generator representation,

\[
z(q)\in P(q)+G\epsilon+R,
\qquad
\epsilon\in[-1,1]^s,
\]

only the physical polynomial coordinates \(q\) are recentered.  Existing
columns of the constant generator matrix \(G\) retain their generator IDs and
coefficients.

Any new binary64 recentering error receives conservative new error provenance;
AutoIA does not invent correlation between independent coefficient-transform
errors.

## 8. What is and is not reused from a semantic solve

If a previously returned solution has already been certified as a polynomial or
shared-generator output, restricting that *output certificate* to a subbox does
not require repeating the original semantic solve proof.  The old proof was
valid on the parent domain, so it remains valid on every subset.

The stored solve certificate metadata is retained for provenance, while the
output polynomial is recentered and re-ranged on the new query box.

This is different from warm-starting a new solve on an unrelated box.  For a
fresh box AutoIA still recertifies the residual and contraction condition.

## 9. Bounded cache policy

The session cache is an in-memory bounded least-recently-used cache.  Exact
lookup is constant-time.  Containment lookup searches the bounded entries and
selects the smallest containing certificate.

The default maximum is 256 entries:

```python
session = prepare(model, cache_max_entries=256)
```

This is intentionally an initial policy rather than a permanent spatial-index
design.  The public session API permits a later R-tree/k-d-tree or adaptive-cell
index without changing user code.

The cache is persistent for the lifetime of the `PreparedAutoIA` object.  Step
9c does **not** serialize trusted certificates to disk across Python processes or
machines.

## 10. Many-box execution

A finite collection can be certified with

```python
results = session.certify_many(
    boxes,
    parallel="process",
    workers=4,
    chunk_size=64,
)
```

`boxes` may be an iterable of interval boxes or a numeric array with shape

\[
(N,m,2),
\]

where:

- \(N\) is the number of query boxes;
- \(m\) is the number of uncertain physical inputs;
- the final axis stores lower and upper endpoints.

Input ordering is preserved.

## 11. CPU parallel modes

### Serial

```python
parallel="serial"
```

uses one prepared session.

### Threads

```python
parallel="thread"
```

shares one prepared model, trusted backend, and certificate cache.  The dynamic
range, branch, and solve states are all context-local.  This mode is safe, but
current AutoIA still performs enough Python-object work that threads need not be
faster.

### Processes

```python
parallel="process"
```

creates one independently prepared and independently audited session per worker
process.  A worker persists across assigned chunks, so it does not prepare the
model once per box.

Process mode requires a pickleable engineering callable, normally a module-level
function.  Local closures should use serial or thread mode.

The proof guarantee is unchanged: every box is independently certified by its
worker.  Parallelism changes scheduling, not mathematics.

## 12. Streaming millions of boxes

`certify_many()` returns all result objects and is therefore inappropriate when
\(N\) is extremely large.

Use

```python
for result in session.iter_certify_many(
    boxes,
    parallel="process",
    workers=8,
    chunk_size=256,
):
    consume(result)
```

The iterator keeps one executor alive and only a bounded number of chunks in
flight.

If only bounds are needed, use

```python
for lower, upper in session.iter_bounds_many(boxes, ...):
    consume(lower, upper)
```

which avoids retaining full `AutoIAResult` objects after each yield.

For a finite moderate-size collection,

```python
lower, upper = session.certify_many_bounds(boxes, ...)
```

returns stacked arrays with shape \((N,n_y)\), where \(n_y\) is the number of
scalar outputs.

## 13. What Step 9c does not yet do

Step 9c is the *execution architecture* for repeated queries.  It is not yet a
true single-kernel batched polynomial representation with a leading box axis.
In particular:

1. unrelated fresh boxes are still executed as independent AutoIA
   certifications;
2. process parallelism is currently the useful coarse CPU parallel mechanism;
3. polynomial coefficient buffers are not yet stored as numeric
   \((N,\text{terms},\ldots)\) arrays;
4. GPU execution is not yet implemented;
5. pathwise parent certificates currently receive exact-cache reuse, but sharp
   geometric restriction of arbitrary piecewise branch domains is deferred;
6. the bounded containment cache uses a simple scan rather than a large spatial
   index.

These choices are deliberate.  They freeze a many-box public API before the
internal numeric-buffer/GPU refactor.

## 14. Why this architecture is GPU-compatible

The public distinction

\[
\text{prepare once} \quad + \quad \text{certify many boxes}
\]

is the same distinction a future GPU implementation needs.  The CPU process
backend can later be joined by a GPU execution backend without changing the
user-facing workload model.

A future device implementation should replace Python object arrays in the hot
path by numeric structure-of-arrays buffers, while retaining the current
`ValidatedValue` objects as tracing/front-end objects.  GPU proposal kernels may
arrive before a trusted GPU proof backend; all proposal results can still be
validated by a CPU proof kernel.


## Step-10a physical-domain proof state (`0.22.0.dev1`)

A **fresh** prepared certification first uses the ordinary fast path without a
natural interval shadow.  If a structural mathematical-domain proof is
unresolved, the path/domain controller replays that same query once with each
uncertain input seeded by its actual physical interval.  Arithmetic then
propagates the independent natural shadow as needed.  It is proof metadata only:
it is neither a physical polynomial coordinate `q` nor an auxiliary shared-error
generator `epsilon`.

Domain-sensitive operations use the same proof portfolio, on-demand replay, and
physical-box refinement as one-shot `autoia(...)`.  This does not introduce
session-global mutable proof state; the active domain runtime is context-local.

An exact cache hit or a sound restriction of an eligible previously certified
parent certificate does not re-execute the engineering model and therefore does
not replay the original elementary-function domain proof.  This is sound because
the cached certificate already encloses the program output on the parent physical
domain, and a restricted query domain is a subset of that certified domain.

## dev11: persistent executor for dynamic box queues

Finite `certify_many(..., parallel="process")` calls own their process pool.  For
dynamic algorithms that repeatedly submit new batches, use
`prepared.manybox_executor(...)` to amortize process spawn and model preparation:

```python
prepared = prepare(model)
with prepared.manybox_executor(
    workers="auto", parallel="auto", chunk_size=1, use_cache=False,
) as pool:
    lower, upper = pool.enclose_many_bounds(batch)
    lower2, upper2 = pool.enclose_many_bounds(next_batch)
```

In process mode each persistent worker owns an independent prepared session and
audited trusted backend.  The executor is explicit and context-managed so process
lifetime is under application control.  Scheduling and caching affect cost only;
each box retains the same independent certification contract.

## dev12: per-box failure reporting for adaptive search

Prepared many-box APIs now accept `on_error="report"`. The default remains
`"raise"` for backward compatibility.

A branch-and-bound search commonly submits sibling boxes together even though
one child may be temporarily uncertifiable. In report mode the failure is
localized to that child rather than aborting the complete batch.

```python
lower, upper, ok = session.enclose_many_bounds(
    boxes,
    workers=2,
    parallel="process",
    on_error="report",
)
```

The arrays retain input order. Successful rows are ordinary rigorous AutoIA
bounds. Failed rows contain `NaN`; `ok[i] == False` is authoritative and means
no bound was returned for that row. `NaN` is deliberately not a vacuous
certificate.

Detailed status is available from:

```python
items = session.certify_many(boxes, on_error="report")
```

Each item is a `ManyBoxCertificationReport`. Successful items carry `.result`.
Failed items carry `.failure`, including `error_type`, stable `error_code`,
`message`, `context`, and `suggestions`.

Streaming report mode uses `ManyBoxBoundsItemReport` for `iter_bounds_many`, so
failure of one streamed box does not terminate later boxes.

`InternalAutoIAError` and unexpected non-public failures still propagate. The
report mode is not permission to treat every failure as subdivision-fixable;
callers should inspect the failure code and choose their search action.

### Process transport for closure models

`parallel="process"` uses spawn workers. Module-level pickleable models need no
extra package, but closures/local functions require optional `cloudpickle`.
Install the process extra when using that pattern:

```bash
python -m pip install "autoia[process]"
```
