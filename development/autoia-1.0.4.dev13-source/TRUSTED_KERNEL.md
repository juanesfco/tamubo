# AutoIA trusted numerical kernel

**Checkpoint:** `0.22.0.dev2`  
**Status:** audited scalar reference kernel plus proof-preserving vectorized/batched reduction paths and trusted elementary kernels.

## Purpose

`autoia/trusted/` is the smallest numerical base on which
`machine_rigorous=True` depends.  Higher-level polynomial, enclosure, semantic,
and controller layers depend downward on this package; the trusted package does
not import them.

The trusted reference implementation favors auditability over speed.  It uses
explicit scalar interval operations and scalar-loop reductions rather than
trusting a BLAS reduction and widening only once afterward.

## Modules

### `environment.py`

Audits the observable binary64 environment before a rigorous backend is used.
It verifies the float format, current round-to-nearest behavior, NumPy
`nextafter`, gradual underflow/subnormal behavior, and ordinary binary64
intermediate rounding.  Failure is fail-closed.

### `interval.py`

Contains the trusted closed interval type plus point/integer helpers.  Every
accepted endpoint is now canonicalized to a Python `float`, closing the Step-2
endpoint-type defect in which arbitrary-precision Python integers could survive
validation without becoming binary64 values.

### `binary64.py`

Contains the auditable outward-rounded scalar backend for `+`, `-`, `*`, and
`/`.  A `RigorousBinary64Backend` audits the floating-point environment on
construction.  Proof-critical finite hardware overflow is rejected.  Ordinary
finite results are expanded outward with adjacent binary64 values.

### `reductions.py`

Contains proof-critical reductions with both a scalar reference path and
proof-preserving vectorized paths for sufficiently long reductions:

- point dot products;
- point×interval and interval×interval dot products;
- point and interval matrix-vector/matrix-matrix products;
- nonnegative dot/matrix upper bounds;
- rigorous infinity-row-sum norm upper bounds;
- small matrix helpers used by linear-solve certificates.

The fast matrix kernels batch many independent pairwise reduction trees; they do
not trust a BLAS reduction and do not widen only once after a floating result.

## Independent Step-2 oracle

Step-1 migration tests compared this code with the historical Version-0.11
implementation.  Step 2 adds an independent exact oracle under
`tests/support/exact_binary64.py`.  Each finite binary64 operand is converted to
the exact rational represented by its bits, and exact `Fraction` arithmetic is
used to verify inclusion.

The independent tests are:

- `tests/test_scalar_binary64_exact.py`;
- `tests/test_reductions_exact.py`.

Standalone stress tools are:

- `tools/audit_scalar_binary64.py`;
- `tools/audit_reductions.py`.

See `STEP2_TRUSTED_KERNEL_AUDIT.md` for the audit methodology, results, and
scope conditions.

## Trust scope

A successful environment audit is required at rigorous-backend construction.
The current machine-rigor contract additionally assumes external code does not
change the process floating-point environment during the subsequent proof
execution.  AutoIA itself does not intentionally alter that environment.

The reference backend is not a formal proof of CPython/NumPy/CPU correctness.
It is an explicitly conditioned, independently tested trusted computing base.
Cross-platform validation should remain part of release engineering.

## Performance principle

The scalar-loop backend remains the slow reference implementation and independent
comparison target. The current fast backend uses vectorized elementwise binary64
arithmetic plus explicit outward `nextafter` at every proof-relevant product and
pairwise addition. A future BLAS-based backend would require a separately proved
error model; no BLAS result is presently trusted directly.

## Step-3 elementary scalar kernels

`0.14.0.dev1` adds `trusted/elementary.py`.  It implements rigorous real
`sqrt`, `exp`, `log`, `log10`, `sin`, and `cos` interval kernels from the already-audited
scalar arithmetic plus analytic series bounds.  Platform transcendental values are
never accepted merely because they were returned by `math` or NumPy.  For example,
`sqrt` proposals are certified by trusted squaring; `log` uses a rigorously bounded
atanh series; and trigonometric argument reduction uses a rigorous Machin-formula
pi enclosure with `[-1,1]` as a fail-safe coarse range when reduction is too wide.

`tools/audit_elementary.py` provides an independent high-precision stress audit
using optional `mpmath`; the installed proof code has no `mpmath` dependency.
See `ELEMENTARY_FUNCTIONS.md` for the full algorithms and polynomial lifting.


## Step 9 vectorized trusted reductions

`0.20.0.dev1` adds a second rigorous reduction implementation for long stored-point
reductions. Products are widened componentwise with vectorized binary64
`nextafter`, then summed in an outward-rounded pairwise interval tree. This is a
proof computation, not a BLAS error estimate. The historical scalar reduction
remains available with `reduction_method="reference"`; the default `"auto"` mode
uses it for short reductions and selects the vectorized tree at 16 or more terms.
The runtime environment audit now checks NumPy float64 elementwise add, multiply,
and vectorized `nextafter` behavior used by the fast path.


## Step 9b interval and batched reductions

`0.20.0.dev2` extends the vectorized proof path to point×interval and
interval×interval reductions. For a point `a` and interval `[l,u]`, the product
endpoints are chosen by the sign of `a`, multiplied in vectorized binary64, and
widened outward. For two intervals, all four endpoint products are widened and
the minimum lower / maximum upper endpoints are retained. Long sums are then
combined by the same outward pairwise tree. Matrix kernels batch many independent
dot trees along a final reduction axis, with bounded temporary work arrays.
Extended/nonfinite cases retain the mature scalar fallback semantics.


## Step 10b release hardening

`0.22.0.dev2` strengthens the fail-closed runtime environment audit with explicit
scalar-division and subnormal tie-to-even probes. The deterministic release audit
adds critical binary64 lattice and pairwise-tree boundary cases checked against
exact rational arithmetic.

The trusted square-root point kernel also handles the upper finite exponent
boundary by rigorous even-power-of-two rescaling. If `x > 0` and an even integer
`e` is chosen, the exact identity `sqrt(x) = 2^(e/2) sqrt(x/2^e)` lets the proof
move the squaring comparison to a moderate scale. All scaling operations remain
trusted interval operations.

Cross-platform validation is driven by `tools/run_v1_hardening.py` and the
Version-1 proof-hardening CI workflow. A workflow definition is not itself
platform evidence; each platform report must be collected and reviewed before
Version 1.0.


## Batched endpoint-array primitives (`0.23.0.dev4`)

The trusted layer now includes elementwise lower/upper endpoint-array arithmetic,
certified batched point/interval exponential and square root, and a fixed
point-valued lower-triangular solve against arbitrary leading families of
interval right-hand sides.  These use the same binary64 environment contract and
outward `nextafter` policy as the scalar oracle.  Scalar implementations remain
available for independent regression.  Exact integer proof constants may be
cached on one already-audited backend for the lifetime of that execution; the
standalone `integer_interval(n)` API retains its historical environment audit.
