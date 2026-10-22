# AutoIA Step 4 array semantics — `0.15.0.dev1`

Step 4 broadens the supported ordinary NumPy subset while keeping scalar
uncertainty semantics in `ValidatedValue`.  AutoIA does **not** introduce a new
custom ndarray class.  During traced execution it uses ordinary NumPy
`dtype=object` arrays whose uncertain entries are validated scalar objects.

## Construction and mutation

Inside AutoIA's private shadow NumPy namespace, the following constructors
default to `dtype=object` when the user did not explicitly request a dtype:

- `np.array`, `np.asarray`
- `np.zeros`, `np.ones`, `np.empty`, `np.full`
- `np.zeros_like`, `np.ones_like`, `np.empty_like`, `np.full_like`
- `np.eye`, `np.identity`

This makes common assembly code valid without AutoIA-specific syntax, e.g.

```python
A = np.zeros((n, n))
A[i, j] = uncertain_expression
```

An explicit dtype is respected.  In particular, explicitly converting a
validated quantity to a floating dtype is not part of the validated contract.

Ordinary object-array indexing, slicing, views, and item/slice assignment are
supported for the tested cases.  The real process-wide NumPy module is not
modified.

## Broadcasting and structural operations

Ordinary NumPy object-array broadcasting of supported scalar arithmetic is
supported and regression tested.  Step 4 also adds explicit scalar-to-numeric-
ndarray broadcasting in `ValidatedValue`, so a numeric array captured in a
closure or module constant can participate in expressions such as

```python
constant_vector * uncertain_scalar
```

without having been constructed through AutoIA's NumPy proxy.

Structural operations that only rearrange object references, such as indexing,
`reshape`, transpose, `stack`, and `concatenate`, use ordinary NumPy semantics.

## Reductions

The following validated reduction forms are frozen in this checkpoint:

- `np.sum`
- `np.prod`
- `np.mean`
- `np.trace`

Common `axis` and `keepdims` use is supported.  For validated reductions,
`dtype` must be omitted or be `object`; nontrivial `where=` masks and `out=` are
not yet supported.

Array methods such as `A.sum(...)`, `A.mean(...)`, and `A.prod(...)` are also
covered by tests through NumPy object-array semantics.

## Linear algebra composition

The following ordinary algebraic array operations are supported for the tested
1D/2D engineering cases:

- `np.dot`
- `np.matmul`
- the `@` operator
- `np.inner`
- `np.outer`

`np.linalg.solve` remains the separate semantic primitive developed earlier.

## Norms

`np.linalg.norm` is locally dispatched for these validated cases:

- `ord=None` for vector Euclidean norms and matrix Frobenius norms,
- vector `ord=2`, and
- matrix `ord='fro'` / `ord='f'`.

The supported norm is mathematically

\[
\|x\|_2 = \sqrt{\sum_i x_i^2}
\]

(or the analogous Frobenius sum over matrix entries).

When ordinary polynomial composition proves the square-root domain directly,
AutoIA retains the richer representation.  If a range artifact makes the
intermediate sum-of-squares enclosure extend slightly below zero, the norm
semantic itself supplies the theorem that every exact square is nonnegative.
AutoIA then builds a sound interval norm from natural interval square ranges.
This is operation-specific semantic tightening, not arbitrary clipping of a
`sqrt` input.

Matrix spectral norms (`ord=2` for a 2D matrix), 1/infinity norms, nuclear norms,
and general p-norms remain unsupported because they require additional
nonsmooth, max, or eigenvalue semantics.

## Output shape

For compatibility, `AutoIAResult.lower`, `.upper`, and `.objects` remain flat
one-dimensional collections.  Step 4 records the model output shape and adds:

- `result.output_shape`
- `result.lower_shaped`
- `result.upper_shaped`
- `result.width_shaped`
- `result.objects_shaped`

so array-valued model outputs can be reconstructed without changing existing
code that used the flat result arrays.

## Explicit current boundaries

This checkpoint does not yet freeze:

- `np.max`, `np.min`, `np.maximum`, `np.minimum`, or uncertain comparisons;
- general `abs`/sign/nonsmooth array semantics;
- arbitrary `einsum` or `tensordot` signatures;
- matrix spectral/eigenvalue norms;
- `out=` mutation for validated ufunc/reduction calls;
- nontrivial `where=` masks;
- casting validated arrays to ordinary numeric dtypes;
- boolean masks whose truth depends on uncertain values;
- arbitrary batched higher-dimensional linear algebra beyond tested object-array
  behavior.

These are intentionally fail-closed or outside the frozen supported subset.
