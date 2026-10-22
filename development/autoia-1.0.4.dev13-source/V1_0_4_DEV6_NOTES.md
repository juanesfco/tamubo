# AutoIA 1.0.4.dev6 notes

This checkpoint adds transparent support for six common NumPy elementary functions:

- `np.tanh`
- `np.sinh`
- `np.arctan` (and `np.atan`)
- `np.abs` / `np.absolute`
- `np.log1p`
- `np.cosh`

Matching `math.sinh`, `math.cosh`, `math.tanh`, `math.atan`, and `math.log1p`
forms are also dispatched locally. NumPy/math modules themselves are never mutated.

Proof semantics:

- `sinh` and `cosh` are enclosed from the trusted exponential kernel;
- `tanh` uses a stable rigorous formula based on `exp(-|x|)`;
- `arctan` uses rigorous range transformations and an explicitly bounded series;
- `log1p` is reduced to the trusted logarithm with the real-domain obligation `x > -1`;
- `abs` uses its exact piecewise interval image and preserves structured dependency when sign-definite.

For Version 1, broad-box polynomial `tanh` and `arctan` calls use the rigorous
monotone interval image rather than a potentially much looser Taylor remainder.

Validation added in `tests/test_added_elementary_functions.py` includes dense
end-to-end containment, interval and polynomial representations, aliases and
object arrays, fail-closed `log1p` domain behavior, `abs` sign/crossing behavior,
and independent 160-digit `mpmath` audits of the trusted point and interval kernels.

Full regression, split only to avoid the harness wall-clock timeout:

- 403 passed
- 3 optional live GP tests skipped
- 0 failures
- 2,451 pytest subtests passed
