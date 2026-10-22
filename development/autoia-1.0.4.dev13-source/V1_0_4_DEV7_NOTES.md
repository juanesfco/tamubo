# AutoIA 1.0.4.dev7 notes

This checkpoint hardens the `abs` / `np.abs` / `np.absolute` semantics added in
`1.0.4.dev6`.

## Sign-definite absolute value

For a validated value `x`, if an independently rigorous natural interval proves
`x >= 0`, AutoIA now passes that physical-domain sign certificate into the
structured representation and returns the original dependency for `abs(x)`.
If it proves `x <= 0`, AutoIA preserves the dependency through `-x`.  Only a
range that genuinely crosses zero uses the nonsmooth interval image
`[0, max(|lower|, |upper|)]`.

This matters when a polynomial/generator structural enclosure is slightly wider
than the independently propagated natural interval.  The structural widening
can no longer cause `abs` to lose a sign that has already been rigorously proved.

A direct regression now checks that `np.abs(x)` and Python `abs(x)` on
`x in [0.2, 0.6]` return a lower endpoint near `0.2`, rather than spuriously
widening to zero.  Negative sign-definite and zero-crossing cases are also
covered.

Example `37_composed_numpy_elementary_stress.py` has also been added as a
composed-function sharpness diagnostic for the elementary functions introduced
in dev6.

## Validation

The complete test suite was run in three chunks because one monolithic pytest
process exceeds the execution harness wall-clock limit:

- 403 passed
- 3 optional live GP tests skipped
- 0 failures
- 2,451 pytest subtests passed
