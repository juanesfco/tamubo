# AutoIA typed failure semantics

**Status:** Version-1 evolving contract, package `0.24.0.dev11`.

AutoIA is fail-closed. A call either returns a result satisfying the documented
inclusion contract or raises a subclass of `AutoIAError`. The public API does
not use a returned flag such as `success=False` to represent an uncertified
result.

## Public hierarchy

```text
AutoIAError
├── InvalidInputError
├── UnsupportedOperationError
├── CertificationError
├── ResourceLimitError
├── NumericalLimitError
├── ModelExecutionError
└── InternalAutoIAError
```

### `InvalidInputError`

The AutoIA call or configuration is malformed before a proof attempt should be
made. Examples include an empty input box, an unknown representation name,
invalid range-policy settings, or invalid subdivision counts.

### `UnsupportedOperationError`

The engineering program uses an operation outside the supported language.
Examples in the current package include `numpy.tan` applied to a
`ValidatedValue`, uncertain exponents such as `x**p[1]`, and complex-valued
power branches.  Constant real powers plus real `sqrt`, `exp`, `log`, `sin`,
and `cos` are supported in the documented Python/NumPy/`math` call forms.

This means "AutoIA does not currently implement these semantics", not "the
mathematical expression is invalid". Examples also include validated
`root_scalar` calls without a bracket, non-bracketed root methods in the current
Version-1 subset, and proof-critical implicit differentiation through operations
whose derivative semantics have not yet been trusted.

### `CertificationError`

The requested mathematical computation is meaningful, but the current
representation/certifier cannot prove a condition required for a rigorous
result on the current box. Examples include:

- a reciprocal denominator whose rigorous range still contains zero after the
  configured proof escalation;
- a semantic linear solve whose rigorous contraction bound satisfies
  `rho_upper >= 1`;
- a bracketed `root_scalar` call whose endpoint sign change, bracket uniqueness,
  or local correction theorem cannot be proved on the current physical box;
- an `fsolve` call whose local nonlinear-system contraction certificate cannot
  prove a nonsingular unique solution branch around the proposal;
- `log` whose certified argument range is not strictly positive, or `sqrt`
  whose certified argument range is not nonnegative;
- a negative integer power whose certified base range contains zero;
- a non-integer real power whose certified base range is not strictly positive
  for the polynomial representation.

A `CertificationError` therefore differs fundamentally from an unsupported
operation. Subdivision or a stronger certifier may turn the same model into a
successful certificate.

### `ResourceLimitError`

Reserved for a proof attempt that is deliberately stopped by an explicit
resource limit. The current Version-0.11 controller normally returns an already
sound weaker enclosure when optional tightening exceeds budget, so this class
is not commonly raised yet.

### `NumericalLimitError`

The trusted/reference numerical layer encountered an arithmetic limit such as
proof-critical overflow or a nonfinite proposal that the current certifier
cannot safely recover from.

### `ModelExecutionError`

The user's engineering function itself raised an exception that is not an
AutoIA proof failure. AutoIA preserves the original exception as `__cause__`.
This distinction prevents a bug in user code from being mislabeled as a failed
mathematical certificate.

### `InternalAutoIAError`

An internal invariant that should follow from soundness was violated, for
example two independently sound interval enclosures becoming disjoint. This
should be treated as an AutoIA bug, not as an ordinary certification failure.

## Common fields

Every public AutoIA exception provides:

- `code`: a stable short machine-readable category;
- `message`: the human-readable explanation;
- `context`: a dictionary of structured details when available;
- `suggestions`: zero or more recovery suggestions.

The research-core exception is retained through Python exception chaining where
possible:

```python
try:
    result = autoia(model, intervals)
except CertificationError as exc:
    print(exc.code)
    print(exc.suggestions)
    print(exc.__cause__)
```

## Important semantic distinction

The hierarchy separates three questions that were previously conflated by
built-in exception classes:

1. **Can AutoIA express the operation?**  If no: `UnsupportedOperationError`.
2. **Can AutoIA prove this supported computation on this box?**  If no:
   `CertificationError`.
3. **Did the user's program itself fail while being executed?**  If yes:
   `ModelExecutionError`.

That separation is part of the candidate Version-1 public contract.


## Reciprocal proof refinement (`0.24.0.dev12`)

A denominator enclosure that contains zero is no longer automatically terminal when the exact denominator may simply be over-enclosed on a broad physical box. Reciprocal nonzero is a universal proof obligation. The path/domain controller may replay with a natural physical-box shadow and geometrically subdivide before giving up. If the allowed refinement budget is exhausted, the public result remains a fail-closed `CertificationError` whose cause is `ZeroDivisionError`, preserving the established failure taxonomy.
