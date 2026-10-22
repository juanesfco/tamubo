# AutoIA constant-real power extension (`0.14.0.dev2`)

This checkpoint extends Step 3 before the NumPy-array work of Step 4.

## Supported mathematical contract

AutoIA supports

```python
validated_base ** p
pow(validated_base, p)
np.power(validated_base, p)
math.pow(validated_base, p)
```

when `p` is one finite constant Python/NumPy integer or floating scalar.
The stored binary64 value of a floating exponent is interpreted as an exact
real constant, consistent with the rest of AutoIA's source-value contract.

### Integer exponents

For an integer `n`, positive powers are evaluated by exponentiation by squaring.
Negative powers use

```text
x**(-n) = 1 / x**n
```

and therefore require the certified base range to exclude zero.
Negative bases are permitted for integer exponents.

### Non-integer real exponents

The reference implementation uses the real branch.  Therefore a genuinely
non-integer exponent requires a positive base for polynomial propagation.  The
interval representation may touch zero only for a positive exponent.
Negative bases with non-integer exponents are not silently reinterpreted as
complex values or as special rational roots.

For a positive base and non-integer binary64 exponent `p`, AutoIA chooses an
exact decomposition

```text
p = n + r,    |r| <= 1/2,
```

where `n` is an integer and `r` is an exactly represented binary64 residual
when possible.  It evaluates

```text
x**p = x**n * x**r.
```

The integer part uses validated algebra.  For the fractional part:

* `r = +1/2` uses the validated `sqrt` primitive;
* `r = -1/2` uses the reciprocal of validated `sqrt`;
* otherwise AutoIA uses the already-certified identity

```text
x**r = exp(r * log(x)).
```

Thus powers reuse the Step-3 elementary-function proof machinery instead of
introducing an unrelated Taylor implementation.

## Trusted interval kernel

The trusted layer also provides `real_power_point_interval` and
`real_power_interval`.  Integer exponents use the natural real interval power
range.  Non-integer powers are reduced to trusted `log`, interval
multiplication by the exact exponent, and trusted `exp`.

## Deliberately unsupported

The exponent itself may not currently be a `ValidatedValue`.  Expressions such
as

```python
x ** uncertain_p
2.0 ** uncertain_p
```

are outside this checkpoint.  Complex-valued branches, `Decimal` exponents,
and exact `Fraction` exponent semantics are also deferred rather than silently
converted.

## Domain-boundary proof

As of `0.22.0.dev1`, real noninteger powers use the same general physical-domain
predicate mechanism as `sqrt` and `log`.  For a real exponent `r`:

- if `r > 0` and `r` is noninteger, the base must satisfy `x >= 0` on the entire
  physical input box;
- if `r < 0` and `r` is noninteger, the base must satisfy `x > 0` everywhere.

The proof may intersect an independent natural interval over the actual physical
box with structural polynomial ranges and may geometrically refine an unresolved
physical box.  It never clips a structural interval to the mathematical domain.
Thus positive fractional powers can certify an exact zero boundary such as
`[0,4]`, while negative fractional powers still reject that same box because the
reciprocal singularity is present.  When a positive fractional-power domain
provably touches zero, AutoIA may return a rigorous interval lift rather than
forcing a polynomial Taylor representation through the boundary.
