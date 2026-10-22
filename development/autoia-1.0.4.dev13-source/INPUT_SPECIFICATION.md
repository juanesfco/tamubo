# AutoIA V1 physical-input specification semantics

**Applies to:** AutoIA 1.0.1 and later V1 patch releases  
**Scope:** how a user declares the physical input interval before AutoIA begins its validated computation.

## 1. Why this boundary exists

AutoIA's trusted arithmetic rigorously encloses proof-critical numerical operations after an input box has been supplied. There is a separate question, however: what real interval did the user intend to supply?

A decimal such as `0.1` is not exactly representable in IEEE-754 binary64. Python converts the literal to a nearby binary64 number before an ordinary function call receives it. Therefore these two statements are mathematically different:

```python
Interval(0.1, 0.3)
Interval.enclosing("0.1", "0.3")
```

AutoIA 1.0.1 makes that distinction explicit without changing the behavior of existing V1 code.

## 2. Direct `Interval`: exact stored-binary64 semantics

```python
I = Interval(0.1, 0.3)
```

means that the two stored binary64 endpoint values are interpreted as exact real numbers. This is the original AutoIA 1.0.0 contract and remains the default.

For example, the stored binary64 value produced by Python for `0.1` is slightly larger than the exact rational number `1/10`, while the stored binary64 value produced for `0.3` is slightly smaller than `3/10`. AutoIA does not guess that the source spelling was intended as an exact decimal.

Use direct `Interval` construction when the stored numerical values themselves define the model or input set.

## 3. `Interval.enclosing`: exact decimal/rational physical endpoints

Use

```python
I = Interval.enclosing("0.1", "0.3")
```

when the intended physical interval is the exact real interval

\[
[1/10,3/10].
\]

Accepted exact endpoint specifications are:

- decimal strings, including scientific notation;
- `decimal.Decimal`;
- `fractions.Fraction`;
- integers, including NumPy integer scalars.

For example:

```python
from decimal import Decimal
from fractions import Fraction
from autoia import Interval

A = Interval.enclosing("0.1", "0.3")
B = Interval.enclosing(Decimal("0.1"), Decimal("0.3"))
C = Interval.enclosing(Fraction(1, 10), Fraction(3, 10))
```

All three describe the same exact requested real interval.

Internally AutoIA chooses finite binary64 proof endpoints `L` and `U` such that

\[
L \le 1/10 \le 3/10 \le U.
\]

The conversion uses the ordinary float conversion only as a proposal. Exact rational comparison verifies containment, and adjacent binary64 values are obtained with the audited `nextafter` machinery. If the finite enclosure cannot be verified, AutoIA fails closed.

Floating-point arguments are intentionally rejected by `Interval.enclosing`:

```python
Interval.enclosing(0.1, 0.3)  # InvalidInputError
```

because the exact decimal/rational intent has already been lost by that point. Use strings, `Decimal`, or `Fraction` instead.

## 4. `Interval.outward`: one-ulp padding of stored floats

```python
I = Interval.outward(0.1, 0.3)
```

moves the supplied lower binary64 endpoint one adjacent value toward negative infinity and the upper endpoint one adjacent value toward positive infinity.

This is useful when the user deliberately wants a conservative one-ulp pad around already stored floats. It does **not** prove that a value computed outside AutoIA differs from some unknown exact quantity by at most one ulp. Therefore it is a convenience constructor, not a provenance theorem.

If one-ulp padding would require an infinite endpoint, AutoIA V1 fails closed because V1 input intervals require finite endpoints.

## 5. Point inputs

A binary64 point is still written

```python
Interval(3.0, 3.0)
```

when the stored value is the intended exact point.

A nonrepresentable exact point is written, for example,

```python
Interval.enclosing("0.1", "0.1")
```

and generally becomes a two-binary64-value proof interval bracketing the exact real point. This slight widening is necessary because the exact point itself has no binary64 representation.

## 6. Constants inside engineering code

Ordinary constants in ordinary engineering code retain the V1 semantics:

```python
def model(p):
    return 0.37*p[0] + 12.5*p[1] - 4.8
```

AutoIA interprets those stored binary64 constants as exact real constants equal to their stored values. Users do **not** need to rewrite every floating literal in an existing engineering program.

Three cases should be distinguished:

1. **Stored numerical coefficient.** Leave it as ordinary code. AutoIA certifies the computation conditional on that stored binary64 value.
2. **Physically uncertain coefficient.** Make it an uncertain AutoIA input and supply its physical interval explicitly.
3. **Genuinely exact mathematical quantity.** Prefer an exact formulation using exactly representable integers/rationals where practical or a supported semantic primitive. AutoIA does not reinterpret arbitrary Python floating literals as exact decimals.

For example, `x/10` uses the exact integer constant 10 under the supported real arithmetic semantics. This is different from silently reinterpreting the stored literal `0.1` as the mathematical rational `1/10`.

## 7. Result metadata

Public one-box and prepared-session results now expose:

```python
result.input_semantics
result.input_specification
result.physical_input_box
```

For an exact-decimal input, a record includes the requested exact endpoint specifications, canonical rational forms, the stored binary64 proof endpoints, and a `containment_verified` flag.

This metadata is descriptive proof provenance. The actual numerical proof continues to use the stored binary64 proof interval.

## 8. Recommended rule

Use the following default decision:

```text
Do the stored binary64 endpoints themselves define my input set?
    yes -> Interval(lo, hi)

Do I mean exact decimal/rational endpoints that may not be binary64 representable?
    yes -> Interval.enclosing("...", "...") or Decimal/Fraction

Do I merely want one adjacent binary64 value of conservative padding?
    yes -> Interval.outward(lo, hi)
```

Do not use `Interval.outward` as a substitute for knowing the provenance of an externally computed quantity.
