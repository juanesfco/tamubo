"""Example 26: choose the intended semantics of physical input endpoints.

AutoIA 1.0.1 keeps the original ``Interval(float, float)`` contract and adds an
explicit constructor for exact decimal/rational physical endpoints.
"""

from fractions import Fraction

from autoia import Interval, autoia


def exact_float(x):
    return Fraction.from_float(float(x))


def model(p):
    x = p[0]
    # Ordinary constants in engineering code keep the V1 rule: 0.1 below means
    # the exact real number equal to the stored binary64 constant.
    return x*x + 0.1*x + 2.0


ordinary = Interval(0.1, 0.3)
exact_decimal = Interval.enclosing("0.1", "0.3")
padded = Interval.outward(0.1, 0.3)

print("ordinary stored-float interval:", ordinary)
print("  semantics:", ordinary.input_semantics)
print("exact-decimal enclosing interval:", exact_decimal)
print("  semantics:", exact_decimal.input_semantics)
print("one-ulp padded interval:", padded)
print("  semantics:", padded.input_semantics)

# The motivating detail: binary64 0.1 lies slightly ABOVE the exact rational
# 1/10, while binary64 0.3 lies slightly BELOW the exact rational 3/10.
assert exact_float(ordinary.lo) > Fraction(1, 10)
assert exact_float(ordinary.hi) < Fraction(3, 10)
assert exact_float(exact_decimal.lo) <= Fraction(1, 10)
assert exact_float(exact_decimal.hi) >= Fraction(3, 10)

# Fractions and Decimal objects may also be used as exact specifications.
exact_fraction = Interval.enclosing(Fraction(1, 10), Fraction(3, 10))
assert (exact_fraction.lo, exact_fraction.hi) == (
    exact_decimal.lo,
    exact_decimal.hi,
)

result = autoia(model, [exact_decimal])
print("\nAutoIA result:")
print(result)
print("input semantics:", result.input_semantics)
print("input specification:", result.input_specification[0])
print("machine rigorous:", result.machine_rigorous)

assert result.machine_rigorous
assert result.input_semantics == ("exact-real-enclosing",)
assert result.input_specification[0]["containment_verified"] is True

print("\nKey rule:")
print("  use Interval(...) for exact stored-binary64 endpoints")
print("  use Interval.enclosing(...) when exact decimal/rational endpoints matter")
print("  use Interval.outward(...) only for explicit one-ulp padding")
