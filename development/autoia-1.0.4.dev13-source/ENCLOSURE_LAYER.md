# AutoIA enclosure layer (`0.12.0.dev5`)

This checkpoint extracts the scalar enclosure representations from the private
Version-0.11 reference core into `autoia/enclosures/`.

The move has two goals:

1. preserve Version-0.11 numerical behavior while reducing the monolith; and
2. make `ValidatedValue` dispatch through a representation contract rather than
   through hard-coded polynomial-versus-interval `isinstance` tests.

## Mathematical invariant

For a scalar exact program quantity `z(q)` and an enclosure representation `E`,
AutoIA requires

\[
 z(q) \in \Gamma_E(q)
 \qquad \text{for every admissible } q.
\]

Here `Gamma_E(q)` denotes the set represented by `E` at the original centered
uncertain-input coordinate `q`.

The universal invariant is therefore representation independent.  Polynomial,
interval, piecewise, and future enclosure families are implementations of this
contract rather than definitions of AutoIA itself.

## Extracted modules

```text
autoia/enclosures/
    __init__.py
    runtime.py
    polynomial.py
    generator.py
    interval.py
    piecewise.py
    value.py
```

### `polynomial.py`

Contains the mature polynomial-plus-remainder representation

\[
 z(q) \in P(q) + R.
\]

The historical internal name `ParameterizedEnclosure` is retained as the class
name during migration, and

```python
PolynomialEnclosure is ParameterizedEnclosure
```

is intentionally true.  This preserves the existing semantic-solve and
regression behavior while the public terminology remains `PolynomialEnclosure`.

The class depends downward on:

* `autoia.trusted` for rigorous interval arithmetic; and
* `autoia.polynomial` for polynomial coefficient arithmetic and rigorous ranges.

It does not import the active reference core, API, results, or controller
modules.

### `interval.py`

Contains the dependency-blind representation

\[
 z \in X.
\]

It uses the same domain/backend metadata as the polynomial representation so
that conversions and mixed operations are explicit.

### `piecewise.py`

Contains a finite union of already-certified branch enclosures.  It is currently
range-capable but is deliberately not enabled for ordinary arithmetic inside a
`ValidatedValue` because branch-domain/path alignment has not yet been given
formal execution semantics.

### `value.py`

Contains the representation-independent `ValidatedValue` wrapper.

The previous implementation explicitly checked

```python
isinstance(rep, PolynomialEnclosure)
isinstance(rep, IntervalEnclosure)
```

for ordinary arithmetic dispatch.  The extracted implementation instead uses a
small structural representation contract.

## Flow-representation software contract

A scalar representation participating in ordinary engineering-model arithmetic
provides:

```text
representation_name
promotion_priority
supports_value_arithmetic
h
m
order
backend
range_method
copy()
range()
constant_like(value)
negated()
binary_same(other, op)
convert_from(other)
```

`ValidatedValue` only relies on that interface for generic arithmetic.

### Same-family arithmetic

If two operands use the same representation family, `ValidatedValue` calls

```python
rep.binary_same(other_rep, op)
```

where `op` is currently one of `"add"`, `"mul"`, or `"div"`.

### Mixed-family arithmetic

For different representation families, the implementation compares

```text
promotion_priority
```

and chooses the higher-priority representation as the target.  That target
performs a sound conversion of both operands through `convert_from` before the
same-family operation is applied.

The priority is a strategy choice, not part of the soundness proof.  Soundness
requires every conversion

\[
 E_A \longrightarrow E_B
\]

to satisfy

\[
 \Gamma_{E_A}(q) \subseteq \Gamma_{E_B}(q)
 \qquad \text{for every admissible } q.
\]

The current priorities are:

```text
generator-polynomial : 150
polynomial           : 100
interval             :   0
```

so mixed interval/polynomial arithmetic reproduces Version-0.11 behavior by
converting the interval value into a constant polynomial plus rigorous remainder.

## Why the contract is real rather than nominal

`tests/test_enclosure_regression.py` defines a third `ToyRepresentation` that is
neither polynomial nor interval.  It implements the structural contract and can
be wrapped by `ValidatedValue`, can participate in same-family arithmetic, and
can win a mixed-operation promotion against `IntervalEnclosure` without any
change to `ValidatedValue`.

This test is not a proposed production representation.  It is an architecture
test demonstrating that arithmetic dispatch no longer depends on the two
currently built-in concrete classes.


## Physical-domain proof metadata (`0.22.0.dev1`)

Let `P` be the original physical input box, `p_c` the stored binary64 center, and

\[
Q_{\rm phys}=\{p-p_c:p\in P\}
\]

the actual centered physical set.  The usual symmetric polynomial proof box

\[
Q=\prod_i[-h_i,h_i]
\]

is constructed so that `Q_phys` is contained in `Q`; it can be slightly larger.
The semantic `ValidatedValue` invariant is required on `Q_phys`.  Polynomial
range kernels may continue proving the stronger statement on all of `Q`.

`ValidatedValue` can additionally carry a `natural_interval` that rigorously
encloses the same program quantity on `Q_phys`.  This shadow is intentionally in
the wrapper rather than a concrete representation: it is independent proof
metadata, not a polynomial coordinate and not a shared-error generator.  Losing
this optional shadow through an unsupported natural operation is permitted; it
may never weaken the primary representation's enclosure.

Domain-sensitive operations can intersect this physical-box shadow with
independent structural ranges to prove admissibility.  Such an intersection is
not clipping because every intersected interval is already a valid enclosure on
`Q_phys`.

## Range-policy runtime context

Polynomial reciprocal certification needs access to the active proof budget and
to proof-policy diagnostics.  The small dynamic stack supporting that interaction
was moved into `autoia/enclosures/runtime.py` rather than leaving a dependency
from the representation layer back to the reference/controller monolith.

This module does **not** contain controller decision algorithms.  It provides
only:

* sound interval intersection;
* a safe point representative inside an interval; and
* a dynamic policy/event context supplied by the higher execution layer.

## Regression standard

For this migration, direct representation operations are compared against the
untouched Version-0.11 oracle.

The tests require bit-for-bit equality of binary64 endpoints/coefficients for:

* polynomial parameter construction;
* polynomial generic arithmetic;
* reciprocal and division;
* interval enclosures;
* `ValidatedValue` polynomial arithmetic;
* `ValidatedValue` interval arithmetic;
* mixed interval/polynomial promotion; and
* piecewise hulls.

End-to-end spring/coupled regressions continue to run as before.

## Dependency boundary

The intended dependency direction is now

```text
API / semantic primitives / controller
             |
      autoia.enclosures
             |
      autoia.polynomial
             |
        autoia.trusted
```

The exact graph is not strictly linear: the interval and piecewise enclosure
modules depend directly on `autoia.trusted`, while the polynomial enclosure also
depends on `autoia.polynomial`.  The important rule is that the enclosure layer
must not import upward from the active reference core, API, results, or public
controller-policy layer.

Boundary tests enforce this rule.

## What did not change

This checkpoint does not intentionally change:

* outward rounding;
* polynomial coefficient arithmetic;
* range algorithms;
* semantic linear-solve mathematics;
* reciprocal mathematics;
* subdivision mathematics; or
* the current representation promotion policy.

During the dev5 migration, the historical compatibility core dropped from 2581
to 1924 lines because the representation implementation moved out of it.  In
dev9 that compatibility module is no longer part of the production package.


## Step-7 shared-generator representation

`generator.py` implements the additional scalar family

\[
z(q) \in P(q) + \sum_j g_j\epsilon_j + R,
\qquad \epsilon_j\in[-1,1].
\]

Stable generator ids preserve error provenance through fan-out.  Stacking the
scalar coefficients across outputs produces the vector matrix `G` in
`P(q)+G epsilon+R`.  The representation participates in the same structural
`ValidatedValue` contract and has a higher promotion priority than the legacy
polynomial family.  Sound conversion back to the polynomial family is obtained
by replacing the generator term by its symmetric interval range.  See
`SHARED_GENERATOR_REPRESENTATION.md` for the arithmetic contract and current
solve limitation.
