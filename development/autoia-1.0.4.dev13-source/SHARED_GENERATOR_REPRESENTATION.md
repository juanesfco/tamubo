# AutoIA shared-generator representation

## Purpose

The established polynomial enclosure stores one scalar quantity as

\[
z(q) \in P(q) + R,
\]

where `q = (q_1,...,q_m)` are the original centered physical uncertain inputs,
`P(q)` is a point-coefficient polynomial in those inputs, and `R` is one
unstructured scalar interval remainder.

That representation is sound, but an interval remainder has no provenance. If
one validated scalar is reused twice, the two downstream appearances of the
same `R` are treated as independent.  This can destroy exact cancellation.

Step 7 adds the representation

\[
\boxed{z(q) \in P(q) + \sum_{j=1}^{s} g_j\,\epsilon_j + R,}
\qquad \epsilon_j\in[-1,1].
\]

For a vector output `z in R^r`, stacking the scalar generator coefficients gives

\[
\boxed{z(q) \in P(q) + G\epsilon + R,}
\]

where:

- `r` is the number of vector components;
- `s` is the number of active auxiliary generators;
- `G in R^{r x s}` is the generator coefficient matrix;
- `epsilon = (epsilon_1,...,epsilon_s)^T` is the vector of auxiliary generator
  variables, each constrained to `[-1,1]`;
- `R` is any remaining unstructured interval fallback.

The auxiliary variables are **not physical uncertain inputs**.  They encode
provenance of approximation, truncation, coefficient, or certification error.

## Generator identity and soundness

A generator id is a provenance label.  Equal ids mean the same auxiliary
variable; distinct ids are independent for enclosure purposes.  AutoIA only
reuses an id when the error actually descends from the same validated scalar.
It never merges unrelated errors merely because their interval bounds happen to
be equal.

If

\[
x=P_x+\sum_j g_j\epsilon_j+R_x,
\quad
y=P_y+\sum_j h_j\epsilon_j+R_y,
\]

then addition retains shared ids:

\[
x+y=(P_x+P_y)+\sum_j(g_j+h_j)\epsilon_j+(R_x+R_y).
\]

Thus if `y` is literally the same validated value as `x`, the shared generator
terms cancel in `x-y`.

## Introducing a new generator from an interval error

Suppose a rigorous operation produces one finite interval error `E=[E_lo,E_hi]`.
AutoIA chooses a binary64 point `c in E` and rigorously bounds

\[
E-c \subseteq [-r,r].
\]

The center `c` is shifted into the polynomial constant coefficient and a fresh
generator `r epsilon_new` is introduced.  The coefficient-shift rounding error
is included in the same rigorous radius.

This operation is inclusion-preserving.  It does not claim the error is random
or probabilistic.

## Multiplication

The generator family is affine in `epsilon`, so it is not algebraically closed
under multiplication.  Write

\[
P_x=p_0+\Delta P_x,
\quad P_y=q_0+\Delta P_y.
\]

AutoIA retains the representable first-order generator term

\[
\sum_j(q_0 g_j+p_0 h_j)\epsilon_j.
\]

Terms such as

\[
\Delta P_x\,h_j\epsilon_j,
\quad
\Delta P_y\,g_j\epsilon_j,
\quad
(g_i\epsilon_i)(h_j\epsilon_j)
\]

are rigorously ranged with the trusted kernel and moved into a new certified
unstructured contribution.  That contribution can then be represented by a
fresh generator.  This preserves soundness while retaining useful first-order
provenance.

## Nonlinear elementary functions

The existing validated Taylor-composition machinery is reused.  Taylor powers
are formed with the generator-aware multiplication rule, so shared generator
information can survive nonlinear compositions when representable.  Terms
outside the affine-generator family are rigorously absorbed into new generator
or residual bounds.

## Reciprocal and division

The current Step-7 reciprocal fallback is deliberately conservative.  The
shared-generator value is soundly collapsed to the mature polynomial-plus-
interval-remainder representation, the audited reciprocal certificate is
applied there, and the resulting certified remainder is re-lifted to a fresh
generator.  This loses generator correlation across that reciprocal operation,
but it does not lose rigor.

A future sharper reciprocal may certify a generator-linearized reciprocal
directly.

## Semantic linear solve

As of `0.18.0.dev2`, the semantic linear solver is generator-aware.  If

\[
A(q,\epsilon)\in \widehat A(q)+\sum_s A_s\epsilon_s+\mathcal R_A,
\qquad
b(q,\epsilon)\in \widehat b(q)+\sum_s b_s\epsilon_s+\mathcal R_b,
\]

AutoIA proposes

\[
X(q,\epsilon)=P(q)+\sum_s u_s\epsilon_s
\]

using the same incoming generator IDs.  It then rigorously certifies the
residual of that actual stored predictor and the defect `I-C A` over the full
joint physical/generator domain.  Thus shared input-generator provenance can
survive through `np.linalg.solve`.

The remaining fixed-point correction is still proved only as a componentwise
box.  AutoIA therefore assigns fresh independent correction generators to the
solution components rather than inventing unsupported coupling.  See
`STEP7B_GENERATOR_AWARE_SOLVE.md` for the derivation and validation.

## Public use

Select the representation with

```python
result = autoia(
    model,
    parameter_intervals,
    representation="generator",
)
```

The scalar representation class is

```python
SharedGeneratorPolynomialEnclosure
```

and ordinary result objects expose

```python
result.generator_ids
result.generator_matrix
```

where `result.generator_matrix` is the matrix `G` obtained by stacking the
scalar generator coefficients of the returned outputs.

## Current limitation: generator growth

Repeated nonlinear operations can create many small generator ids.  This is a
sharpness/performance issue, not a soundness issue.  The Step-7 sampled audit
observed up to 24 active ids in the tested nonlinear families.  Generator
compression/merging must preserve provenance and therefore belongs in later
controller/performance work rather than being introduced ad hoc here.
