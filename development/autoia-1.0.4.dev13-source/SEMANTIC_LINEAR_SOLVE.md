# AutoIA semantic linear-solve layer — current contract

The production implementation is `autoia/semantic/linear_solve.py`.

The polynomial semantic primitive certifies

\[
A(q)x(q)=b(q)
\]

when the retained matrix and right-hand side are general point-coefficient
polynomials in the original centered uncertain inputs, with rigorous additive
interval remainders:

\[
A(q)\in\sum_\beta A_\beta q^\beta+\mathcal R_A,
\qquad
b(q)\in\sum_\beta b_\beta q^\beta+\mathcal R_b.
\]

The proof remains based on the residual of the actually stored predictor and the
fixed-point identity

\[
e=Cr+(I-CA)e.
\]

All proposal predictors and point preconditioners may be ordinary floating-point
calculations.  Inclusion is established only by trusted outward-rounded bounds

\[
|Cr|\le c,
\qquad
|I-CA|\le B,
\qquad
\|B\|_\infty<1.
\]

The Step-5 general path adds:

- arbitrary retained polynomial matrix/right-hand-side coefficients;
- a general coefficient recurrence for polynomial solution proposals;
- direct signed polynomial ranging of the preconditioned defect;
- a rigorous preconditioner portfolio using both the constant coefficient matrix
  and a rigorous full-family interval-hull midpoint;
- a predictor portfolio whose selection prioritizes small proven unstructured
  correction remainders so downstream dependence is preserved;
- standard NumPy multiple-right-hand-side solves `A X = B`.

The established affine retained-input path remains unchanged and continues to
match the historical oracle bit-for-bit.  The dependency-blind interval semantic
solve also remains available.

For the full notation, derivation, portfolio logic, limitations, and validation
cases, see `STEP5_GENERAL_LINEAR_SOLVE.md`.


## Step 7b shared-generator solve (`0.18.0.dev2`)

When the active representation is `generator`, the solver now certifies the
joint family

\[
A(q,\epsilon)\in\widehat A(q)+\sum_s A_s\epsilon_s+\mathcal R_A,
\qquad
b(q,\epsilon)\in\widehat b(q)+\sum_s b_s\epsilon_s+\mathcal R_b.
\]

Its predictor has the form

\[
X(q,\epsilon)=P(q)+\sum_s u_s\epsilon_s,
\]

with the original incoming generator IDs.  Proposal sensitivities satisfy a
center/reference equation of the form

\[
A_{\rm ref}u_s=b_s-A_sx_0,
\]

but are not trusted.  The actual stored predictor is certified through

\[
r=b-AX,\qquad e=Cr+(I-CA)e.
\]

Both `Cr` and `I-CA` are ranged over the complete physical-input and auxiliary
generator domain.  If the rigorous contraction bound is below one, the returned
solution keeps the input generators explicitly and adds only the remaining
componentwise correction as fresh independent generators.  See
`STEP7B_GENERATOR_AWARE_SOLVE.md`.
