# Certified `scipy.optimize.linprog` optimal-value semantics

AutoIA 1.0.4.dev13 includes a Version-1 semantic boundary for ordinary
`scipy.optimize.linprog` calls executed inside an AutoIA engineering model.

## Supported problem

For uncertain physical inputs `q` ranging over the current AutoIA box, AutoIA
supports the LP

\[
V(q)=\min_z c(q)^T z
\]

subject to

\[
A_{ub}(q)z\le b_{ub}(q),\qquad A_{eq}(q)z=b_{eq}(q),\qquad l\le z\le u.
\]

Here:

- `z` is the LP decision vector;
- `c(q)` is the objective-coefficient vector;
- `A_ub(q)` and `b_ub(q)` are inequality-constraint data;
- `A_eq(q)` and `b_eq(q)` are optional equality-constraint data;
- `l` and `u` are constant variable bounds;
- `V(q)` is the optimal LP value.

The Version-1 semantic certifies `res.fun`. It does **not** certify `res.x`.
Attempting to use `res.x` during validated execution fails closed.

Equality constraints (`A_eq`, `b_eq`) are supported by the exact semantic
reduction

\[
A_{eq}z=b_{eq}\quad\Longleftrightarrow\quad
A_{eq}z\le b_{eq},\; -A_{eq}z\le -b_{eq}.
\]

The two copies retain the same AutoIA dependency-bearing coefficients. A fixed
primal witness must therefore prove both inequalities, which is exactly an equality
proof. If a parametric equality has no fixed feasible witness across the box, the
existing lower-only `[L,+inf]` behavior applies. Mixed-integer `integrality` remains
deferred.

## Ordinary user code

```python
from scipy.optimize import linprog
from autoia import autoia, Interval


def engineering_model(p):
    x = p[0]
    result = linprog(
        [-1.0, -x],
        A_ub=[[1.0, 1.0]],
        b_ub=[1.0],
        bounds=[(0.0, 1.0), (0.0, 1.0)],
    )
    return result.fun


r = autoia(engineering_model, [Interval.enclosing("0.5", "2.0")])
```

The user does not write an interval LP and does not replace SciPy with an
AutoIA-specific optimizer API.

## Proof architecture

SciPy/HiGHS is proposal machinery only. A successful floating-point solver
status is never trusted as the certificate.

### Rigorous lower bound

For any fixed vector `y <= 0`, define

\[
d(q)=c(q)-A(q)^T y.
\]

Then weak duality gives

\[
V(q)\ge
b(q)^T y+
\sum_j \inf_{l_j\le z_j\le u_j} d_j(q)z_j.
\]

AutoIA evaluates this expression using the same dependency-carrying
`ValidatedValue` objects that formed `c`, `A`, and `b`. Repeated occurrences of
one uncertain physical input therefore remain coupled.

HiGHS dual marginals from several numerical proposal LPs generate candidate
`y` vectors. Every candidate is clipped/projected onto the mathematically safe
cone `y <= 0` before proof evaluation. `y=0` is also included as a fallback.
The candidate whose *globally certified* lower endpoint is largest is selected.

For finite variable bounds, a reduced-cost sign that is rigorously fixed keeps
the dependency-bearing expression exactly. If its sign changes across the box,
Version 1 collapses only that support term to a rigorous scalar lower bound
rather than triggering hidden physical subdivision merely for tightness.

For a one-sided infinite variable bound, a finite lower certificate requires
the corresponding reduced-cost sign to be proved uniformly.

### Lifted McCormick/RLT lower bound for polynomial-parametric coefficients

Version `1.0.4.dev13` adds a second, independent lower-bound theorem for
parametric LPs.  It is a portfolio member: the original fixed-dual weak-duality
bound remains available and AutoIA keeps the larger certified lower endpoint.

Let `q` denote the centered physical input coordinates on the current AutoIA
box and let `z` denote the LP decision variables.  The lower endpoint of the
parametric value function can be written as the joint problem

\[
\inf_{q,z} c(q)^Tz
\]

subject to the physical box for `q`, the LP constraints, and the constant bounds
on `z`.  Polynomial dependence in `c(q)` or `A(q)` then produces products such
as

\[
m_\alpha z_j,
\qquad
m_\alpha=q_1^{\alpha_1}\cdots q_d^{\alpha_d},
\]

where `alpha` is a multi-index of nonnegative integer exponents.

AutoIA introduces shared lift variables for the physical monomials `m_alpha`
and for repeated products `m_alpha*z_j`.  If two factors `x` and `z` satisfy

\[
x\in[x_L,x_U],\qquad z\in[z_L,z_U],
\]

and `y` represents their product, AutoIA adds the four McCormick inequalities

\[
y\ge x_Lz+z_Lx-x_Lz_L,
\]
\[
y\ge x_Uz+z_Ux-x_Uz_U,
\]
\[
y\le x_Uz+z_Lx-x_Uz_L,
\]
\[
y\le x_Lz+z_Ux-x_Lz_U.
\]

These rows form an outer relaxation of the exact product graph.  Higher-order
physical monomials are built recursively as a shared lifted polynomial DAG, so
repeated appearances of the same physical monomial reuse the same lift rather
than becoming independent coefficient intervals.

The retained AutoIA polynomial has the form `P(q)+R`, where `R` is a rigorous
scalar remainder interval.  A nonzero unstructured remainder is represented by
a bounded auxiliary variable.  Products between that remainder variable and an
LP decision variable receive their own McCormick envelope.  This is deliberately
conservative but sound: it enlarges, rather than shrinks, the feasible lifted
set.

Every McCormick row is constructed with the trusted outward-rounded binary64
backend.  HiGHS solves the enlarged constant-data LP only to propose a dual
witness.  AutoIA independently recomputes a weak-duality lower certificate for
that constant LP.  Thus the lifted solver is not trusted merely because its data
are constant.

The lift requires finite bounds on factors that participate in a McCormick
product.  If a required decision-variable bound is infinite, the lifted theorem
is skipped and the fixed-dual theorem remains authoritative.

#### Certified optimization-based bound tightening

For small lifted LPs, dev13 performs up to two cost-capped OBBT rounds on
original decision variables that participate in lifted products.  To tighten a
lower bound on `z_j`, AutoIA certifies the minimum of `z_j` over the current
relaxation.  To tighten an upper bound, it certifies the minimum of `-z_j`; if
that certified lower endpoint is `L`, then

\[
z_j\le -L.
\]

Because the lifted relaxation contains the original feasible set, every such
certified bound is also valid for the original problem.  AutoIA then rebuilds
the McCormick envelopes with the tighter factor bounds.  OBBT is optional proof
work: it is triggered automatically only when the first lifted relaxation has
already improved on the cheaper fixed-dual lower bound, and it is skipped when
more than eight original decision variables require product tightening.

### Rigorous upper bound

A fixed numerical vector `z_star` gives an upper bound only after AutoIA proves

\[
A(q)z_\star-b(q)\le0
\]

for every `q` in the current box and verifies the constant variable bounds.
Then

\[
V(q)\le c(q)^Tz_\star.
\]

Candidate primal witnesses come from a portfolio:

1. ordinary numerical LP proposals at several coefficient scenarios;
2. a conservative independent-interval robustified proposal LP;
3. small inward-margin variants of that robust proposal;
4. simple bound-feasible anchors such as zero/lower/upper-bound vectors when
   applicable.

Every candidate is independently verified against the original coupled
AutoIA expressions. The robustified LP is never trusted for proof.

The lower and upper certificates are independent. If no fixed primal candidate
can be proved feasible for every `q` in the current box, AutoIA does **not**
discard the already-proved weak-duality lower bound. It returns

\[
V(q)\in[L,+\infty]
\]

with `value_upper_certified=False` and
`certificate_completeness="lower-only"`. This means AutoIA has not proved a
universal primal witness; it does **not** mean AutoIA has proved that no such
witness exists. A box that contains infeasible LP instances can therefore also
produce a lower-only extended-value enclosure.

Exact-affine AutoIA metadata is used when available so an exact residual such
as `x - x == 0` is not rejected merely because ordinary outward arithmetic
produces a subnormal-width enclosure around zero.

## Certificate fields

A successful semantic call records a certificate containing, among other
fields:

- `operation = "scipy.optimize.linprog"`;
- `semantic_mode = "certified-optimal-value"`;
- `certificate_completeness` (`"two-sided"` or `"lower-only"`);
- `value_lower_certified`;
- `value_upper_certified`;
- `optimal_value_lower`;
- `optimal_value_upper` (possibly `+inf` for a lower-only result);
- `certified_value_width`;
- `selected_dual_y`;
- `dual_candidate_count`;
- `fixed_dual_lower_bound`;
- `rlt_lower_bound`;
- `selected_lower_bound_method` (`"fixed-dual"` or `"lifted-mccormick-rlt"`);
- `rlt_initial_lower`;
- `rlt_final_lower`;
- `rlt_obbt_rounds`;
- `rlt_lifted_variable_count`;
- `rlt_lifted_row_count`;
- `rlt_monomial_count`;
- `rlt_product_count`;
- `selected_primal_z`;
- `primal_residual_upper`;
- `primal_feasible_for_all_q` (`True` when proved, otherwise `None`);
- `universal_primal_witness_found`;
- `primal_candidate_count`;
- `verified_primal_candidate_count`;
- `machine_rigorous`.

## Active-set changes and the lifted relaxation

A single fixed dual witness can still be loose across an LP active-set
transition.  dev13 does not hide physical-domain subdivision inside `linprog`
for tightness, but the lifted RLT portfolio can substantially improve the
whole-box lower bound before subdivision.

For

\[
\min_z -z
\]

subject to

\[
xz\le1,\qquad z\le x,\qquad 0\le z\le10,
\]

with `x in [0.5,2]`, the exact value is

\[
V(x)=-\min(1/x,x)
\]

and its exact range is `[-1,-0.5]`.  The fixed-dual theorem gives a lower
endpoint near `-2`.  The first McCormick/RLT relaxation gives approximately
`-1.75`.  Two certified OBBT/rebuild rounds give approximately `-1.0625`, while
retaining the same physical box.  Further caller-controlled subdivision remains
valid and can continue tightening around the basis switch.

This is a convergence-rate improvement, not a claim that one finite
McCormick/RLT relaxation is exact for every active-set transition.

## Fail-closed cases

Version 1 fails closed when, for example:

- the nominal proposal LP is unbounded or infeasible;
- no finite weak-duality lower witness can be established;
- mixed-integer integrality is requested;
- the user accesses `res.x` instead of the certified `res.fun`.

Failure to prove a fixed universal primal witness is **not** on this list:
that case returns a rigorous lower-only interval `[L,+inf]`. Proposal success
never substitutes for proof.
