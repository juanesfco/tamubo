# Step 6: path-aware validated execution

Checkpoint: `0.17.0.dev1`

This step adds rigorous scalar Python control flow without rewriting the user's
engineering model into an interval-specific programming language.

## 1. Supported control-flow form

A scalar `ValidatedValue` may now appear in ordinary real comparisons

```python
x < y
x <= y
x > y
x >= y
x == y
x != y
```

where either side may be a validated scalar or an ordinary real scalar.  Python
truth testing

```python
if x:
```

is interpreted using ordinary real semantics, i.e. `x != 0`.

Consequently ordinary `if` / `elif` / `else`, conditional expressions, and
short-circuit `and` / `or` work when their Boolean decisions reduce to supported
scalar comparisons.  Same-module Python helpers use the same path runtime
because they are executed inside AutoIA's existing private dispatch namespace.

The current frozen contract is **scalar control flow**.  Boolean masks over
validated arrays, `np.where` with an uncertain mask, uncertain indexing, and a
general contract for uncertain loop termination are not yet frozen.

## 2. Predicate semantics

For a comparison between validated scalars `lhs` and `rhs`, AutoIA forms the
validated difference

\[
d(q)=\mathrm{lhs}(q)-\mathrm{rhs}(q),
\]

where `q` denotes the vector of original centered uncertain inputs.

For example, `lhs < rhs` is true on the complete active input box if AutoIA
proves

\[
\overline d < 0,
\]

where `[\underline d,\overline d]` is a rigorous enclosure of `d(q)` on that
box.  It is false on the complete box if AutoIA proves

\[
\underline d \ge 0.
\]

Otherwise the branch is unresolved.

The analogous exact real rules are used for `<=`, `>`, `>=`, `==`, and `!=`.
Equality is proved true only when the rigorous difference enclosure is exactly
`[0,0]`; it is proved false when zero is excluded.

Branch predicates use the strong final range controller, not an unvalidated
floating-point sign test.

## 3. Why finite box splitting alone is insufficient

A decision surface such as

\[
g(p)=0
\]

can pass through a multidimensional box.  No finite collection of ordinary
closed Cartesian boxes can, in general, partition an arbitrary curved decision
surface into boxes on which the predicate is everywhere strictly true or
strictly false.

Therefore Step 6 uses two complementary mechanisms:

1. **geometric subdivision** to resolve or reduce ambiguous regions;
2. **path enumeration** when a finite box partition cannot exactly represent the
   decision surface.

For an exact affine predicate depending on one original parameter, AutoIA can do
better: it recognizes the exact comparison root and partitions directly at that
root.

## 4. Soundness of path enumeration

Let `Q` be the active input box and suppose an unresolved Boolean predicate has
two branch formulas

\[
F_T(q), \qquad F_F(q).
\]

For every `q in Q`, ordinary Python execution selects exactly one of these
formulas according to the predicate.

When AutoIA enumerates an unresolved predicate, it re-executes the model twice:
once with the Boolean outcome forced true and once forced false.  Each forced
branch formula is certified over the **whole current box** `Q`, or over a
geometrically reduced box produced earlier.  This may be a strict superset of
the inputs that can actually follow that path.

If

\[
F_T(q) \in Y_T \quad \text{for all }q\in Q
\]

and

\[
F_F(q) \in Y_F \quad \text{for all }q\in Q,
\]

then the original branch program satisfies

\[
F(q)\in Y_T\cup Y_F \quad \text{for all }q\in Q.
\]

Thus forcing both formulas over a superset is conservative but sound.  Nested
branches follow by induction: every feasible path is independently certified,
and the final result is the finite union of those path certificates.

A forced path that later contradicts a rigorously decided predicate is pruned as
infeasible.

## 5. Exact affine one-variable predicates

Suppose the branch difference is exactly

\[
d(p_i)=a+b p_i,
\]

with no retained interval remainder and only one active original parameter
`p_i`.  If `b != 0`, the decision boundary is

\[
p_i^*=-a/b.
\]

AutoIA recognizes this structure and partitions the original parameter interval
at `p_i^*` rather than repeatedly bisecting around the boundary.

For strict predicates, a closed side box may contain the equality point even when
that side is executed using the strict-side formula.  This is still sound because
the formula is certified over a superset of its actual feasible path domain.  If
neither side has the correct equality semantics, as for `==` and `!=`, AutoIA
adds a degenerate point branch at the root.

This mechanism is generic and is not specific to conditions such as `x > 0`.

## 6. Downstream propagation without premature hulling

AutoIA does not create a `PiecewiseEnclosure` in the middle of user execution and
then immediately perform arithmetic on its hull.  Instead it re-executes the
complete remainder of the engineering model separately on every retained path.

For example,

```python
if x >= 0:
    y = f(x)
else:
    y = g(x)
z = y*y + h(x)
return z
```

is certified as two separate computations

\[
z_T=f(x)^2+h(x),
\qquad
z_F=g(x)^2+h(x),
\]

on their path domains/supersets.  Only the final returned object is assembled as
a piecewise finite union.  This preserves branch-specific dependence through all
downstream supported arithmetic and semantic primitives.

## 7. Result object

A branchless model still returns the historical `AutoIAResult`.

A model that encounters unresolved control flow returns `AutoIAPathwiseResult`
(alias `PathwiseResult`).  It preserves the familiar

```python
result.lower
result.upper
result.width
result.lower_shaped
result.upper_shaped
```

interface and additionally provides

```python
result.branch_results
result.branch_boxes
result.branch_decisions
result.branch_traces
result.path_events
```

Each top-level output object is a `PiecewiseEnclosure`.  Its `.branches` retain
the independently certified scalar branch values and `.branch_metadata` retain
the associated box/path information.

## 8. Branch policy and fail-closed budgets

`BranchPolicy` controls optional path work:

```python
BranchPolicy(
    max_branches=64,
    max_path_decisions=24,
    max_split_depth_per_decision=2,
    max_domain_split_depth_per_obligation=6,
    max_total_splits=64,
    prefer_geometric_split=True,
)
```

These values affect cost and sharpness, not proof validity.  If a hard budget is
exhausted before every feasible path is certified, AutoIA raises
`ResourceLimitError` and returns no rigorous result.


## 9. Domain-predicate refinement (`0.22.0.dev1`)

A mathematical-domain obligation is different from an uncertain Boolean branch.
For example, before evaluating `sqrt(z)`, AutoIA must prove

\[
z(q)\ge 0
\]

for every physical input in the current box.  There is no second formula to
enumerate when this statement is unresolved: the model is valid on a child box
only if that child proves the domain predicate.

`DomainRefinementRequired` first triggers an on-demand replay of the **same
physical box** with trusted natural-interval tracking enabled when the current
representation can benefit from that independent proof.  This replay is not a
new path and consumes no geometric split budget.  Only if the combined proof is
still unresolved does AutoIA geometrically refine the physical input box.  Each
child re-executes the same engineering program and must independently prove the
obligation.  A derivative-based structural score is used when available to
select a split direction; otherwise the widest physical coordinate is used.  Domain refinement has its own
`max_domain_split_depth_per_obligation` budget because the appropriate depth is
not the same as the shallow heuristic used for Boolean decision surfaces.  The
shared hard `max_total_splits` and `max_branches` limits still apply.

This refinement is proof work, not path enumeration.  A wholly invalid child
causes certification failure.  If the predicate remains unresolved when the
budget is exhausted, AutoIA also fails closed.  See `STEP10A_DOMAIN_PROOFS.md`.

## 10. Interaction with existing subdivision

`autoia_subdivide(...)` composes with path-aware execution: every prescribed
subbox may itself produce a pathwise result.

The older structure-aware `autoia_adaptive_subdivide(...)` controller is not yet
combined with uncertain Python control flow because its split scoring assumes an
ordinary single-path `AutoIAResult` with polynomial range diagnostics.  Calling
that controller on a path-dependent model fails explicitly with
`UnsupportedOperationError`.  This is a controller-composition limitation, not a
soundness limitation of `autoia(...)` itself.

## 11. Current limitations

The Step-6 frozen subset does not yet claim automatic semantics for:

- uncertain NumPy Boolean masks or `np.where`;
- uncertain array indices/slices;
- arbitrary `switch`/dispatch constructs encoded through dictionaries or object
  identity;
- a general termination guarantee for uncertainty-dependent `while` loops;
- branch predicates evaluated inside unsupported compiled/external code;
- exact non-box representation of a general curved decision surface.

A branch formula may also fail certification when evaluating it over the closed
box superset required by path enumeration reaches a singular/domain boundary.
AutoIA does not discard such a feasible path.  If it cannot certify every
feasible path, it fails closed.

## 12. Validation in this checkpoint

The complete package suite contains 178 passing tests.  New tests cover exact and
strict inequalities, equality boundaries, nested `if`/`elif`, short-circuit
Boolean logic, truth testing, nonlinear multivariate predicates, branch-local
semantic linear solves, branch metadata, fixed-subdivision composition, resource
limits, output-shape consistency, and thread-local branch runtime state.

The randomized diagnostic audit in `tools/audit_path_branching.py` completed
4,000 sampled output checks across 20 randomly parameterized nested nonlinear
branch families with no sampled containment failure.  Sampling is diagnostic
only; the guarantee is the pathwise inclusion argument above combined with the
validated operations used on every retained path.


## V1 closed-endpoint comparison hardening

As of `0.24.0.dev12`, scalar predicates retain a proof-only exact affine shadow whenever the compared expression is exactly affine in the original physical inputs. This shadow uses exact rational arithmetic on the stored binary64 constants and physical box endpoints. It is consulted only for Boolean proof decisions and exact representable affine roots; it never replaces the rigorous enclosure used for numerical output bounds.

This prevents harmless outward-rounding noise from inventing a feasible strict path at a closed endpoint, e.g. `x > 1.0` on `x in [0.95, 1.0]`. Nonlinear expressions do not retain this shadow.
