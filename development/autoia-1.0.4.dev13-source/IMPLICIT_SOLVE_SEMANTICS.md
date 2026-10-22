# Validated implicit nonlinear solves for Version 1

**Checkpoint:** AutoIA `0.24.0.dev11`  
**Scope:** common SciPy `root_scalar` and `fsolve` syntax as semantic primitives.

## 1. Why implicit solves are semantic primitives

Many engineering programs do not evaluate every state explicitly.  Instead they
compute an unknown state `y` from a residual equation

\[
F(y,p)=0,
\]

where `p` denotes the uncertain physical inputs.  A floating-point nonlinear
solver is an algorithm for proposing `y`; it is not itself a proof that the
exact real equation has a solution for every `p` in the AutoIA input box.

AutoIA therefore interprets supported SciPy solve calls by their mathematical
meaning.  SciPy remains an optional proposal engine and is never trusted as the
certificate.

The current Version-1 surface is:

* `scipy.optimize.root_scalar` with a finite constant bracket and a bracketed
  method (`bisect`, `brentq`, `brenth`, `ridder`, `toms748`, or `method=None`);
* `scipy.optimize.fsolve` for square real systems with a finite constant numeric
  initial guess.

Uncertain physical quantities may be supplied through SciPy `args=` or captured
by ordinary Python residual closures/lambdas.  Pure-Python callbacks handed to
the recognized SciPy primitive are locally prepared for AutoIA NumPy semantics.
SciPy is still not an AutoIA installation dependency.

## 2. Dependency-bearing implicit predictor

Let the active residual arguments be scalar quantities

\[
a(p)=(a_1(p),\ldots,a_r(p)),
\]

where `r` is the number of active argument leaves and each `a_j(p)` may itself
depend on the same original physical inputs `p`.

At a nominal parameter representative `a_0`, SciPy proposes a nominal solution
`y_0`.  AutoIA uses forward-mode differentiation only to construct a predictor.
For a scalar root, define

\[
J_y=\left.\frac{\partial F}{\partial y}\right|_{(y_0,a_0)},
\qquad
J_a=\left.\frac{\partial F}{\partial a}\right|_{(y_0,a_0)}.
\]

These nominal derivative values are proposal data; they are not trusted proof
bounds.  When `J_y` is nonzero, the implicit-function sensitivity proposal is

\[
S=-J_y^{-1}J_a.
\]

AutoIA retains the predictor

\[
\widehat y(p)=y_0+S\bigl(a(p)-a_0\bigr).
\]

The important point is that `a(p)` remains an AutoIA dependency-bearing value.
The solve does **not** immediately replace `y` by an independent interval.
Downstream engineering calculations therefore retain first-order implicit
input dependence plus a rigorously certified correction.

For a vector system with `n` unknowns,

\[
F(y,p)=0,
\qquad y\in\mathbb R^n,
\]

AutoIA forms the analogous proposal

\[
\widehat y(p)=y_0+S\bigl(a(p)-a_0\bigr),
\qquad
S=-J_y^{-1}J_a,
\]

where `J_y` is the `n x n` Jacobian with respect to the unknown state and `J_a`
is the Jacobian with respect to active residual arguments.

## 3. Scalar `root_scalar` certificate

For a supplied bracket

\[
[a,b],\qquad a<b,
\]

AutoIA first proves the bracket semantics themselves.

### 3.1 Uniform endpoint sign change

AutoIA encloses

\[
F(a,p)\in A,
\qquad
F(b,p)\in B
\]

for every physical input `p` in the current AutoIA box and requires `A` and `B`
to have strict opposite signs.

### 3.2 Uniqueness in the bracket

AutoIA differentiates the residual with respect to `y` and rigorously encloses

\[
F_y(y,p)=\frac{\partial F}{\partial y}(y,p)
\]

over the bracket.  If a single whole-bracket enclosure is too loose, the
bracket is bisected and the derivative proof is repeated piecewise.  Every
accepted piece must exclude zero and all pieces must have the same derivative
sign.

The strict endpoint sign change plus a derivative of one nonzero sign proves
that every admissible physical parameter value has exactly one root in the
supplied bracket.

### 3.3 Local dependency-preserving correction

Let

\[
e=y-\widehat y(p)
\]

be the unknown correction to the implicit predictor and let

\[
C=J_y^{-1}
\]

be the stored binary64 scalar preconditioner proposal.  AutoIA considers the
fixed-point map

\[
T(e,p)=e-CF(\widehat y(p)+e,p).
\]

It rigorously proves two quantities over

\[
e\in[-d,d].
\]

First,

\[
|C F(\widehat y(p),p)|\le c.
\]

Second,

\[
|1-CF_y(\widehat y(p)+e,p)|\le\rho.
\]

If

\[
\rho<1
\]

and

\[
\frac{c}{1-\rho}\le d,
\]

then the contraction theorem proves a unique correction `e` in the tube.
AutoIA returns

\[
\boxed{
y(p)\in\widehat y(p)+[-d,d].
\]

Because this enclosure lies inside the supplied bracket and the bracket root is
already proven unique, the returned implicit solution is the mathematical root
requested by the bracketed `root_scalar` call.

## 4. `fsolve` certificate

`fsolve` does not supply a mathematical bracket.  Its Version-1 semantics are
therefore deliberately local.

Let

\[
C\approx J_y^{-1}
\]

be a finite ordinary binary64 preconditioner proposal at the nominal solution.
AutoIA rigorously bounds the preconditioned residual

\[
\|C F(\widehat y(p),p)\|_\infty\le c,
\]

where `||v||_infinity` is the largest absolute component of vector `v`.

Over the correction box

\[
\|e\|_\infty\le d,
\]

AutoIA rigorously encloses the Jacobian and forms componentwise upper bounds

\[
B\ge |I-CJ_y(\widehat y(p)+e,p)|,
\]

where `I` is the identity matrix and absolute value is componentwise.  It then
computes the rigorous row-sum norm bound

\[
\rho=\|B\|_\infty.
\]

If

\[
\rho<1,
\qquad
\frac{c}{1-\rho}\le d,
\]

then the fixed-point theorem proves a unique solution in that local tube for
every physical input in the AutoIA box.  Each returned solution component is a
dependency-bearing predictor plus the certified correction interval.

This certificate is **local**.  AutoIA does not claim that the nonlinear system
has no other remote solutions outside the certified tube.

## 5. Derivatives and trust boundary

AutoIA does not trust SciPy finite differences or a user-supplied `fprime` as a
proof.  Forward-mode dual propagation differentiates the residual through the
same currently supported scalar language used by validated execution.  The
nominal Jacobian is proposal information only.  The proof-critical derivative
and Jacobian bounds are themselves AutoIA validated values and are ranged with
the trusted backend.

The current derivative surface includes ordinary arithmetic, constant real
powers, `sqrt`, `exp`, `log`, `log10`, `sin`, and `cos` through their supported
AutoIA semantics.  A proof-critical derivative through `erf` is deliberately
rejected in this checkpoint until the derivative's `2/sqrt(pi)` constant has a
separate trusted enclosure.  This is a fail-closed limitation, not a silent
floating-point approximation.

## 6. SciPy behavior that is and is not trusted

Trusted:

* AutoIA outward-rounded arithmetic and elementary kernels;
* AutoIA dependency-bearing residual and derivative propagation;
* bracket endpoint and derivative-sign proof for validated `root_scalar`;
* contraction/Krawczyk-style residual and Jacobian proof;
* the returned predictor-plus-correction enclosure.

Not trusted:

* SciPy's iteration history;
* SciPy's convergence tolerance as a proof tolerance;
* SciPy finite-difference derivatives;
* the nominal floating-point root itself.

The SciPy result is useful because it is usually an excellent proposal around
which to build a small certificate.  If the proposal is poor or the theorem
conditions cannot be proved, AutoIA fails closed.


### Cross-module pure-Python helpers

The same semantics apply when the solve is reached through an imported
source-backed Python helper, including `module.helper(...)`. AutoIA creates a
private prepared execution view only when semantic values reach that helper;
the imported module, SciPy, and NumPy are not monkey-patched or mutated.
Compiled-extension helper bodies remain outside this source-level preparation
boundary unless they expose a separately supported semantic primitive.

The canonical tutorial demonstration is `examples/16_cross_module_implicit_solves.py`,
where an external wrapper imports an ordinary engineering helper module that
owns both a bracketed `root_scalar` call and a square-system `fsolve` call.

## 7. Current Version-1 limitations

### `root_scalar`

* requires a finite constant bracket;
* supports bracketed methods only;
* requires a uniform strict sign change for every physical input in the current
  AutoIA box;
* requires the residual derivative with respect to the root variable to be
  provably nonzero with one sign across the bracket, using bounded internal
  bracket subdivision when necessary;
* may require physical input subdivision when one input box spans different
  root branches or destroys a uniform bracket proof.

### `fsolve`

* requires a square residual system;
* requires a finite constant numeric `x0` because `x0` is an algorithmic
  proposal, not a physical uncertain input;
* certifies a unique solution only in a local tube around the implicit predictor;
* can fail near singular Jacobians, bifurcations, or multiple/merging solution
  branches;
* preserves the common SciPy `full_output=True` tuple shape and inserts the
  AutoIA certificate into the returned `infodict`.

Neither primitive promises successful certification of every mathematically
solvable nonlinear system.  The Version-1 contract is that recognized calls are
either certified rigorously or rejected explicitly.

## 8. Why this belongs in AutoIA

Unlike downstream GP application kernel or acquisition theorems, an implicit nonlinear solve is a
general computational semantic used throughout engineering software.  The
implementation therefore belongs in AutoIA's semantic layer.

The design also incorporates the general lesson from the downstream GP application application
study: shared physical dependence should not be destroyed earlier than
necessary.  The implicit predictor carries dependence on the original AutoIA
values into downstream calculations; only the remaining theorem-certified
correction is represented as an independent bounded uncertainty.
