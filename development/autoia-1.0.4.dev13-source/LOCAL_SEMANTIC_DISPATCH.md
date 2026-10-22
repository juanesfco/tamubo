# Local validated-operation dispatch in AutoIA 0.14.0.dev1

## Purpose

Earlier research versions intercepted a semantic linear solve by temporarily
assigning AutoIA's traced solve function to `numpy.linalg.solve`.  That changed
process-global NumPy state.  Even though the original function was restored on
exit, the mechanism was unsuitable for a Version-1 architecture because
unrelated threads, nested code, or other libraries could observe the temporary
replacement.

`0.12.0.dev7` introduced that global mutation from the active implementation.

## Execution model

For an ordinary Python engineering function `model`, AutoIA creates a *shadow
copy* of the function's global namespace.  The original function object and its
original globals dictionary are not changed.

Within the shadow namespace:

- names bound to the NumPy module are replaced by a lightweight read-only NumPy
  proxy;
- names bound to `numpy.linalg` are replaced by a lightweight read-only linalg
  proxy;
- names bound directly to the original `numpy.linalg.solve` function are
  replaced by the active AutoIA semantic solve dispatcher;
- module/global NumPy and standard-library `math` bindings for `sqrt`, `exp`,
  `log`, `sin`, and `cos`, including direct aliases, are replaced by local
  validated unary dispatchers;
- ordinary Python helper functions that share the same original globals
  dictionary are cloned to execute against the same shadow namespace.

All other NumPy and `math` attributes are forwarded to their real modules.
Proposal calculations *inside AutoIA's semantic solver* continue to use the real
`numpy.linalg.solve` directly.

Thus, for a model containing

```python
u = np.linalg.solve(A, b)
```

only the private execution namespace sees AutoIA's dispatched `solve`.  The
process-wide object `numpy.linalg.solve` remains unchanged.

## Supported import styles

The reference dispatcher recognizes module/global bindings produced by common
styles such as

```python
import numpy as np
import numpy.linalg as la
from numpy.linalg import solve
import math
from numpy import sin
from math import cos
```

including aliases of these module-level bindings.

It also follows ordinary Python helper functions defined in the same global
namespace as the model. Therefore a model may call a same-module helper which
then calls `np.linalg.solve`. Imported source-backed pure-Python helpers are
prepared locally and lazily when AutoIA semantic values actually reach them.
This includes both `from helper import function` and the common module form
`import helper; helper.function(...)`. The imported source module and its globals
are never mutated.

## Deliberate current boundary

The current dispatcher does not rewrite Python bytecode or install a global import
hook.  Therefore the following are not yet semantically intercepted:

- a local `import numpy` or `import math` executed inside a function body;
- helper functions implemented in compiled extension modules or source modules
  whose execution environment cannot be safely cloned;
- compiled C/C++/Fortran extension code that calls NumPy or another solver
  internally;
- arbitrary callable objects whose `__call__` execution environment cannot be
  safely cloned.

These cases must either execute entirely through supported generic validated
operations or fail closed.  They are not grounds for restoring global
monkeypatching.

## Reentrancy and threads

Removing the NumPy monkeypatch eliminated the largest process-global semantic
interception hazard.  In dev8, AutoIA's remaining proof-policy/event stacks were
also replaced with Python `contextvars`, so AutoIA-owned dynamic policy state is
context-local and nested contexts restore exactly.  A synchronized two-thread
regression verifies that simultaneous AutoIA contexts do not share policy/event
state.  This does not make arbitrary user model code or third-party compiled
libraries thread-safe; those may contain their own global mutation or other side
effects outside AutoIA's contract.

## Regression requirement

The dispatch tests require that:

1. real `numpy.linalg.solve` has the same identity before, during observation
   from the user model, and after AutoIA execution;
2. same-module helper calls are intercepted locally;
3. directly imported and module-qualified source-backed pure-Python helpers are
   prepared lazily without mutating their source modules;
4. a module-level `from numpy.linalg import solve` alias is intercepted locally;
5. unsupported local imports fail closed;
6. no active AutoIA source assigns to `np.linalg.solve`; and
7. all prior numerical and certificate regressions remain unchanged.

The untouched historical Version-0.11 oracle now lives only at
`tests/support/oracle_v11.py`.  It intentionally retains the historical
monkeypatch implementation solely for development-time regression and is not
installed with AutoIA.


## Constant powers

The shadow NumPy/math namespace also dispatches `numpy.power` and `math.pow`
when the base is validated and the exponent is one constant real scalar.
Direct `**` and built-in `pow(base, exponent)` use `ValidatedValue.__pow__`
directly and require no proxy.  As with the unary elementary functions, no
process-global NumPy or `math` object is modified.


## Step-4 array dispatch

The private NumPy proxy now also owns the frozen Step-4 array-construction,
reduction, dot/matmul, trace, and norm entry points.  Constructors default to
`dtype=object` only inside the shadow execution namespace when the user did not
explicitly request a dtype.  Direct module-level aliases of these NumPy
functions are replaced locally in the same manner as `solve` and elementary
functions.  Process-wide NumPy remains unchanged.

## Step 4b SciPy compatibility

The same shadow-namespace mechanism now recognizes common module-level SciPy
normal-distribution imports.  It wraps only private references in the execution
namespace and never patches SciPy itself.  Supported forms are documented in
`NORMAL_API_COMPATIBILITY.md`.  Runtime-local imports and helpers from other
modules remain outside the current tracing boundary.


## Path-aware Boolean execution

As of `0.17.0.dev1`, scalar comparisons of validated values are handled by a
context-local branch runtime rather than by AST rewriting or process-global
monkeypatching.  Same-module helpers participate automatically because they run
in the same prepared shadow namespace.  See `PATH_AWARE_EXECUTION.md`.

## Explicit deferred linear-algebra diagnostics (`0.23.0.dev1`)

The engineering validation corpus exposed a diagnostics problem for a common
modal-analysis boundary. A validated call to `np.linalg.eigvals` previously fell
through to real NumPy and eventually failed with an indirect object-dtype
`isfinite` error. The private linalg proxy now recognizes the observed deferred
eigenvalue family `eig`, `eigvals`, `eigh`, and `eigvalsh`, including module-level
direct aliases such as `from numpy.linalg import eigvals`. If validated uncertain
data reach one of these functions, AutoIA raises an explicit unsupported-operation
error naming the requested operation. Ordinary numeric calls still forward to
NumPy unchanged. This is a fail-closed diagnostics improvement only; AutoIA does
not implement an eigenvalue semantic primitive in Version 1.


## Closure-variable and captured-object dispatch (`1.0.4.dev10`)

The private shadow namespace now extends through ordinary Python closure cells.
A factory may capture a pure-Python function and call it later, for example

```python
fn = response
def model(p):
    return fn(p[0])
```

and `fn` receives the same local NumPy/SciPy semantic preparation as a global
helper. Source-level user objects captured by a closure are also proxied locally,
so a pattern such as `prob.g(p[0])` can reach a supported `fsolve`, `linprog`, or
other semantic inside `g`; nested `self.other_method(...)` calls stay behind the
same local proxy. The original closure object, source module, and SciPy/NumPy
globals are not monkey-patched. Opaque compiled objects and third-party objects
with dedicated semantics are not generically proxied.

## dev11: callable attributes on captured Python objects

A source-level object captured by a closure may store a pure-Python response
function directly on an instance attribute:

```python
problem.g = response

def model(p):
    return problem.g(p[0])
```

Unlike a class-defined method, `problem.g` is a plain Python function under the
normal descriptor rules.  dev11 recognizes this case in the private object proxy
and lazily prepares that function when validated values reach it.  This gives
`problem.g(x)` the same semantic behavior as hoisting `g = problem.g` into the
closure.  Opaque extension callables are not guessed or wrapped.
