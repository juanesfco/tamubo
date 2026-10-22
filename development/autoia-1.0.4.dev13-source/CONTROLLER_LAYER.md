# Controller and subdivision layer in AutoIA 0.12.0.dev9

`0.12.0.dev8` extracted strategy decisions from the historical Version-0.11
compatibility module into `autoia/controller/`.  In `dev9` the compatibility
module itself is retired from the production package.

The controller is **not trusted numerical arithmetic**.  It decides which sound
method to try, how much optional work to spend, and when to subdivide.  A poor
controller decision may waste time or return a wider enclosure; it must not make
an accepted enclosure unsound.

## Modules

- `controller/policies.py` defines range/subdivision/branch policies.
- `controller/configuration.py` defines the Step-20a representation/order reference portfolio and diagnostics.
- `controller/ranging.py` implements adaptive final-output ranging.
- `controller/execution.py` orchestrates one certified model execution.
- `controller/subdivision.py` implements fixed and targeted adaptive subdivision.
- `controller/diagnostics.py` freezes the dictionary schemas used for final-range,
  proof-escalation, and subdivision diagnostics through `TypedDict` definitions.

The public API imports these extracted implementations directly.  The historical
compatibility module no longer exists in the production package.

## Range-controller logic

For a polynomial output `P(q)+R`, the adaptive final controller first intersects
rigorous termwise and monotonicity ranges.  Let `U` be the set of active input
directions for which derivative-sign analysis remains unresolved.

If `U` is empty, monotonicity has reduced the polynomial extrema to certified
corners, so Bernstein is skipped.  If `U` is nonempty, the controller estimates
the cost of Bernstein restricted to those unresolved directions and runs it only
when that reduced cost fits `RangePolicy.bernstein_max_coefficients`.

These choices affect sharpness and cost only.  Each retained range is already a
rigorous enclosure before optional tightening is attempted.

## Proof-policy context

Representation-level proof obligations, currently notably reciprocal nonzero
certification, need temporary access to the active proof-cost policy and an event
recorder.  Earlier prototypes implemented this with mutable module-level stacks.

`dev8` replaces those stacks with Python `contextvars.ContextVar` objects.
`RangePolicyContext` installs the current policy and event list using ContextVar
tokens and resets those tokens on exit.  This gives correct nested restoration
and prevents concurrent threads/tasks from sharing AutoIA policy state through a
mutable global stack.

The regression suite includes a synchronized two-thread test in which different
proof budgets are active simultaneously.  Each thread observes only its own
policy and event list.

## Subdivision

Fixed subdivision partitions the input box into a finite Cartesian family and
certifies every branch independently.

Adaptive subdivision acts only when final-range diagnostics indicate both:

1. unresolved monotonicity directions remain; and
2. monotonicity-reduced Bernstein exceeds the current cost cap.

Candidate split directions are restricted to unresolved variables. As of
`0.19.0.dev1`, the default `SubdivisionPolicy(split_method="cheap")` does not
execute candidate child models merely to choose a direction. For each candidate
direction it shifts the already-certified parent output polynomial to the two
hypothetical child centers, halves the corresponding coordinate radius, and
cheaply predicts:

1. reduction in unresolved directions;
2. reduction in worst monotonicity-reduced Bernstein cost;
3. reduction in cheap output-hull width;
4. local child-level monotonicity resolution and aggregate Bernstein burden; and
5. derivative ambiguity as a final tie breaker.

The score is lexicographic and has no fitted weights. The child-level local
resolution quantity is also used as an acceptance cue because it captures
genuine stationary-point progress where one child becomes monotone even though
the union of both children still contains an unresolved direction.

After the cheap score selects a direction, AutoIA certifies only those two
children. The realized child metrics are checked against the same structural/
width acceptance thresholds; an optimistic score therefore cannot force a
cascade of useless splits.

`SubdivisionPolicy(split_method="lookahead")` retains the historical reference
controller. It fully certifies every probed candidate direction and ranks them
lexicographically by observed unresolved reduction, observed reduced-Bernstein
cost reduction, observed hull-width reduction, and derivative ambiguity. This
mode remains the research oracle used to benchmark the cheap score.

## Diagnostics

For migration compatibility, runtime diagnostics remain ordinary dictionaries.
Their intended fields are now explicit in `controller/diagnostics.py` as:

- `FinalRangeDiagnostic`;
- `ProofRangePolicyEvent`;
- `BranchControllerMetrics`;
- `SubdivisionProbeDiagnostic`; and
- `SubdivisionEvent`.

This freezes meanings without changing Version-0.11 serialization or numerical
behavior during the structural migration.

## Regression standard

The extracted controller is compared directly against the untouched historical
Version-0.11 oracle at `tests/support/oracle_v11.py`.  Tests require bit-for-bit equality of certified
output endpoints for representative default/adaptive range policies and fixed
subdivision. Adaptive subdivision is compared bit-for-bit in explicit
`split_method="lookahead"` mode; the new cheap default is benchmarked against
that reference controller for decision quality and cost.  Diagnostic dictionaries are compared
after canonicalizing interval objects so equivalent oracle/new interval classes
are compared by binary64 endpoint bits.



## Step 20a automatic configuration (`0.24.0.dev1`–`0.24.0.dev3`)

The existing controller chooses ranging and subdivision strategy *within* one
representation/order run. Step 20a adds a distinct non-trusted controller above
that layer.

`0.24.0.dev1` introduced an expensive bounded reference portfolio that intersects
independently rigorous representation/order certificates. `0.24.0.dev2` adds the
release-oriented `ConfigurationPolicy(mode="auto")` selector.  It starts with
one baseline and retries only when baseline structural diagnostics expose either
(1) unstructured remainder dominating retained physical-polynomial variation or
(2) shared approximation ancestry recombining in an output whose retained
physical-polynomial variation has cancelled to zero.  The first mechanism can
justify one higher polynomial order; the second can justify shared-generator
mode.

Approximation ancestry is diagnostic metadata only and never enters trusted
arithmetic.  Prepared automatic sessions create alternate candidate sessions
lazily and reuse them across boxes.  The expensive `mode="reference"` policy is
retained as the research oracle.  See `STEP20A_AUTOMATIC_CONFIGURATION.md`.


## Step 20a Phase 3 novice API freeze (`0.24.0.dev3`)

The public zero-knob calls `autoia(model, box)` and `prepare(model)` now select
`ConfigurationPolicy(mode="auto")` automatically. The release baseline remains
plain polynomial/order 4. Explicit `order=` or `representation=` is treated as
a fixed expert/reproducibility request unless an explicit configuration policy
is also supplied. This rule makes the beginner path automatic without removing
research control.

Configured results print only certified output intervals, the machine-rigorous
flag, and a compact statement of whether an optional retry occurred.
`explain_configuration()` exposes a short rationale; the full diagnostic
dictionary remains available for research/debugging but is not part of what a
new user must interpret.

No trusted arithmetic changed in Phase 3. A configuration decision can only
change execution cost or enclosure sharpness.
