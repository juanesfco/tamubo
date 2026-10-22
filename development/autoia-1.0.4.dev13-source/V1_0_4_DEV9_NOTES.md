# AutoIA 1.0.4.dev9 notes

## `scipy.optimize.linprog`: preserve one-sided certificates

Version 1.0.4.dev9 changes the validated `linprog(...).fun` semantic so that a
failed universal-primal-witness search no longer discards an independently
proved weak-duality lower bound.

For a minimization LP, if AutoIA proves

```text
L <= V(q)    for every q in the physical box
```

but cannot prove any fixed primal candidate feasible for every `q`, the result
is now

```text
V(q) in [L, +inf]
machine_rigorous = True
```

with certificate fields

```text
certificate_completeness       lower-only
value_lower_certified          True
value_upper_certified          False
universal_primal_witness_found False
primal_feasible_for_all_q      None
selected_primal_z              None
primal_residual_upper          None
```

A missing witness is not interpreted as a proof of infeasibility. The returned
`+inf` upper endpoint explicitly records that no finite upper value bound was
proved on the whole box.

The previous two-sided behavior is unchanged whenever a universal primal
witness is verified. Unsupported semantics such as `res.x`, equality
constraints, and mixed-integer `integrality` continue to fail closed.
