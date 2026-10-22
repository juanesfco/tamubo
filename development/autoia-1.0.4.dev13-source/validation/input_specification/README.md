# Physical-input specification audit

`audit_input_specification.py` independently checks the AutoIA 1.0.1 exact-real
physical-input boundary using Python `Fraction` arithmetic as the reference.
It verifies 81 deterministic and 10,000 seeded randomized exact-rational
interval specifications, for 10,081 total containment checks.

The audit is deliberately narrower than the normal AutoIA regression suite. It
checks the theorem

\[
[\ell^\star,u^\star]\subseteq[\ell_{64},u_{64}],
\]

where the starred endpoints are the exact real values requested by the user and
the binary64 endpoints are the finite proof interval constructed by
`Interval.enclosing`.

Run:

```bash
python validation/input_specification/audit_input_specification.py
```
