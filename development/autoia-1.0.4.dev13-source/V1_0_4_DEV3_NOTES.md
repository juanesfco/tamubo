# AutoIA 1.0.4.dev3 development notes

This development checkpoint adds advanced acquisition-function examples without
adding EI, EHVI, Bayesian-optimization search, or acquisition-optimization
semantics to AutoIA core.

## New numbered examples

- `30_handcoded_gp_expected_improvement.py`: fixed hand-coded exact GP plus ordinary analytic EI.
- `31_expected_improvement_user_theorems.py`: EI proof portfolio using user-supplied posterior/EI monotonicity, latent-variance nonnegativity, and RBF box geometry.
- `32_handcoded_analytic_ehvi.py`: ordinary analytic two-objective, independent-posterior, q=1 EHVI.
- `33_ehvi_user_gaussian_theorem.py`: EHVI proof portfolio using user-supplied Gaussian positive-part and finite-band monotonicity.

## Architectural boundary

The user-defined physical box is not subdivided or changed by these examples.
The acquisition formulas live only in `examples/`.  AutoIA core supplies generic
validated execution, normal PDF/CDF, the public application-theorem context, and
rigorous bound portfolios.

## Observed example behavior

For the EI theorem-motivating box `[0.20,0.40]`, the ordinary generic certificate
was approximately `[-0.55139, 0.27023]`.  The user theorem portfolio tightened it
to approximately `[0, 0.10391]`; the winning upper bound was the posterior plus EI
monotonicity theorem.

For the EHVI theorem-motivating box `[0.265,0.335] x [0.665,0.735]`, the generic
cell-form certificate was approximately `[-2.03594, 2.14649]`.  The user Gaussian
cell theorem produced approximately `[0.0339973, 0.0817242]`, containing the dense
diagnostic range approximately `[0.0385795, 0.0719398]`.

## Regression status

The unchanged AutoIA source-test suite was rerun in bounded groups after the
example additions:

- 393 passed;
- 3 optional live GPyTorch/BoTorch tests skipped on the build host;
- 0 failures;
- 2,405/2,405 pytest subtests passed.

The four new examples were executed directly and their diagnostic samples were
contained in the reported rigorous certificates.
