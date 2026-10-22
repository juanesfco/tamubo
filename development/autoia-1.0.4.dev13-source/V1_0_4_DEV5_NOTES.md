# AutoIA 1.0.4.dev5 development notes

Purpose: package the live BoTorch analytic-EHVI acceptance result as advanced
numbered examples while preserving the application/core boundary.

No new EHVI semantic primitive is added to AutoIA core in this checkpoint.

New examples:

- `34_botorch_analytic_ehvi.py`: ordinary BoTorch two-output `SingleTaskGP`,
  `fit_gpytorch_mll`, `FastNondominatedPartitioning`, and analytic
  `ExpectedHypervolumeImprovement` all appear in normal user code.  AutoIA
  transparently certifies the uncertain final query using the general GP/tensor
  semantics introduced in dev4.
- `35_botorch_ehvi_user_gaussian_theorem.py`: fits one fixed BoTorch model and
  uses that exact same fitted state for both the generic EHVI certificate and a
  user-asserted Gaussian/cell theorem.  The theorem first asks AutoIA to enclose
  the BoTorch posterior mean/variance, then uses monotonicity of Gaussian
  positive-part and finite-band expectations to build an independent EHVI upper
  bound.  EHVI nonnegativity supplies the lower bound.

The live dev4 diagnostic motivating Example 35 showed posterior width inflation
of only about 1.00008x--1.00609x, while the full generic BoTorch EHVI width was
about 1865.63x the dense sampled width.  Thus this case localizes the important
loss to downstream EHVI composition rather than the GP posterior.
