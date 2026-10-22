# AutoIA 1.0.4.dev4 development notes

Purpose: continue the transparent fitted-GP compatibility work using ordinary
BoTorch analytic EHVI as an external acceptance case.

Implemented in this checkpoint:

- accept ``posterior_transform=None`` at the transparent BoTorch posterior boundary;
- support the common independent multi-output ``SingleTaskGP`` state with no extra
  model batch dimensions;
- preserve output-specific mean, kernel lengthscale/outputscale, observation noise,
  and Standardize affine output parameters;
- return BoTorch-shaped validated posterior mean/variance arrays;
- support the deterministic validated tensor operations exercised by analytic EHVI:
  broadcast arithmetic, clamp, sqrt, exp, erf, stack, gather, product, and sum;
- adapt fixed ``torch.distributions.Normal`` helper objects in private BoTorch
  acquisition subclasses so Normal PDF/CDF arithmetic can operate on validated
  values without teaching AutoIA the EHVI formula;
- keep non-None posterior transforms, correlated multitask models, model batches,
  variational GPs, and Monte Carlo qEHVI/qNEHVI outside the supported boundary.

The source regression suite after these changes is 394 passed, 3 optional live
GP-package skips, 0 failures, with 2,405/2,405 pytest subtests passing.

A live BoTorch 0.18.1 / GPyTorch 1.15.2 acceptance run is still required.
