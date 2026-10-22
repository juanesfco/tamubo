# AutoIA 1.0.2 Release Notes

AutoIA 1.0.2 is a backward-compatible V1 maintenance release focused on two
release-quality issues: transparent supported fitted-GP posterior semantics and
strict standalone-package hygiene.

## Transparent GPyTorch and BoTorch semantics

The normal user workflow no longer requires a manual GP freeze adapter. During
local semantic execution, AutoIA can recognize a deliberately narrow supported
class of fitted scalar exact-GP objects and substitute validated posterior
semantics while leaving the user's original source ordinary.

A supported GPyTorch workflow may therefore contain ordinary code such as:

```python
x = torch.tensor([[p[0]]], dtype=torch.float64)
posterior = model(x)
return posterior.mean[0]
```

inside a function passed directly to `autoia(...)`. A supported BoTorch
`SingleTaskGP` may likewise use its ordinary `model.posterior(x)` surface. The
complete live examples are numbered examples 27 and 28.

The initial supported subset is intentionally narrow and fail-closed:

- scalar exact GP;
- RBF kernel;
- Matern kernel with `nu = 1/2`, `3/2`, or `5/2`;
- optional scale kernel;
- zero or scalar constant mean;
- Gaussian observation noise;
- common affine input transforms compatible with `Normalize` and
  `InputStandardize`;
- affine outcome standardization compatible with `Standardize`;
- fitted source state on CPU, CUDA, or MPS provided it can be copied to CPU
  binary64.

Unsupported structures fail explicitly. AutoIA does not claim arbitrary Torch
validated execution.

The lower-level functions in `autoia.integrations` remain available as expert
APIs, but they are not required by the normal user examples.

## Trust boundary

GP fitting, autograd, training convergence, and accelerator arithmetic are not
certified. Fitted numerical parameters are copied to CPU binary64 and treated as
fixed stored data. Proof-critical posterior evaluation over uncertain query
inputs is reconstructed through AutoIA's trusted validated arithmetic.

The independent 80-digit posterior audit passes 504 containment checks across
RBF and all three supported Matern profiles.

## Input specification retained

The 1.0.1 physical-input hardening remains unchanged:

- `Interval(lo, hi)` uses the exact stored binary64 endpoint values;
- `Interval.enclosing(...)` rigorously encloses intended exact decimal/rational
  endpoints;
- `Interval.outward(...)` is explicitly one-ULP outward padding of already
  stored floats and is not a general provenance proof.

The independent exact-rational audit passes 10,081 checks.

## Standalone release boundary

The 1.0.2 release removes historical downstream-package references from the
public AutoIA distribution and adds a regression scanner that enforces the
standalone boundary. The final release audit checks the working source, extracted
source archive, extracted wheel, extracted text of all five PDFs, and extracted
complete-release archive.

## Validation summary

- 388 named pytest cases collected;
- 386 passed;
- 2 optional live-GP tests skipped because the optional third-party packages are
  unavailable on the build host;
- 0 failures;
- 2,405 pytest subtests passed;
- 10,081 exact-rational input-containment checks passed;
- 504 independent 80-digit GP posterior containment checks passed;
- 29/29 numbered source examples exited successfully;
- 29/29 numbered clean-wheel examples exited successfully;
- examples 27 and 28 are explicit optional-dependency skips on this host;
- installed wheel package code matches the source package code.

## Documentation

The synchronized documentation set is:

- `AUTOIA_MATHEMATICAL_FOUNDATIONS_V1.pdf`
- `AUTOIA_TUTORIAL_MANUAL_V1.pdf`
- `AUTOIA_USER_DEVELOPER_MANUAL_V1.pdf`
- `AUTOIA_TEST_VALIDATION_STRATEGY_V1.pdf`
- `AUTOIA_V2_PATH_FORWARD.pdf`

All five were rebuilt for 1.0.2 and rendered for visual verification.
