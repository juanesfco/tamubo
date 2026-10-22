# AutoIA V1 Documentation Index

**Current release:** AutoIA `1.0.2`

This directory contains the coordinated V1 documentation set. The documents have distinct purposes so mathematical proof statements, user workflow, implementation details, validation evidence, and future research are not mixed together.

## 1. Mathematical foundations

### `AUTOIA_MATHEMATICAL_FOUNDATIONS_V1.tex`
Self-contained mathematical specification of AutoIA V1. It defines all notation and develops the proof architecture from rigorous physical-input specification through interval, polynomial, shared-generator, pathwise, domain-refinement, linear-solve, implicit-solve, interpolation, structured SPD matrix-family, ranging, automatic-configuration, application-theorem, repeated many-box, heterogeneous-assist, and trusted binary64 layers.

### `AUTOIA_MATHEMATICAL_FOUNDATIONS_V1.pdf`
Compiled rendering of the LaTeX source for convenient reading. The `.tex` file is the editable source of record.

## 2. Tutorial manual

### `AUTOIA_TUTORIAL_MANUAL_V1.md` / `AUTOIA_TUTORIAL_MANUAL_V1.pdf`
Progressive copy-paste tutorial for new users. It begins with `autoia(model, box)`, explains the three V1.0.2 physical-input endpoint semantics, and builds through arrays, linear solves, branching, representations/range methods, prepared execution, validated `root_scalar`/`fsolve`, fixed-data linear `RegularGridInterpolator`, application theorems, and failure interpretation.

The runnable numbered examples in `../examples/` are the executable companion to this tutorial.

## 3. User and developer/code manual

### `AUTOIA_USER_DEVELOPER_MANUAL_V1.md` / `AUTOIA_USER_DEVELOPER_MANUAL_V1.pdf`
Operational API reference and source-code map. It documents public classes/functions, supported-language boundaries, results and policies, semantic dispatch, repeated-query execution, module responsibilities, trusted versus proposal/controller code, extension recipes, debugging, testing, and the release checklist.

## 4. V2 path forward

### `AUTOIA_V2_PATH_FORWARD.md` / `AUTOIA_V2_PATH_FORWARD.pdf`
Release-gated plan for post-V1 work. Major proposed research directions include validated symmetric/generalized-symmetric eigenvalues, ODE semantics, carefully scoped optimization, richer interpolation, stronger dependency representations, constrained physical domains, large-scale/GPU acceleration, language coverage, theorem/plugin infrastructure, and certificate explainability.


## 5. Validation strategy and test evidence

### `AUTOIA_TEST_VALIDATION_STRATEGY_V1.md` / `AUTOIA_TEST_VALIDATION_STRATEGY_V1.pdf`
Explains the authoritative 388-test AutoIA 1.0.2 pytest collection, why each test family was chosen, the final grouped-run counts, all 58 test files, and an appendix containing every collected test ID. It also distinguishes the 388 named tests from 2,405 pytest subtests, 29 numbered examples, optional live integration tests and standalone high-precision audit scripts.

## 6. Executable capability map

### `../examples/COVERAGE_MATRIX.md`
Maps every major V1 capability and deliberate unsupported boundary to one or more numbered examples.

### `../examples/README.md`
Short description and run instructions for Examples `00` through `26`, plus hardware-specific Apple MPS and NVIDIA CUDA smoke checks.

## 7. Focused design/history notes

Repository-root Markdown files preserve design decisions and verification history for individual development steps. Important current references include:

- `CONTRACT.md`
- `INPUT_SPECIFICATION.md`
- `TRUSTED_KERNEL.md`
- `ENCLOSURE_LAYER.md`
- `POLYNOMIAL_LAYER.md`
- `SHARED_GENERATOR_REPRESENTATION.md`
- `PATH_AWARE_EXECUTION.md`
- `SEMANTIC_LINEAR_SOLVE.md`
- `IMPLICIT_SOLVE_SEMANTICS.md`
- `APPLICATION_THEOREM_INTERFACE.md`
- `AUTOIA_APPLICATION_LESSONS.md`
- `FAILURE_SEMANTICS.md`
- `ROAD_TO_V1.md`
- the Step 20 verification notes.

These are useful for implementation history. The five coordinated documents above plus `INPUT_SPECIFICATION.md` are the current V1 reference set.

## 8. Historical documents

Files named `*_v021.*` and `NEW_USER_TUTORIAL.md` are retained as historical checkpoints. They predate the final V1 semantic additions such as validated implicit solves, exact-affine comparison hardening, proof-driven reciprocal refinement, fixed-data linear `RegularGridInterpolator`, structured SPD matrix-family semantics, multicore many-box proof, and MPS/CUDA proposal-assist support. Do not use them as the current support contract when they conflict with the V1 documents listed above.

## 9. Final release records

### `../V1_EXECUTION_POLICY.md`
Final CPU/GPU scheduling policy frozen at AutoIA 1.0.0 and retained unchanged in 1.0.2.

### `../V1_0_1_RELEASE_NOTES.md`
Release notes for the backward-compatible physical-input specification hardening patch.

### `../V1_0_1_FINAL_VERIFICATION.md`
Authoritative 1.0.2 regression, exact-rational input-enclosure audit, clean-wheel, example, documentation, and transparent-GP, release-policy, and packaging verification record.

### `../V1_FINAL_VERIFICATION.md`
Historical final verification record for the original 1.0.0 freeze.
