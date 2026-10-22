# AutoIA numbered examples

Run the numbered examples with the same Python interpreter used to install
AutoIA. They are ordinary Python programs intended to demonstrate supported
semantics rather than internal implementation APIs.

| # | File | Purpose |
|---:|---|---|
| 00 | `00_smoke_test.py` | Minimal import and interval execution. |
| 01 | `01_scalar_nonlinear.py` | Scalar nonlinear arithmetic. |
| 02 | `02_vector_numpy.py` | NumPy vector semantics. |
| 03 | `03_linear_solve.py` | Ordinary `numpy.linalg.solve`. |
| 04 | `04_branching.py` | Path-aware branching. |
| 05 | `05_subdivision.py` | Physical subdivision. |
| 06 | `06_prepared_many_box.py` | Prepared repeated execution. |
| 07 | `07_generator_cancellation.py` | Shared-generator dependency. |
| 08 | `08_normal_functions.py` | Normal PDF/CDF functions. |
| 09 | `09_failure_handling.py` | Explicit fail-closed behavior. |
| 10 | `10_gaussian_tail_expression.py` | Generic nonlinear normal-tail expression. |
| 11-14 | application theorem examples | Generic theorem composition/monotonicity interfaces. |
| 15-16 | implicit solve examples | `root_scalar` and `fsolve` semantics. |
| 17 | `17_regular_grid_interpolator.py` | Fixed-data linear interpolation. |
| 18 | `18_recognized_unsupported_eigh.py` | Recognized unsupported operation. |
| 19-22 | representation/domain/SPD examples | Expert representations, domain proofs, matrix families. |
| 23-25 | execution/scaling examples | Parallel execution, many-box workflows, SPD family. |
| 26 | `26_exact_input_specification.py` | Stored-float vs exact-real input endpoints. |
| 27 | `27_gpytorch_transparent_exact_gp.py` | Full ordinary GPyTorch fit and transparent posterior evaluation. |
| 28 | `28_botorch_transparent_single_task_gp.py` | Full BoTorch `SingleTaskGP` fit and transparent posterior evaluation. |
| 29 | `29_sklearn_transparent_gpr.py` | Full ordinary scikit-learn `GaussianProcessRegressor` fit and transparent posterior evaluation. |
| 30 | `30_handcoded_gp_expected_improvement.py` | Hand-coded exact GP plus ordinary analytic EI; no acquisition semantics in AutoIA core. |
| 31 | `31_expected_improvement_user_theorems.py` | Advanced EI proof portfolio using user-known EI monotonicity, GP nonnegativity, and RBF box geometry. |
| 32 | `32_handcoded_analytic_ehvi.py` | Hand-coded analytic two-objective q=1 EHVI passed directly to AutoIA. |
| 33 | `33_ehvi_user_gaussian_theorem.py` | Advanced EHVI proof portfolio using user-known Gaussian positive-part and finite-band monotonicity. |
| 34 | `34_botorch_analytic_ehvi.py` | Ordinary BoTorch analytic EHVI built and evaluated inside the engineering function. |
| 35 | `35_botorch_ehvi_user_gaussian_theorem.py` | Advanced BoTorch EHVI proof portfolio using the same fitted posterior plus a user-known Gaussian cell theorem. |
| 36 | `36_additional_numpy_elementary.py` | Transparent `np.tanh`, `np.sinh`, `np.arctan`, `np.abs`, `np.log1p`, and `np.cosh`. |
| 37 | `37_composed_numpy_elementary_stress.py` | Composed elementary-function dependency stress. |
| 38 | `38_scipy_linprog_optimal_value.py` | Certified optimal value of ordinary `scipy.optimize.linprog`. |
| 39 | `39_scipy_linprog_lower_only.py` | One-sided lower-only LP value certificate. |
| 40 | `40_closure_object_semantic_dispatch.py` | Supported SciPy semantics through closure/object calls. |
| 41 | `41_scipy_linprog_equalities.py` | Ordinary LP equality constraints. |
| 42 | `42_persistent_manybox_bnb.py` | Persistent prepared process execution for B&B-style batches. |
| 43 | `43_scipy_linprog_rlt_obbt.py` | Automatic lifted McCormick/RLT lower bound plus certified OBBT. |

Examples 27 and 28 skip explicitly when their optional third-party dependencies
are not installed. No manual GP freeze call is used in either normal workflow.
