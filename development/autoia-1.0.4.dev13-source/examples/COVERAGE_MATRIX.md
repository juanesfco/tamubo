# AutoIA V1 example coverage matrix

This matrix is the release-oriented map from important AutoIA V1 behavior to a runnable numbered example.  The examples are intentionally progressive, but they also serve as smoke/regression demonstrations.  `python examples/run_all_examples.py` runs every numbered example in a fresh subprocess.

| Capability / contract | Primary example(s) | What is demonstrated |
|---|---:|---|
| Zero-knob `autoia(model, box)` | 00, 01 | Ordinary engineering function plus physical intervals; automatic release configuration. |
| Physical-input endpoint specification | 26 | `Interval`, `Interval.enclosing`, and `Interval.outward`; exact decimal/rational containment and result provenance. |
| Rigorous scalar arithmetic | 00, 01 | Addition, multiplication, repeated dependence, nonlinear composition. |
| Elementary functions | 01, 08, 09 | `exp`, `sin`, `sqrt`, `log10`, error function/normal functions; real-domain proof behavior. |
| Vector outputs and NumPy assembly | 02 | Object-array construction, vector outputs, `dot`, `sum`, shared uncertain scalars. |
| Parameter-dependent linear systems | 03 | `np.linalg.solve` as a semantic mathematical solve with a certificate. |
| Uncertain Python control flow | 04 | Path enumeration for an uncertain `if`. |
| Exact closed-boundary comparisons | 20 | Strict `>`/`<` and equality at a closed physical-box endpoint. |
| Manual and adaptive subdivision | 05 | Prescribed grid subdivision and adaptive refinement. |
| Proof-driven domain refinement | 20 | A broad reciprocal proof requests physical-box subdivision before failing. |
| Prepared/repeated certification | 06 | `prepare`, exact cache reuse, contained-box reuse, finite many-box execution. |
| Polynomial dependence | 00, 01, 19 | Default physical-input polynomial representation and expert fixed-order use. |
| Shared-generator dependence | 07, 19 | Approximation-error provenance and downstream cancellation. |
| Interval fallback representation | 19 | Expert interval representation comparison. |
| Termwise polynomial range | 19 | Explicit termwise range method. |
| Monotonicity range | 19 | Explicit derivative-based monotonicity method. |
| Bernstein polynomial range | 19 | Exact power-to-Bernstein basis conversion and coefficient range. |
| Automatic configuration intersection | 00, 07, 19, 20 | Baseline plus bounded optional retries; intersection of independent rigorous hulls. |
| Standard-normal PDF/CDF | 08, 10 | Certified normal functions in ordinary model code. |
| Typed fail-closed errors | 09, 18 | Certification failure versus recognized unsupported semantics. |
| Generic EI-like code | 10 | Application-style nonlinear calculation without application-specific theorem code. |
| Exact-GP analytic EI as ordinary code | 30 | Hand-coded fixed exact GP posterior followed by the conventional analytic EI formula, with no acquisition primitive in AutoIA. |
| Advanced EI application theorems | 31 | User-supplied EI monotonicity, latent-variance nonnegativity, RBF box geometry, and rigorous upper-bound portfolios. |
| Analytic two-objective EHVI as ordinary code | 32 | q=1 independent-objective EHVI cell decomposition built from GP posterior quantities and normal PDF/CDF. |
| Advanced EHVI application theorems | 33 | User-supplied Gaussian positive-part monotonicity and finite-band theorem, combined with the generic EHVI enclosure. |
| Application theorem interface | 11–14 | Monotonicity, composition, Lipschitz bounds, and rigorous theorem portfolios. |
| Bracketed scalar implicit solve | 15 | `scipy.optimize.root_scalar`, nominal proposal plus independent validated root certificate. |
| Square nonlinear system solve | 15 | `scipy.optimize.fsolve`, local unique-branch certificate with dependency-bearing predictor. |
| Cross-module implicit solve | 16 | Imported helper owns `root_scalar`/`fsolve`; local lazy preparation only when reached. |
| Fixed-data regular-grid interpolation | 17 | `RegularGridInterpolator(method='linear')`; multiple crossed cells and vector-valued tables. |
| Cross-module regular-grid interpolation | 21 | Imported helper owns a prebuilt lookup table and the query crosses multiple cells. |
| Parameter-dependent SPD matrix families | 22, 25 | Structural symmetry, rigorous positive eigenvalue proof, `slogdet`, determinant, inverse quadratic form, an RBF-like shared-parameter family, and shrinking-box behavior. |
| Certified `scipy.optimize.linprog` optimal value | 38, 39, 41, 43 | Two-sided and one-sided `res.fun` certificates, equality constraints, fixed-dual portfolio, and lifted McCormick/RLT tightening. |
| Polynomial-parametric LP lifting | 43 | Shared physical monomial/product lifts, rigorous McCormick rows, and two cost-capped certified OBBT rounds across an active-set switch. |
| Persistent failure-tolerant B&B batches | 42 plus dev12 acceptance | Prepared process workers and per-box failure reporting for dynamic search. |
| Multicore CPU many-box proof | 23, 24 | `workers="auto"`, `parallel="auto"`, process/thread resolution, explicit execution metadata, compact bounds, prepared reuse, and streaming. |
| Apple MPS / NVIDIA CUDA proposal assist | 23 plus hardware smoke checks | Torch-backed accelerator values are explicitly non-rigorous and final certificates remain independent CPU binary64 proofs. |
| Linear extrapolation / fill semantics | 17 plus regression tests | The semantic implementation supports `fill_value=None` linear extrapolation and finite fixed fill values. |
| Recognized unsupported generalized eigensolve | 18 | `scipy.linalg.eigh` fails with an explicit AutoIA semantic message; no certificate is produced. |
| Expert/reproducibility controls | 05, 07, 19 | Explicit representation, order, range, and subdivision choices. |

## Important V1 boundaries that are intentionally not disguised as examples of support

The following are not claimed as validated V1 semantics: general `scipy.linalg.eigh` eigenvalues/eigenvectors, general ODE/DAE solvers such as `solve_ivp`, arbitrary numerical optimization, uncertain `RegularGridInterpolator` table/grid data, non-linear RGI methods such as `nearest`, `cubic`, `quintic`, and `pchip`, trusted GPU proof arithmetic, arbitrary compiled-extension internals, and generic local imports that bypass the current shadow namespace.  When a recognized boundary is reached, the release goal is an explicit fail-closed diagnostic rather than accidental nominal execution.
