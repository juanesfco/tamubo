# AutoIA Version-1 engineering validation corpus

This directory separates **ordinary engineering model code** from the AutoIA
validation harness.

The modules under `models/` import ordinary Python/NumPy only. They do not import
AutoIA, construct interval objects, query AutoIA internals, or branch on whether a
value is validated. `cases.py` supplies the uncertainty boxes and expected release
behavior, while `tools/run_engineering_corpus.py` performs certification and records
results.

The current corpus is an **internal canonical-engineering corpus**. Its model
implementations were written as ordinary numerical routines from standard
engineering equations, not as AutoIA API examples. This is useful for integration
hardening, but it is not yet a substitute for the roadmap requirement to run
pre-existing code authored outside the AutoIA effort. Step 19b therefore remains
open after this phase.

## Current model families

- Euler-Bernoulli cantilever stress, deflection, and first bending frequency
- thin-walled pressure-vessel membrane/von-Mises stress
- first-order lumped thermal transient
- ideal-gas isentropic nozzle relations
- Darcy-Weisbach pipe loss with an uncertain laminar/turbulent branch
- parameter-dependent thermal-resistance network using `np.linalg.solve`
- counterflow heat-exchanger LMTD calculation
- two-degree-of-freedom modal analysis using `np.linalg.eigvals`, retained as an
  explicit expected unsupported Version-1 boundary

The LMTD model also has a deliberately invalid input box containing a physical
point where the literal source expression evaluates as `0/0`. That case is an
expected certification failure, not an AutoIA defect.

## Running the corpus

From the source-tree root:

```bash
python tools/run_engineering_corpus.py \
  --output ENGINEERING_CORPUS_RESULTS.json \
  --text-output ENGINEERING_CORPUS_RESULTS.txt
```

The runner evaluates ordinary floating-point samples as a compatibility sanity
check. **Sampling is not the soundness proof.** AutoIA's machine-rigorous guarantee
comes from its trusted arithmetic and proof kernels; the engineering corpus tests
whether realistic source structures execute unchanged and fail clearly when they
cannot be certified.

## Adding an externally authored model

Prefer a module-level function that can run numerically without AutoIA. Preserve
its ordinary coding style and do not rewrite it merely to fit the current supported
subset. Put AutoIA-specific bounds and expected behavior in the validation manifest,
not in the model itself. Record provenance and license information separately when
an external source is used.
