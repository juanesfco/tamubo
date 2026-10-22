# AutoIA 1.0.2 Final Verification

## 1. Release identity

- Package: `autoia`
- Version: `1.0.2`
- Trusted proof backend: CPU binary64 validated arithmetic
- Optional accelerator role: proposal assistance only, never proof

## 2. Named source regression

The final source collection contains **388 named pytest cases in 59 test files**.
The verified result is:

```text
386 passed
2 skipped (optional live fitted-GP dependencies unavailable)
0 failed
2,405 pytest subtests passed
```

The suite was run in bounded fresh-process groups because the long proof-heavy
collection can exceed the execution harness time budget when run monolithically.
Every constituent group completed successfully.

## 3. Independent mathematical audits

Exact intended-input audit:

```text
10,081 exact-rational endpoint-containment checks
PASS
```

Fitted-GP posterior audit:

```text
profiles = 4
high_precision_posterior_containment_checks = 504
status = PASS
```

The four GP profiles are RBF and Matern `nu = 1/2`, `3/2`, `5/2`. The reference
uses 80-digit arithmetic and interprets frozen binary64 constants as their exact
stored values.

## 4. Transparent GP regression

Dependency-free structural tests verify transparent handling of supported fitted
GP objects captured as globals and closure values, narrow query-tensor
construction, RBF and Matern posterior evaluation, the expert freeze API, and
explicit failure for an unsupported kernel.

Two live tests define and fit real third-party GP models and run ordinary model
code through `autoia(...)` when their optional dependencies are installed. Those
packages are unavailable on this build host, so the two live tests skip
explicitly. Complete runnable examples are distributed as examples 27 and 28.

## 5. Numbered examples

The source and clean-wheel runs both execute numbered examples `00` through `28`:

```text
29 scripts completed successfully
0 script failures
2 explicit optional-dependency skips inside scripts 27 and 28
```

No manual GP freeze call is used in the normal workflow of examples 27 or 28.

## 6. Clean-wheel verification

The release wheel was installed with `--no-deps` into an isolated target. The
installed package reported version `1.0.2`, imported the GP integration surface,
and ran all numbered examples from a working directory outside the source tree.

A recursive diff of the source `autoia/` package and installed wheel `autoia/`
package, excluding runtime caches, was empty.

## 7. Documentation verification

The five synchronized PDFs were rebuilt from their source documents. PDF
preflight confirmed that each file is openable and unencrypted. All pages were
rendered to PNG for inspection; the new GP and validation sections were checked
visually and showed no clipped text, overlap, black-square glyph failures, or
broken page geometry.

## 8. Release-content policy

The release scanner constructs the prohibited downstream token from fragments so
that the scanner source does not contain the token it enforces. Case-insensitive
scans must report zero occurrences in file names and text for the working tree,
clean source archive, extracted wheel, PDF extracted text, and complete-release
archive.

The post-construction scans reported zero occurrences in every required scope:

```text
working source tree = 0
extracted source ZIP = 0
extracted wheel = 0
extracted text of all distributed PDFs = 0
extracted complete-release ZIP = 0
```

## 9. Release acceptance

The source code, tests, independent audits, numbered examples, isolated wheel
install, source/wheel code comparison, documentation rendering, package-boundary
scans, archive-debris checks, and SHA-256 manifest all satisfy the AutoIA 1.0.2
release criteria.
