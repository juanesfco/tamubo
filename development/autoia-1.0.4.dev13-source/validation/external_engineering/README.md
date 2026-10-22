# Pre-existing external engineering validation corpus

This directory is the external-provenance continuation of AutoIA Version-1 Step 19b.
The engineering formulas originate in public projects that predate this validation work.
The vendored source modules do **not** import AutoIA.  `adapters.py` is separate validation
scaffolding: it unpacks AutoIA's one-vector model input, supplies fixed engineering constants,
and selects outputs, while AutoIA's local-dispatch hook prepares the external function itself.

## External sources

- **fluids 1.3.1**, MIT, Caleb Bell.  Pinned source commit `e49c8b0`.  The corpus uses
  `Clift_Gauvin`, `Morsi_Alexander`, `Churchill_1977`, and `Swamee_Jain_1976`.
- **ht 1.2.0**, MIT, Caleb Bell.  PyPI provenance identifies commit
  `980e16062beb09d4533b5f83cad47e736f33461c`.  The corpus uses
  `cylindrical_heat_transfer`.
- **grheat 0.5.1-era source**, MIT, Scott Prahl.  The corpus uses the computational body
  of `Plane._instantaneous` with a minimal object scaffold for its stored constants.

Long upstream documentation strings are not reproduced.  The compact executable statements
under test are retained.  License notices are under `licenses/`, and exact retrieval metadata
and local hashes are recorded in `SOURCE_MANIFEST.json`.

## Validation policy

Each case is first run with ordinary binary64 inputs.  The same external source logic is then
run with AutoIA over a physical uncertainty box.  Ordinary point sampling of the box is used
only as a compatibility sanity check; it is not the interval-soundness proof.

The first external run found one general compatibility gap: `math.log10` in the pre-existing
Swamee-Jain friction-factor implementation.  AutoIA `0.23.0.dev2` adds rigorous `log10` using
`log10(x) = ln(x)/ln(10)`, with both numerator and denominator enclosed by trusted interval
arithmetic.  No external source was rewritten to avoid the operation.
