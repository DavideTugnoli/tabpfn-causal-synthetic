# CI Preservation Final Results

Status: final.

This lightweight bundle contains the final empirical CI-preservation results:

- comparison: 40 dataset/training-size cells, `N <= 500`;
- interventional: 54 dataset/training-size cells, `N <= 1000`;
- exactly 100 identical repetitions per condition and cell;
- 11 discovered-CPDAG comparison rows recomputed from the same recovered
  synthetic seeds and patched by exact key;
- SimGlucose interventional CI computed on the complete finite-variable view
  that excludes the undefined `CR` and `CF` columns.

`validation_summary.json` records `94/94` valid cells. Two
`csuite_mixed_simpson` cells each contain one repetition for which the real
data have zero empirical reference independencies. The preservation fraction
is therefore mathematically undefined for all five conditions in that
repetition (`10` rows total); these rows are retained and explicitly marked in
`validation.csv`, while paired tests use the remaining 99 finite pairs.

The denominator is stored as `n_reference_independent`; the actual number of
enumerated CI tests is stored as `n_triples_total`. The misleading historical
duplicate field `n_triples_tested` has been removed. The deterministic builder
also validates and removes that legacy alias when reading older source files.

No seed was selected or replaced based on metrics, direction, significance, or
whether it preserved a previous claim. Heavy synthetic and recovery artifacts
remain outside Git. `build_final_bundle.py` documents the deterministic source
precedence, exact-key patching, path sanitization, and validation logic.
