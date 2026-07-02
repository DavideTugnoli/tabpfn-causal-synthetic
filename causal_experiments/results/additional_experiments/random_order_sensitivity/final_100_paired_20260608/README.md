# Random-order sensitivity final 100-paired results

Status: final full-scope aggregation completed.

This directory will contain only lightweight final CSV tables, validation
reports, and provenance. It must not contain synthetic NPZ files or partial
outputs.

Every random-order row must be paired with the canonical cleaned seed used by
each comparator in the same dataset/train-size cell. Final tables are promoted
only after every compared method has exactly the same 100 valid seeds.

No seed may be selected or replaced based on metric values, significance, or
whether it preserves a previous claim. Previously reported counts must be audited after
the final paired dataset is reconstructed.

## Scopes

The final tables deliberately report two scopes:

- `regular_7_dag_aware`: seven datasets with a full DAG-aware comparator;
- `paper_8_comparator`: the same seven plus SimGlucose, where the paper's
  comparator is vanilla topological because a complete DAG is unavailable.

Every cell in both scopes has exactly 100 paired valid seeds.

A six-dataset subset was reported separately at an earlier stage. Its published
counts remain unchanged and must not be replaced retroactively by the
full-scope counts.

## Files

- `tables/random_order_full_scope_paired_long.csv`: final paired lightweight
  rows.
- `tables/random_order_full_scope_pairing_validation.csv`: per-cell seed gate.
- `tables/random_order_full_scope_pairwise_detail.csv`: paired Wilcoxon and
  Hodges-Lehmann results.
- `tables/random_order_full_scope_pairwise_summary.csv`: W/L/NS summaries.
- `tables/random_order_full_scope_metadata.json`: source and scope provenance.
