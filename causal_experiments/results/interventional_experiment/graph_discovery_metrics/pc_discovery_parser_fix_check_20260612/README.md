# Interventional CPDAG discovery metrics after the parser fix

## Scope

This folder recalculates `tab:pc_discovery_metrics_ate`, which measures
PC-stable discovery quality against the per-run mutilated DAG used by the ATE
experiments. No Overleaf file or historical result was modified.

The standard cleaned CSVs under:

`tabpfn-causal-synthetic/causal_experiments/results/interventional_experiment/data/`

are byte-identical to the non-SimGlucose files in:

`tabpfn-causal-synthetic/causal_experiments/results/interventional_experiment/cpdag_parser_fix_20260428/definitive/data/`

Every one of the 42 published `(dataset, train_size)` cells contains exactly
100 valid `cpdag_discovered` rows.

## Calculation

The unchanged original repository script was executed:

`tabpfn-causal-synthetic/causal_experiments/results/interventional_experiment/graph_discovery_metrics/scripts/compute_discovery_metrics.py`

For each discovered-CPDAG row, it matches the corresponding `dag/topological`
row by train size, seed, and repetition. That row contains the per-run
interventional DAG, already mutilated by removing incoming edges into the
treatment node. The script remaps both graphs to the same column order,
computes the per-run metrics, and averages over the 100 seeds.

Run `bash rebuild_all.sh` from this folder to rebuild the original-script
outputs and report.

## Result

The parser-fix recalculation is exactly identical to the historical table:

- 42/42 cells have 100 runs;
- zero metric cells change at full floating-point precision;
- zero metric cells change at the two-decimal paper precision.

Therefore `tab:pc_discovery_metrics_ate` and its accompanying numeric claims do
not need to change.

## Files

- `outputs/discovery_metrics_per_run.csv`: per-run metrics from the original script.
- `outputs/discovery_metrics_summary.csv`: aggregate metrics from the original script.
- `pc_discovery_metrics_parser_fix.csv`: complete recalculated paper table.
- `pc_discovery_metrics_parser_fix.md`: Markdown table.
- `pc_discovery_metrics_parser_fix_rows.tex`: LaTeX replacement rows.
- `published_vs_parser_fix.csv`: historical-versus-current comparison.
- `changed_cells_exact.csv`: empty; no exact changes.
- `changed_cells_at_paper_precision.csv`: empty; no visible changes.
- `build_report.py`: builds the table and comparison artifacts.
- `rebuild_all.sh`: one-command local rebuild.
