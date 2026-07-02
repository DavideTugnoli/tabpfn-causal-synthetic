# CPDAG discovery metrics after the parser fix

## Scope

This folder recalculates the comparison-experiment table
`tab:pc_discovery_metrics` from the local definitive parser-fix cleaned CSVs.
The Overleaf checkout was not modified.

The standard local cleaned files under:

`tabpfn-causal-synthetic/causal_experiments/results/comparison_experiment/data/`

are byte-identical to:

`tabpfn-causal-synthetic/causal_experiments/results/comparison_experiment/cpdag_parser_fix_20260428/definitive/data/`

for the eight non-SimGlucose datasets. Every `cpdag_discovered` cell contains
exactly 100 unique seeds. The paper table includes CSM and the six CSuite
datasets; `custom_scm_noise1e-2` is not included.

## Original calculation

The unchanged original script was executed directly:

```bash
python3 tabpfn-causal-synthetic/causal_experiments/results/comparison_experiment/graph_discovery_metrics/scripts/compute_discovery_metrics.py \
  --repo-root tabpfn-causal-synthetic \
  --output-dir causal_experiments/results/comparison_experiment/graph_discovery_metrics/pc_discovery_parser_fix_20260612/outputs
```

For a one-command local rebuild of both the original-script outputs and the
camera-ready comparison report, run:

```bash
bash causal_experiments/results/comparison_experiment/graph_discovery_metrics/pc_discovery_parser_fix_20260612/rebuild_all.sh
```

For each cleaned `cpdag_discovered` row, the script:

1. loads the ground-truth DAG;
2. parses directed and undirected edges from `graph_structure`;
3. computes per-run metrics;
4. averages them over the 100 seeds for each `(dataset, train_size)` cell.

Definitions:

- skeleton recall: correctly recovered skeleton edges / true skeleton edges;
- direction recall: correctly oriented directed edges / true directed edges;
- oriented fraction: mean directed-edge count / mean discovered-skeleton-edge count;
- direction precision: correctly oriented directed edges / directed edges,
  averaged only over runs with at least one directed edge. A dash means no run
  in the cell contains directed edges.

## Main result

At the two-decimal precision used in the paper, only one table entry changes:

- CWA, N=50, direction precision: `0.36 -> 0.34`.

Four cells change at full precision:

| Dataset | N | Metric | Published | Parser-fix | Delta |
|---|---:|---|---:|---:|---:|
| CMC | 50 | Direction recall | 0.066000 | 0.065333 | -0.000667 |
| CMC | 50 | Direction precision | 0.632240 | 0.626776 | -0.005464 |
| CMC | 200 | Direction recall | 0.089333 | 0.090000 | +0.000667 |
| CMC | 200 | Direction precision | 0.246114 | 0.247832 | +0.001718 |
| CMC | 500 | Direction recall | 0.114667 | 0.114000 | -0.000667 |
| CMC | 500 | Direction precision | 0.191126 | 0.189698 | -0.001429 |
| CWA | 50 | Direction recall | 0.013333 | 0.012667 | -0.000667 |
| CWA | 50 | Direction precision | 0.362179 | 0.342949 | -0.019231 |

Skeleton recall and oriented fraction do not change in any cell.

The parser bug affected the conversion of causal-learn's CPDAG matrix into the
parent mapping used for synthetic generation. It did not change PC-stable
itself. Therefore, graph-discovery metrics remain almost entirely unchanged.
The small differences above come from the parser-fix recovery rows generated
on Demetra for missing seeds, whose discovered graphs differ slightly from the
historical rows.

The qualitative statements accompanying the table remain valid:

- maximum direction recall is still `0.32`;
- CMC at N=500 still rounds to oriented fraction `0.86` and direction
  precision `0.19`.

## Outputs

- `outputs/discovery_metrics_per_run.csv`: original-script per-run output.
- `outputs/discovery_metrics_summary.csv`: original-script aggregate output.
- `pc_discovery_metrics_parser_fix.csv`: new paper-table values with full precision.
- `pc_discovery_metrics_parser_fix.md`: complete new table at paper precision.
- `pc_discovery_metrics_parser_fix_rows.tex`: replacement LaTeX rows.
- `published_vs_parser_fix.csv`: complete old-versus-new comparison.
- `changed_cells_exact.csv`: cells changed at full precision.
- `changed_cells_at_paper_precision.csv`: cells changed after two-decimal rounding.
- `build_report.py`: rebuilds the table and comparison from the original-script output.
- `rebuild_all.sh`: runs the unchanged repository calculation script and then
  rebuilds this folder's table/report artifacts.
