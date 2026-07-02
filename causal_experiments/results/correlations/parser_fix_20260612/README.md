# Custom SCM spurious correlations after the CPDAG parser fix

## Scope

This folder rebuilds Table A1 (`tab:spurious_correlations`) from the canonical
comparison NPZ archive associated with the current cleaned CSVs. No Overleaf
file or historical repository result was modified.

Canonical Leonardo source:

`<work_root>/data/cleaned_npz_archive/comparison_current_cleaned_20260429`

The archive contains exactly the synthetic NPZs selected by the current
cleaned CSVs, including the parser-fixed `cpdag_discovered` rows.

## Calculation

`build_spurious_correlations_table.py`:

1. reads the canonical Custom SCM cleaned CSV;
2. verifies exactly 100 unique seeds for every `(method, train_size)` cell;
3. loads the corresponding 3,000 synthetic NPZs and the canonical test NPZ;
4. computes Pearson correlations for the two marginally independent pairs
   `X0-X3` and `X0-X2`;
5. aggregates mean and sample standard deviation over 100 repetitions;
6. writes CSV, Markdown, and LaTeX table outputs.

The script was executed on the Leonardo CPU partition `dcgp_usr_prod`.
Successful calculation job: `45987230`; final
camera-ready-label regeneration job: `45987739` (both `COMPLETED 0:0`).

The first two attempts (`45986166`, `45986764`) completed the correlation
calculation but failed in the optional graph-audit step because the new report
script initially did not handle `no_graph` and DAG-style graph structures.
The audit code was corrected; neither failed attempt modified data or affected
the final results.

`build_comparison_report.py` compares the rebuilt summary against the
historical repository summary:

`tabpfn-causal-synthetic/causal_experiments/results/correlations/spurious_correlations_summary.csv`

## Result

At the three-decimal precision used in the paper:

- `vanilla_original`, `vanilla_topological`, `vanilla_reverse_topological`,
  `dag_topological`, and `cpdag_minimal_original` are unchanged;
- only `cpdag_discovered_original` changes, at every train size.

The corrected discovered-CPDAG means are generally closer to zero, although
their standard deviations increase. This is consistent with correcting the
orientation convention while leaving the seed selection and all non-discovered
synthetic datasets unchanged.

## Files

- `build_spurious_correlations_table.py`: canonical NPZ-based calculation.
- `run_on_leonardo.slurm`: CPU job wrapper.
- `build_comparison_report.py`: historical-versus-parser-fix comparison.
- `outputs/spurious_correlations_per_run.csv`: all 3,000 per-run correlations.
- `outputs/spurious_correlations_summary.csv`: all 30 aggregated cells.
- `outputs/spurious_correlations_table.md`: camera-ready table view.
- `outputs/spurious_correlations_table_rows.tex`: replacement LaTeX rows.
- `outputs/coverage.csv`: 100-seed coverage audit.
- `outputs/graph_structure_audit.csv`: graph-format audit.
- `published_vs_parser_fix.csv`: complete old-versus-new comparison.
- `changed_rows_at_paper_precision.csv`: rows visibly changed at three decimals.
- `discovered_cpdag_old_vs_new.md`: concise discovered-CPDAG comparison.
