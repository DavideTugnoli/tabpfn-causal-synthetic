# Noise=1e-2 spurious correlations after the CPDAG parser fix

## Scope

This folder rebuilds `tab:spurious_correlations_noise` from the canonical
comparison NPZ archive associated with the current parser-fixed cleaned CSV.
No Overleaf file or historical repository result is modified.

Canonical Leonardo source:

`<work_root>/data/cleaned_npz_archive/comparison_current_cleaned_20260429`

Dataset:

`custom_scm_noise1e-2`

## Calculation

`build_spurious_correlations_noise_table.py`:

1. reads the canonical noise=1e-2 cleaned CSV;
2. requires exactly 100 unique seeds for every method and train-size cell;
3. loads the corresponding synthetic NPZs and canonical test set;
4. computes Pearson correlations for independent pairs `X0-X3` and `X0-X2`;
5. aggregates mean and sample standard deviation over the 100 repetitions;
6. writes CSV, Markdown, and LaTeX table artifacts.

The calculation was submitted to Leonardo's CPU partition as job `46026864`
and completed successfully (`COMPLETED 0:0`, elapsed `00:00:55`).

`build_comparison_report.py` compares the rebuilt table with the historical
summary used for the paper.

## Result

- All 30 method/train-size cells contain exactly 100 unique seeds.
- No row changes at the three-decimal precision used in the paper.
- No row changes above a `1e-7` numerical tolerance.
- The largest full-precision difference is `1.69e-8`, attributable to numeric
  serialization/rounding rather than a scientific change.

The published `Discovered CPDAG` rows therefore remain valid and need no
camera-ready replacement.

## Files

- `build_spurious_correlations_noise_table.py`: canonical NPZ-based calculation.
- `run_on_leonardo.slurm`: Leonardo CPU wrapper.
- `build_comparison_report.py`: historical-versus-parser-fix comparison.
- `outputs/`: generated calculation outputs after synchronization from Leonardo.
- `published_vs_parser_fix.csv`: complete old-versus-new comparison.
- `changed_rows_at_paper_precision.csv`: rows visibly changed at three decimals.
- `changed_rows_above_numeric_tolerance.csv`: empty at tolerance `1e-7`.
- `discovered_cpdag_old_vs_new.md`: concise discovered-row comparison.
