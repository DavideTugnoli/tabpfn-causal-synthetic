# NNAA Distance-to-0.5 Forest Plots

Date: 2026-06-03 (combined kMTVD+NNAA figures added 2026-06-11)

This directory is an isolated robustness visualization. The official cleaned
CSV files are not modified. Each input CSV here is a copy of a cleaned
reps-100 CSV where `nnaa` has been replaced by `abs(nnaa - 0.5)` and the
original value is retained as `nnaa_raw_original`.

The forest plots are produced by invoking the original comparison
`forest_plots.py` script with `--metrics k_marginal_tvd nnaa`,
`--paper-root`, and `--comparison-results-root` pointing inside this
analysis directory. Requesting both metrics also produces the combined
two-panel `forest_combined_*_2marginal_nnaa_distance0p5.pdf` figures used
in the paper appendix; their kMTVD panels show the raw, unchanged kMTVD
values.

NNAA panels are titled '|Nearest-Neighbor Adversarial Accuracy - 0.5|'
via the forest-plot `--nnaa-title` option; the produced PDFs are then
renamed with a `_distance0p5` suffix so a transformed figure is never
mistaken for a raw-NNAA plot. Interpret every NNAA plot in this
directory as distance from the ideal raw NNAA value 0.5, where lower is
better.

## Bundles

- `main_cleaned`: original main cleaned comparison datasets.
- `noise1e-2_cleaned`: same bundle but replacing `custom_scm` with
  `custom_scm_noise1e-2`.

## Raw-vs-distance summary

- Compared raw-NNAA vs |NNAA - 0.5| Wilcoxon summaries across 430 posthoc contrasts.
- Holm significance changes: 0.
- Max absolute HL delta (distance - raw, un-oriented posthoc scale): 0.001125.
