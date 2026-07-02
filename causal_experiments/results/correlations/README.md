# Spurious correlations (Custom SCM)

- `parser_fix_20260612/` — CANONICAL camera-ready results. Table A1 values
  recomputed from the canonical synthetic NPZ archive after the
  discovered-CPDAG parser fix (only `cpdag_discovered` rows change; all other
  methods reproduce the historical summary to 1e-8).
- `spurious_correlations_summary.csv` — historical pre-parser-fix summary,
  kept as the comparison baseline for
  `parser_fix_20260612/build_comparison_report.py`. Do not use its
  `cpdag_discovered` rows.
- `noise1e-2/` — noise-variant summary.
- `parser_fix_noise1e-2_check_20260612/` — verification (2026-06-12, Leonardo
  job 46026864): the noise table was recomputed from the canonical noise NPZ
  archive; all 30 cells match the published `tab:spurious_correlations_noise`
  at paper precision (max delta 1.7e-8, numeric noise). No table change
  needed.
