# REX (DAG discovered) — Final Camera-Ready Bundle

DAG-aware generation with REX-discovered graphs compared against vanilla
TabPFN with original ordering, on the four ANM-compatible CSuite datasets
(CLB, CNS, CSS, CWA), with exactly 100 paired seeds per dataset and training
size. Thirty-six REX repetitions that failed in the historical campaign were
recovered on alternative hardware with the preserved historical code and
inputs: eight on CSS (seeds N=50: 170, 209; N=100: 307; N=200: 389, 411;
N=500: 530, 590, 597) and twenty-eight on CLB/CNS/CWA (see
`sources/rex_recovered_remaining/`). All recovered rows carry
campaign-consistent repetition numbers.

## Layout

- `data/` — the four paired cleaned CSVs used by the plots (full cleaned
  condition set plus REX rows; columns `nnaa`, `nnaa_raw`,
  `nnaa_distance_0p5`). `data/nnaa_distance_inputs/` holds the derived
  copies where `nnaa` carries the distance representation.
- `forest_plots/` — `cmd_kmtvd/` (CMD + kMTVD, paper-style combined PDF),
  `nnaa_raw/`, `nnaa_distance_0p5/`.
- `comparison_results/` — Wilcoxon/Holm statistics per representation.
- `reports/` — `coverage.csv` (paired seeds per cell),
  `published_vs_fixed_comparison.csv` (cell-by-cell vs the published
  figures), `source_manifest.csv` (sha256 of every source).
- `sources/` — read-only inputs: `leonardo_raw_rex/` (historical REX raw
  export), `leonardo_historical_cleaned/` (CSS vanilla source, same campaign
  as REX), `leonardo_historical_raw_baselines/` (WARNING: legacy
  `frobenius_corr_norm` CMD definition; never use as a CMD source),
  `rex_recovered_symprod/` and `rex_recovered_remaining/` (recovered rows).
- `sources/rex_discovery_code/` — the code that produced the REX-discovered DAGs
  (ReX driver `run_rex_npz.py`, boot20 task prep, slurm wrappers, and the
  causalexplain v0.9.1 dtype patch). See the header of `run_rex_npz.py`.
- `scripts/build_rex_results.py` — rebuilds everything above.
- `scripts/csuite_comparison_experiment_rex.py` — the experiment runner that
  loads the REX DAGs and generates the synthetic data (DAG-aware path).

## Vanilla sources per dataset

- CLB, CNS, CWA: official paper cleaned CSVs (`../../comparison_experiment/data`);
  the 100-seed vanilla cohorts are fully paired with REX after the
  missing-seed recovery.
- CSS: historical cleaned (same campaign as REX) plus recovered REX rows.

All 20 dataset/training-size cells have exactly 100 paired seeds
(`reports/coverage.csv`).

## Differences vs the published figures

Completing the missing REX seeds changes no significance decision on
CLB/CNS/CWA (effects shift by at most 0.022 in CMD, 0.002 in kMTVD, 0.005 in
NNAA). Only CSS changes state; all three flips are at N=500: CMD NS ->
significant improvement (+0.031, REX better; REX direction precision is 1.00
there), kMTVD and NNAA NS -> significant degradation. Counts: CMD 15
degradations + 1 improvement, kMTVD 20/20, NNAA 20/20. Raw vs |NNAA - 0.5|
changes no significance.

## Graph storage note

`graph_structure` in REX rows is expressed in the coordinates of the
generation-time (topologically reordered) frame; decode it through
`actual_column_order`. CPDAG rows (no reorder) use dataset coordinates.
Both follow the same rule: the logged graph is the one actually consumed by
the generator.

## Rebuild

```bash
python3 scripts/build_rex_results.py
```
