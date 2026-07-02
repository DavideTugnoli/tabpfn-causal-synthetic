# External Baselines — Paper-Facing Material

This folder holds the tables for the external-generator appendix section
(camera-ready). The full result bundles, runner scripts, and per-baseline
documentation live in the repository's top-level `external_baselines/` folder.

## Contents

- `scripts/build_custom_scm_baseline_table.py` — builds the Custom SCM
  appendix table: per-cell medians of CMD, kMTVD, and |NNAA - 0.5| for the
  five external generators plus the two TabPFN references, and paired
  Wilcoxon-Pratt + Holm tests against both references (Holm family =
  baseline x comparator x metric across the five training sizes).
- `tables/custom_scm_baseline_medians.csv` — the appendix table source.
- `tables/custom_scm_baseline_wilcoxon_holm.csv` — significance audit:
  150/150 contrasts have positive paired median difference (TabPFN better);
  75/75 significant vs DAG-aware, 68/75 vs vanilla (the 7 non-significant
  cells are CMD for DATGAN at N<=200 and CausalDiffTab at 50<=N<=200).
- The TabularARGN five-dataset comparison (25/25 cells, all Holm-significant)
  is reported directly from
  `external_baselines/tabularargn/results/final_five_datasets_20260428/analysis/`.

## Status per baseline

| Baseline | Result bundle | Status |
|---|---|---|
| TabularARGN | `external_baselines/tabularargn/results/final_five_datasets_20260428/` | Final: 5 datasets x 5 train sizes, 100 paired seeds per cell; TabPFN (vanilla and DAG-aware) wins 25/25 cells on CMD, kMTVD, NNAA, all Holm-significant |
| CausalDiffTab | `external_baselines/causaldifftab/results/final_custom_scm_20260604/` | Final on Custom SCM; included in the appendix table |
| CTGAN | `external_baselines/ctgan/results/final_custom_scm_20260606/` | Final on Custom SCM; included in the appendix table |
| DATGAN | `external_baselines/datgan/results/final_custom_scm_20260606/` | Final on Custom SCM; included in the appendix table |
| DECAF | `external_baselines/decaf/results/final_custom_scm_20260606/` | Final on Custom SCM; included in the appendix table |
| Causal-TGAN | none | Not run: under official defaults, every project training size (N <= 500) yields zero generator updates (see `external_baselines/causaltgan/DEFAULTS.md`) |

The tables are generated with the same statistical conventions as the main
experiments (paired Wilcoxon-Pratt with Holm correction).
