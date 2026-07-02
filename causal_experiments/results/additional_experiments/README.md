# Additional Experiments

Additional experiments reported in the appendix of the UAI 2026 paper. The
main-paper experiments live in `../comparison_experiment`
and `../interventional_experiment`.

- `rex/` — causal discovery with REX: DAG-aware generation with REX-discovered
  graphs vs vanilla TabPFN (4 CSuite datasets, 100 paired seeds per cell).
- `custom_scm_noise_recoverable/` — Custom SCM variant with `noise=0.2`, where
  PC reliably recovers the structure (recoverable-discovery regime).
- `random_order_sensitivity/` — random column permutations as an alternative
  vanilla baseline (100 paired seeds per cell).
- `ci_preservation/` — empirical conditional-independence preservation on
  synthetic data (comparison and interventional branches).
- `external_baselines/` — tables for the external-generator appendix section
  (result bundles live in the repository's `external_baselines/` top-level
  folder).

Each subfolder keeps the standard layout: cleaned CSVs, statistics, plots,
and rebuild scripts.
