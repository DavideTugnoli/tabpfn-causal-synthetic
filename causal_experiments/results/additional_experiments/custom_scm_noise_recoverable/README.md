# Custom SCM noise-recoverable experiment

This folder contains the self-contained post-processing bundle for the Custom
SCM variant used to test the claim that CPDAG-discovered generation improves
once PC can recover the relevant conditional independences.

Scientific setup:

- The legacy Custom SCM DAG and coefficients are unchanged.
- Only the SCM noise level is varied.
- The main run uses `noise_level=0.2` and all six comparison strategies.
- The robustness runs use `noise_level=0.1` and `noise_level=0.5`, with only
  `vanilla_original` and `cpdag_discovered`.
- All cleaned outputs use train sizes `20, 50, 100, 200, 500` and the first
  100 common valid seeds per train size.

Folder layout:

- `data/`: final cleaned CSVs plus `raw/` Demetra CSVs and cleaning audit files.
- `statistics/`: Wilcoxon/Friedman summaries generated from the cleaned CSVs.
- `figures/noise0p2/`: standard forest plots for the main `noise_level=0.2`
  run.
- `figures/robustness_summary/`: compact robustness figure linking CMD
  improvement to exact CPDAG recovery.
- `scripts/`: local post-processing scripts for cleaning and robustness plots.

Regeneration commands, from `causal_experiments/results/comparison_experiment`:

```bash
python ../additional_experiments/custom_scm_noise_recoverable/scripts/clean_custom_scm_noise_recoverable.py
python forest_plots.py \
  --result-files ../additional_experiments/custom_scm_noise_recoverable/data/result_custom_scm_noise0p2_comparison_experiment_cleaned_reps_100.csv \
  --paper-root ../additional_experiments/custom_scm_noise_recoverable/figures/noise0p2 \
  --comparison-results-root ../additional_experiments/custom_scm_noise_recoverable/statistics
python forest_plots.py \
  --result-files \
    ../additional_experiments/custom_scm_noise_recoverable/data/result_custom_scm_noise0p1_robustness_comparison_experiment_cleaned_reps_100.csv \
    ../additional_experiments/custom_scm_noise_recoverable/data/result_custom_scm_noise0p5_robustness_comparison_experiment_cleaned_reps_100.csv \
  --paper-root /tmp/custom_scm_noise_recoverable_discarded_forest_plots \
  --comparison-results-root ../additional_experiments/custom_scm_noise_recoverable/statistics
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib-tabpfn \
  python ../additional_experiments/custom_scm_noise_recoverable/scripts/plot_custom_scm_noise_recoverability_summary.py
```

The second `forest_plots.py` call for robustness is used only to regenerate the
statistical summaries required by the compact robustness figure. The forest
plots from that call should not be used, because the robustness runs do not
include `cpdag_minimal`.
