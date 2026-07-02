# Custom SCM noise-recoverable comparison results

This folder contains the cleaned CSVs for the Custom SCM variant used to test
whether CPDAG-discovered generation improves when PC can recover the relevant
conditional independences.

Final cleaned CSVs:

- `result_custom_scm_noise0p2_comparison_experiment_cleaned_reps_100.csv`: main run, all six strategies, train sizes `20, 50, 100, 200, 500`.
- `result_custom_scm_noise0p1_robustness_comparison_experiment_cleaned_reps_100.csv`: robustness run, `vanilla_original` and `cpdag_discovered`, train sizes `20, 50, 100, 200, 500`.
- `result_custom_scm_noise0p5_robustness_comparison_experiment_cleaned_reps_100.csv`: robustness run, `vanilla_original` and `cpdag_discovered`, train sizes `20, 50, 100, 200, 500`.

The `raw/` subfolder stores the copied Demetra CSVs and the cleaning audit
files. The cleaned files keep the first 100 seeds that are valid and common
across all strategies in each run and train size.
