# ATE Outcome-Scale Reference

Purpose: provide an outcome-scale reference for ATE preservation analyses where forest plots show paired Hodges-Lehmann reductions in absolute ATE error without an explicit denominator.

This folder contains a deterministic post-processing analysis only. It does not regenerate synthetic data and does not run TabPFN.

## Inputs

Raw ATE CSVs are read from:

`causal_experiments/results/interventional_experiment/data/`

The required columns are:

- `ate_test`
- `ate_synthetic`
- `ate_difference`
- `ate_relative_error`
- `algorithm`
- `column_order`
- `train_size`
- `seed`

## Outputs

Generated tables go in `tables/`.

Generated SimGlucose diagnostic plots go in `figures/`.

The main table is:

`tables/primary_scale_reference_main_all_train_sizes.md`

The same content is also written as CSV and LaTeX. It includes all available train sizes for the main non-noise datasets. Noise-robustness rows are retained in the full CSV output.

Full-reference tables are in `tables/full_reference/`:

- `ate_scale_reference_compact.{csv,md,tex}`: 3 representative datasets (`CSM`, `CMC`, `SGL`) at `N = 20, 100, 500`.
- `ate_scale_reference_full.{csv,md,tex}`: all 8 main datasets and all 6 train sizes.

These tables remove confidence intervals to keep the scale reference compact:
they report `Abs. ATE_test`, the median absolute ATE error for vanilla, the
primary comparator and its median absolute ATE error. Inference is reported with
the paired Hodges-Lehmann effect used in the forest plots,
`HL diff (vanilla - comparator)`, and a `Significant` flag from the paired
Wilcoxon signed-rank test with Pratt ties and Holm correction.

## Comparators

The baseline is `vanilla_original`.

Primary comparator:

- Custom SCM and CSuite datasets: `dag_topological`.
- SimGlucose: `vanilla_topological`, because the saved interventional SimGlucose run only contains vanilla original/topological settings, reflecting partial causal-order information rather than full DAG-aware conditioning.

## Units

- Custom SCM and CSuite ATE values are in the native outcome units of each SCM/dataset.
- SimGlucose ATE values are in `mg/dL`: the target variable is `subcut_glucose`, computed as the CGM-like subcutaneous glucose observation. The intervention variable is `action_insulin_U_per_min`, measured in U/min.

## Reproduction

From the repository root:

```bash
python3 causal_experiments/results/interventional_experiment/ate_outcome_scale/scripts/build_ate_scale_reference.py
```

The reference table can be regenerated with:

```bash
python3 causal_experiments/results/interventional_experiment/ate_outcome_scale/scripts/build_ate_scale_reference_table.py
```

On Demetra, the SLURM wrapper is:

The cluster launcher is site-specific and is not included. The analysis is a
lightweight deterministic post-processing of the raw ATE CSVs and can be run
directly with the Python scripts in `scripts/`.
