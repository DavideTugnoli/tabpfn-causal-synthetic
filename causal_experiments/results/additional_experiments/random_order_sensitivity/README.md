# Random Order Sensitivity

This experiment tests whether results depend on the arbitrary column order used by the `vanilla_original` baseline.

It is a sensitivity analysis, not a replacement for the main paired comparison protocol. For each dataset and train size, it fixes one or more existing cached train NPZ splits and varies only the column order used by vanilla autoregressive generation. This isolates column-order variability from row-sampling variability.

Default exploratory protocol:

- reuse existing `datasets/train_ts{N}_s{seed}.npz` and `datasets/global_test_set.npz`;
- run one `original_reference` generation on the same split;
- sample up to 10 unique random column permutations without excluding topological or reverse-topological orders;
- if the number of possible permutations is lower than the requested count, enumerate all permutations;
- compute the official comparison metrics: `correlation_matrix_difference`, `k_marginal_tvd`, `nnaa`;
- save synthetic NPZ files and copy the exact input NPZ splits used by the run for later diagnostics.

Paper-aligned paired protocol:

- select the 100 valid `vanilla_original` seeds per train size from the cleaned reps CSV;
- sample one fixed pool of up to 10 random permutations per dataset using `order_seed`;
- assign one random ordering to each selected split by cycling through that fixed pool;
- run one random-order vanilla generation per selected split;
- compare the resulting random-order vanilla CSV against existing cleaned `vanilla_original` and causal-aware rows using the paper's paired protocol.

This costs roughly one additional vanilla run per dataset, not 10 times the full paper experiment.

Primary script:

```bash
python scripts/run_random_order_sensitivity.py \
  --dataset-root /path/to/comparison/results/csuite_mixed_confounding \
  --dataset-name csuite_mixed_confounding \
  --dataset-kind csuite \
  --train-sizes 20,50,100,200,500 \
  --design grid \
  --n-row-repetitions 1 \
  --n-orderings 10 \
  --output-root runs/csuite_mixed_confounding
```

The output files are:

- `random_order_sensitivity_results.csv`: one row per train size, row seed, and ordering;
- `random_order_sensitivity_summary.csv`: metric-level comparison of random-order average/median against the same-split `original_reference`;
- `orderings_manifest.json`: sampled permutations and run metadata.
- `input_datasets/`: copied train/test NPZ files and the cleaned reference CSV used to select seeds, when `--copy-input-datasets` is enabled;
- `synthetic_data/`: generated synthetic NPZ files, when `--save-synthetic` is enabled.

## Cluster execution

The generation runs were executed on HPC clusters through SLURM. The cluster
launchers are site-specific and are not included. All tables and statistics
are reproduced from the saved CSVs by the scripts in `scripts/` and
`final_100_paired_20260608/`.
