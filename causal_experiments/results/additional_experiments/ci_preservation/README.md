# Conditional-Independence Preservation

This analysis tests whether conditional independencies visible in real data are
preserved in saved synthetic datasets. It does not regenerate synthetic data.

Primary protocol:

- Use the empirical reference strategy: test each selected `(X, Y | Z)` triple
  on the real NPZ split and on the corresponding synthetic NPZ.
- Mark a real independence as preserved when both Holm-corrected CI tests fail
  to reject independence at `alpha = 0.05`.
- Use the same CI test choices as the paper's PC discovery pipeline:
  Fisher-Z for the Custom SCM, KCI for continuous nonlinear CSuite and
  SimGlucose, G^2 for categorical CSuite, and the existing hybrid
  KCI/G^2/kNN-CMI tester for `csuite_mixed_confounding`.
- By default, use the train split as empirical reference and downsample the
  synthetic dataset to the same sample size. This keeps the CI test sample size
  aligned with the PC discovery protocol.

Estimate the cost without running CI tests:

```bash
python causal_experiments/results/additional_experiments/ci_preservation/conditional_independence_preservation.py \
  --data-root /path/to/data \
  --output-dir causal_experiments/results/additional_experiments/ci_preservation/runs/estimate \
  --sources original \
  --sample-sizes 20,50,100,200,500 \
  --max-repetitions 100 \
  --max-triples 200 \
  --exclude-discovered-cpdag \
  --estimate-only
```

For full execution on Leonardo, prefer a compute job rather than the login node:

```bash
sbatch causal_experiments/results/additional_experiments/ci_preservation/run_ci_preservation.slurm
```

Set `EXCLUDE_DISCOVERED_CPDAG=0` only after the discovered-CPDAG synthetic data
have been regenerated with the fixed parser.
