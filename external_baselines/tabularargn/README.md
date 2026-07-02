# TabularARGN external baseline

This directory contains a standalone external baseline. It does not
modify the TabPFN codebase or the historical environments.

## Scope

The baseline trains `mostlyai.engine.TabularARGN` on the same cached train NPZ splits used
by the comparison experiments, samples an unconditional synthetic dataset with the same
number of rows as the global test set, and evaluates the same quality metrics:

- correlation matrix difference
- k-marginal TVD
- NNAA

This is intended as an external generator sanity-check baseline, not as a causal variant
of TabularARGN. The public engine supports any-order autoregressive learning and
conditional sampling, but this baseline uses the model as-is.

## Environment

Local setup uses `uv` because `pixi` is not installed on the local machine.

```bash
cd external_baselines/tabularargn
uv sync
```

The Leonardo environment must be created separately in the project WORK area and must not
modify the historical TabPFN venv.

## Minimal smoke command

```bash
cd external_baselines/tabularargn
.venv/bin/python scripts/run_tabularargn_comparison.py \
  --dataset custom_scm \
  --dataset-dir /path/to/custom_scm/comparison/output \
  --tabpfn-repo /path/to/tabpfn-causal-synthetic \
  --train-sizes 20 \
  --seed-start 0 \
  --repetitions 2 \
  --max-training-time 2 \
  --output-dir results/smoke_custom_scm
```

## Datasets

Start with:

- `custom_scm`
- `custom_scm_noise1e-2`
- `csuite_symprod_simpson`
- `csuite_mixed_confounding`
- `csuite_large_backdoor`

Extend to the remaining CSuite datasets only if runtime is acceptable.
