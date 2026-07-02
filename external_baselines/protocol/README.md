# External Baseline Protocol

This directory defines the common experimental protocol for non-TabPFN tabular
generators. It is intentionally separate from the main TabPFN experiment code:
the paper's original TabPFN pipeline and historical outputs should not be
modified when adding external baselines.

## Fixed Scientific Protocol

Every external generator must use the same:

- canonical cleaned train/test NPZ archive;
- paired cleaned seeds from the `vanilla/original` comparison CSV;
- train sizes requested by the experiment;
- number of synthetic samples, equal to the global test-set size;
- output schema;
- paper metrics: CMD, k-marginal TVD, and raw NNAA;
- synthetic NPZ metadata format.

The loader supports both historical experiment roots
(`dataset_dir/datasets/train_ts...`) and the canonical cleaned archive layout
(`dataset_dir/train/train_ts...`). The exact resolved train/test NPZ paths are
stored in every result row and synthetic NPZ metadata.

The model-specific code is only responsible for:

1. fitting on one train split;
2. sampling a synthetic table with the requested number of rows;
3. returning the generated `pandas.DataFrame`.

## Adapter Contract

Implement `ExternalGeneratorAdapter.fit_sample(...)` from `core.py`.

```python
class MyGeneratorAdapter(ExternalGeneratorAdapter):
    name = "my_generator"
    column_order = "unconditional"

    def fit_sample(self, train_df, n_samples, seed, workspace_dir):
        # Call the official implementation here.
        # Keep official/default hyperparameters unless explicitly justified.
        return synthetic_df, {"implementation": "official repo commit ..."}
```

Then call `run_external_baseline_protocol(adapter, config)`.

## Recoverability

The shared protocol writes one CSV row and, when enabled, one synthetic NPZ per
completed seed. Reruns with `resume=True` skip completed valid `(train_size,
seed)` pairs. This protects completed seeds after node failure or walltime
timeout.

This does not magically checkpoint a model's internal training loop. If an
external official implementation cannot resume a partially trained single seed,
then an interrupted seed must be recomputed. Do not hide this limitation in
scientific reporting.

## Current Adapters

The existing `external_baselines/tabularargn` and
`external_baselines/causaldifftab` wrappers predate this shared module but
follow the same protocol: cleaned paired seeds, one CSV row per seed, paper
metrics, synthetic NPZ outputs, and resume by completed seed. Future adapters
should use `external_baselines/protocol/core.py` directly.
