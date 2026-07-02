# Causal-TGAN Default Configuration

Upstream repository:
`https://github.com/BiggyBing/Causal-TGAN-Public`

Pinned commit:
`5d5b35bc3ae96e9ce6b7a4fc7cd90ca7679ff859`

The adapter uses the official defaults exposed by `upstream/train.py`:

- batch size: `500`
- epochs: `400`
- PAC size: `1`
- exogenous noise dimension: `2`
- discriminator iterations: `3`
- transformer: `ctgan`

The upstream model code fixes both generator and discriminator learning rates
to `2e-4`, Adam betas to `(0.5, 0.9)`, and weight decay to `1e-6`.

The causal graph is not a tunable hyperparameter. Causal-TGAN requires a causal
graph, so the adapter supplies the known `custom_scm` DAG:

- `X0`: no parents
- `X1`: parents `X0`, `X2`
- `X2`: parent `X3`
- `X3`: no parents

Important limitation: with the official `batch_size=500` and `D_iter=3`, the
upstream training loop may not update the generator when a training set yields
only one batch per epoch. This is intentionally not changed silently. The
smoke-test output must be inspected before a full launch.

The upstream repository targets Python 3.6, PyTorch 1.9, and a legacy RDT class
name. Leonardo uses a modern Python/PyTorch/RDT environment. The adapter adds a
runtime alias from the removed `OneHotEncodingTransformer` name to the current
`OneHotEncoder` name without modifying upstream code. `custom_scm` is fully
continuous, so the categorical transformer is not used by this experiment.
