# CTGAN Default Configuration

Upstream repository:
`https://github.com/sdv-dev/CTGAN`

Pinned commit:
`f4fcd21d96e291fb1d6b7a14b83236927560b81e`

The comparison adapter constructs `CTGAN()` without model hyperparameter
overrides. Therefore it uses the official constructor defaults from
`upstream/ctgan/synthesizers/ctgan.py`:

- embedding dimension: `128`
- generator dimensions: `(256, 256)`
- discriminator dimensions: `(256, 256)`
- generator learning rate: `2e-4`
- generator weight decay: `1e-6`
- discriminator learning rate: `2e-4`
- discriminator weight decay: `1e-6`
- batch size: `500`
- discriminator steps: `1`
- log-frequency sampling: enabled
- epochs: `300`
- PAC size: `10`
- GPU: enabled when available

The adapter only sets the random state and supplies the dataset. `custom_scm`
contains only continuous columns, so the official `fit` call receives an empty
discrete-column list.

