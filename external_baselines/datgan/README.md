# DATGAN external baseline

This directory integrates the official DATGAN repository with the shared
paired-cleaned-seed protocol. DATGAN requires TensorFlow 2.8.0, so it runs in a
dedicated Python 3.9 environment managed by `uv` and never modifies the TabPFN
environment.

See `DEFAULTS.md` for the necessary low-sample batch-size adaptation.
