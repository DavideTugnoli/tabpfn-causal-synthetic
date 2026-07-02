# External Baselines

This directory contains the lightweight experiment adapters, documented
upstream defaults, and final result summaries used to compare external
synthetic-data generators.

Each baseline is isolated from the main TabPFN environment. Upstream source
trees, virtual environments, raw runs, trained models, and synthetic NPZ files
are intentionally excluded from Git. The per-baseline README records the
official upstream repository, pinned revision, defaults, and any scientifically
necessary adaptation.

Only result bundles that pass the same paired-seed validation policy as the
main experiments should be placed under a baseline's `results/final_*`
directory.

The TabularARGN bundle includes a documented metric-only
recovery for one CMD cell. It recomputes the metric from the exact same saved
synthetic datasets and does not retrain models or replace seeds.
