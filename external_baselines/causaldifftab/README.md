# CausalDiffTab external baseline

This directory contains a standalone external baseline for CausalDiffTab.
It does not modify the TabPFN codebase, the historical TabPFN environment, or
the TabularARGN environment.

## Scope

The baseline trains the public CausalDiffTab implementation on the same cached
train NPZ splits used by the comparison experiments, samples an unconditional
synthetic dataset with the same number of rows as the global test set, and
evaluates the same paper metrics:

- correlation matrix difference
- k-marginal TVD
- NNAA

The wrapper adapts each NPZ split to the CausalDiffTab file layout, runs the
official train/test entry points with the repository default hyperparameters,
and then stores the synthetic table as NPZ plus a CSV row with the metrics.

## Upstream

The upstream repository is cloned under `upstream/`:

```text
https://github.com/Jia-Chen-Zhang/CausalDiffTab
```

Pinned local clone at setup time:

```text
b81f44236cd7513abb98845ff71d519545ad4c06
```

## Leonardo

The Leonardo environment must be created under this baseline directory and must
not touch existing environments.

Default training is intentionally expensive: the upstream config uses 8000
training steps. Smoke tests should use `--debug`; full runs should not.

