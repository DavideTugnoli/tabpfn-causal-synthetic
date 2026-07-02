# Causal-TGAN External Baseline

This directory isolates the official Causal-TGAN implementation and evaluates
it under the shared paired-seed external-baseline protocol.

Scope:

- dataset: `custom_scm`
- train sizes: `20,50,100,200,500`
- seeds: exactly the 100 valid `vanilla/original` seeds in the canonical
  cleaned comparison CSV for each train size
- outputs: one synthetic NPZ, one metrics row, and an upstream checkpoint per
  completed seed
- metrics: the paper CMD, k-marginal TVD, and raw NNAA implementations

See `DEFAULTS.md` for the pinned upstream commit, graph, defaults, and known
small-sample limitation.

