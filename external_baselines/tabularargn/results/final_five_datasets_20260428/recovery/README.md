# CMC N=20 CMD Metric Recovery

This recovery recomputes only CMD for the 36 exact paired seeds whose
historical TabularARGN synthetic datasets contain a constant column and
therefore produced undefined Spearman correlations.

- No model is retrained.
- No synthetic dataset is regenerated or replaced.
- Historical CSVs and synthetic NPZs remain untouched.
- The deterministic convention maps undefined off-diagonal Spearman
  correlations to zero before computing the Frobenius norm.
- Output is written to a separate Leonardo patch root and must be validated
  before the public paired analysis is regenerated.

Status: complete. Leonardo jobs `45760195` and `45761078` recomputed the metric
patch and regenerated the paired outputs. The final public analysis has 100
finite pairs in every cell; `regeneration_validation.json` records the final
gate.

Historical source:
`<work_root>/runs/tabularargn_baseline_train_size_split/csuite_mixed_confounding`

Patch root:
`<work_root>/runs/tabularargn_cmd_patch_20260611`
