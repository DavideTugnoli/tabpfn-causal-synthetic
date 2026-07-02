# Custom SCM noise-recoverable plots

This folder contains only the plots for the Custom SCM noise-recoverable
experiment.

- `noise0p2/`: standard forest plots for the main run with `noise_level=0.2`.
  These use all six comparison strategies and train sizes `20, 50, 100, 200, 500`.
- `robustness_summary/`: compact robustness plot for `noise_level` values
  `0.1`, `0.2`, and `0.5`.

The robustness summary has two panels:

- `CMD improvement`: paired Hodges-Lehmann effect on CMD,
  `CMD(vanilla_original) - CMD(cpdag_discovered)`. Positive values mean that
  CPDAG-discovered generation improves over vanilla original.
- `PC recovery`: percentage of the 100 cleaned repetitions where the discovered
  CPDAG exactly matches the expected CPDAG for the legacy Custom SCM.

The robustness claim is not that all noise values have the same magnitude of
improvement. It is that the advantage of CPDAG-discovered generation appears
throughout the regime where PC recovers the relevant structure.
