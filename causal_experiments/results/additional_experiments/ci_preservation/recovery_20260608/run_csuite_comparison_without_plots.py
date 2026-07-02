#!/usr/bin/env python3
"""Run the standard CSuite comparison CLI while disabling diagnostic plots.

This recovery-only entry point preserves the generation and metric pipeline.
It bypasses PairGrid plotting because duplicate column labels can make plotting
fail after valid synthetic data has already been generated.
"""

from __future__ import annotations

import runpy
from pathlib import Path

from tabpfn_extensions.unsupervised.experiments import GenerateSyntheticDataExperiment


def _skip_plot(self: GenerateSyntheticDataExperiment, **kwargs: object) -> None:
    return None


GenerateSyntheticDataExperiment.plot = _skip_plot

PROJECT_ROOT = Path(__file__).resolve().parents[4]
TARGET = (
    PROJECT_ROOT
    / "causal_experiments/csuite_experiment/comparison_experiment_csuite"
    / "csuite_comparison_experiment.py"
)
runpy.run_path(str(TARGET), run_name="__main__")
