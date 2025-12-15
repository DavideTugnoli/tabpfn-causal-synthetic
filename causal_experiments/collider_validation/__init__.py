"""Collider validation experiment.

This experiment validates the implementation of collider bias in structural causal models
for use in causal inference experiments with TabPFN.

The experiment uses the SCM: X4 → X3 → X2 ← X1, where X2 is a collider.
"""

from .measure_correlation import conditional_correlation, measure_collider_bias

__all__ = [
    "measure_collider_bias",
    "conditional_correlation",
]