"""Causal experiments for TabPFN Extensions.

This package contains utilities and experiments for testing causal inference
capabilities of TabPFN when provided with causal graph constraints.

Available experiments:
- collider_validation: Validates collider bias in structural causal models
"""

from . import utils
from .utils.scm_data import (
    generate_mixed_scm_data,
    generate_numeric_scm_data,
    get_mixed_cpdag_and_config,
    get_mixed_dag_and_config,
    get_numeric_cpdag_and_config,
    get_numeric_dag_and_config,
)

__all__ = [
    "utils",
    "generate_numeric_scm_data",
    "generate_mixed_scm_data",
    "get_numeric_dag_and_config",
    "get_mixed_dag_and_config",
    "get_numeric_cpdag_and_config",
    "get_mixed_cpdag_and_config",
]