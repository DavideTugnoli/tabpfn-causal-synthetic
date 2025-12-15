"""Simple SCM data generation - separate functions for numeric and mixed (numeric + categorical) data."""
from __future__ import annotations

import numpy as np

# Column names and categorical columns for numeric SCM
NUMERIC_COL_NAMES = ["X0", "X1", "X2", "X3"]
NUMERIC_CATEGORICAL_COLS: list[int] = []

# Column names and categorical columns for mixed SCM
MIXED_COL_NAMES = ["X0", "X1", "X2", "X3", "X4_cat"]
MIXED_CATEGORICAL_COLS = [4]  # X4_cat is categorical (index 4)

def generate_numeric_scm_data(
    n_samples: int,
    random_state: int = 42,
    *,
    noise_level: float = 1e-5,
    noise_x2: float | None = None,
    noise_x3: float | None = None,
) -> np.ndarray:
    """Generate numeric SCM data: X3 → X2 → X1 ← X0
    Returns array with columns [X0, X1, X2, X3] (all float32).
    Optimized for maximum collider bias (0.929): x2_coeff=0.5, x0_coeff=5.0, x2_to_x1_coeff=10.0.
    """
    noise_x2 = noise_level if noise_x2 is None else noise_x2
    noise_x3 = noise_level if noise_x3 is None else noise_x3
    rng = np.random.default_rng(random_state)
    # Generate data: X3 → X2 → X1 ← X0
    X3 = rng.normal(0, 1, n_samples)
    X0 = rng.normal(0, 1, n_samples)
    X2 = 0.5 * X3 + rng.normal(0, noise_x3, n_samples)
    X1 = 5.0 * X0 + 10.0 * X2 + rng.normal(0, noise_x2, n_samples)
    data = np.column_stack([X0, X1, X2, X3]).astype(np.float32)
    return data

def generate_mixed_scm_data(
    n_samples: int,
    random_state: int = 42,
    *,
    noise_level: float = 0.3,
    noise_x2: float | None = None,
    noise_x3: float | None = None,
) -> np.ndarray:
    """Generate mixed SCM data: X3 → X2 → X1 ← X0, with X4_cat categorical depending on X1.
    Returns array with columns [X0, X1, X2, X3, X4_cat] (X4_cat is int).
    """
    noise_x2 = noise_level if noise_x2 is None else noise_x2
    noise_x3 = noise_level if noise_x3 is None else noise_x3
    rng = np.random.default_rng(random_state)
    # Generate data: X3 → X2 → X1 ← X0
    X3 = rng.normal(0, 1, n_samples)
    X0 = rng.normal(0, 1, n_samples)
    X2 = 2.0 * X3 + rng.normal(0, noise_x3, n_samples)
    X1 = 1.5 * X0 + 1.5 * X2 + rng.normal(0, noise_x2, n_samples)
    data = np.column_stack([X0, X1, X2, X3]).astype(np.float32)
    # X4_cat depends on X1 (vectorized)
    X1_norm = (X1 - X1.min()) / (X1.max() - X1.min())
    probs = np.zeros((n_samples, 3))
    probs[X1_norm < 0.33] = [0.7, 0.2, 0.1]
    probs[(X1_norm >= 0.33) & (X1_norm < 0.67)] = [0.2, 0.6, 0.2]
    probs[X1_norm >= 0.67] = [0.1, 0.2, 0.7]
    X4_cat = np.array([rng.choice(3, p=probs[i]) for i in range(n_samples)], dtype=int)
    data = np.column_stack([data, X4_cat])
    return data

def get_numeric_dag_and_config() -> tuple[dict[int, list[int]], list[str], list[int]]:
    """Get DAG and column info for numeric SCM. DAG: X3 → X2 → X1 ← X0."""
    dag = {
        0: [],      # X0 has no parents
        1: [0, 2],  # X1 has parents X0 and X2 (collider)
        2: [3],     # X2 has parent X3
        3: []       # X3 has no parents
    }
    return dag, NUMERIC_COL_NAMES, NUMERIC_CATEGORICAL_COLS

def get_mixed_dag_and_config() -> tuple[dict[int, list[int]], list[str], list[int]]:
    """Get DAG and column info for mixed SCM (with categorical). DAG: X3 → X2 → X1 ← X0, X4_cat ← X1."""
    dag = {
        0: [],      # X0 has no parents
        1: [0, 2],  # X1 has parents X0 and X2 (collider)
        2: [3],     # X2 has parent X3
        3: [],      # X3 has no parents
        4: [1]      # X4_cat has parent X1
    }
    return dag, MIXED_COL_NAMES, MIXED_CATEGORICAL_COLS

def get_numeric_cpdag_and_config() -> tuple[np.ndarray, list[str], list[int]]:
    """Get CPDAG and column info for numeric SCM. CPDAG: X0 -> X1 <- X2 - X3."""
    cpdag_adj = np.array([
        [0, -1, 0, 0],    # X0 -> X1
        [1, 0, 1, 0],     # X1 <- X0, X1 <- X2
        [0, -1, 0, -1],   # X2 -> X1, X2 - X3
        [0, 0, -1, 0]     # X3 - X2
    ])
    return cpdag_adj, NUMERIC_COL_NAMES, NUMERIC_CATEGORICAL_COLS

def get_mixed_cpdag_and_config() -> tuple[np.ndarray, list[str], list[int]]:
    """Get CPDAG and column info for mixed SCM (with categorical). CPDAG: X0 -> X1 <- X2 - X3, X4_cat <- X1."""
    cpdag_adj = np.array([
        [0, -1, 0, 0, 0],    # X0 -> X1
        [1, 0, 1, 0, -1],    # X1 <- X0, X1 <- X2, X1 -> X4_cat
        [0, -1, 0, -1, 0],   # X2 -> X1, X2 - X3
        [0, 0, -1, 0, 0],    # X3 - X2
        [0, 1, 0, 0, 0]      # X4_cat <- X1
    ])
    return cpdag_adj, MIXED_COL_NAMES, MIXED_CATEGORICAL_COLS

