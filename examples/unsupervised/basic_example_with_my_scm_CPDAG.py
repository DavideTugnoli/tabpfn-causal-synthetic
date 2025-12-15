#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""Example demonstrating original CPDAG approach for synthetic data generation.

This example shows how to use the original CPDAG approach that treats
nodes with undirected edges as correlational and conditions them on all
previous features.
"""

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised


def generate_scm_data(n_samples: int, random_state: int = 42) -> np.ndarray:
    """Generate optimized SCM data: X4 → X3 → X2 ← X1 (collider at X2).
    
    Optimized for maximum collider bias (0.929):
    - X3 = 0.5 * X4 + ε₃ (noise=1e-5)
    - X2 = 5.0 * X1 + 10.0 * X3 + ε₂ (noise=1e-5)
    """
    rng = np.random.default_rng(random_state)
    X4 = rng.normal(0, 1, n_samples)
    X1 = rng.normal(0, 1, n_samples)
    X3 = 0.5 * X4 + rng.normal(0, 1e-5, n_samples)
    X2 = 5.0 * X1 + 10.0 * X3 + rng.normal(0, 1e-5, n_samples)
    return np.column_stack([X1, X2, X3, X4]).astype(np.float32)


# Generate SCM data with optimized collider bias
X = generate_scm_data(n_samples=1000, random_state=42)
attribute_names = ["X1", "X2", "X3", "X4"]

# Split data (only X needed for unsupervised experiment)
X_train, X_test = train_test_split(X, test_size=0.5, random_state=42)

# Initialize TabPFN models
clf = TabPFNClassifier(n_estimators=3)
reg = TabPFNRegressor(n_estimators=3)

# Initialize unsupervised model
model_unsupervised = unsupervised.TabPFNUnsupervisedModel(
    tabpfn_clf=clf,
    tabpfn_reg=reg,
)

# Define CPDAG structure: X4 - X3 → X2 ← X1
# CPDAG format: {node: {"parents": [directed_parents], "undirected": [undirected_neighbors]}}
cpdag = {
    0: {"parents": [], "undirected": []},        # X1 has no parents or undirected edges
    1: {"parents": [0, 2], "undirected": []},    # X2 has directed parents X1 and X3 (collider)
    2: {"parents": [], "undirected": [3]},       # X3 has undirected edge with X4
    3: {"parents": [], "undirected": [2]}        # X4 has undirected edge with X3
}

# Create and run synthetic experiment
exp_synthetic = unsupervised.experiments.GenerateSyntheticDataExperiment(
    task_type="unsupervised",
)

# Convert to torch tensor
X_tensor = torch.tensor(X_train, dtype=torch.float32)

print(f"SCM data: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Structure: X4 - X3 → X2 ← X1 (CPDAG format)")
print(f"Training set: {X_train.shape[0]} samples")
print(f"CPDAG: {cpdag}")

# Run experiment with CPDAG - use all 4 SCM variables
results = exp_synthetic.run(
    tabpfn=model_unsupervised,
    X=X_tensor,
    y=None,
    attribute_names=attribute_names,
    temp=1.0,
    n_samples=X_train.shape[0] * 3,
    indices=[0, 1, 2, 3],  # All SCM variables
    cpdag=cpdag,
)

import matplotlib.pyplot as plt
plt.show()