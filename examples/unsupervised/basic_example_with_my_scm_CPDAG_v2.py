#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""Example demonstrating CPDAG 2.0 approach for synthetic data generation.

This example shows how to use the new CPDAG 2.0 approach that leverages
column order to resolve undirected edges in causal graphs.
"""

import numpy as np
import torch
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
from tabpfn_extensions.unsupervised.experiments import GenerateSyntheticDataExperiment


def create_cpdag_v2_example():
    """Create a CPDAG example with undirected edges to demonstrate CPDAG 2.0."""
    
    # Create TabPFN models
    tabpfn_clf = TabPFNClassifier()
    tabpfn_reg = TabPFNRegressor()
    
    # Create the unsupervised model
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_clf, tabpfn_reg=tabpfn_reg)
    
    # Create synthetic data with known structure
    np.random.seed(42)
    n_samples = 1000
    
    # Generate data with structure: X -> Y - Z (where Y-Z is undirected)
    X = np.random.normal(0, 1, n_samples)
    Y = 0.5 * X + np.random.normal(0, 0.5, n_samples)  # Y depends on X
    Z = 0.3 * Y + np.random.normal(0, 0.7, n_samples)   # Z depends on Y
    
    # Create dataset with features in order: [X, Y, Z]
    data = np.column_stack([X, Y, Z])
    feature_names = ["X", "Y", "Z"]
    
    # Define CPDAG 2.0 structure
    # X -> Y (directed), Y - Z (undirected)
    cpdag_v2 = {
        0: {"parents": [], "undirected": []},           # X: no parents, no undirected
        1: {"parents": [0], "undirected": [2]},        # Y: parent X, undirected with Z
        2: {"parents": [], "undirected": [1]}          # Z: no parents, undirected with Y
    }
    
    # Fit the model
    model.fit(data)
    
    # Generate synthetic data using CPDAG 2.0
    synthetic_data = model.generate_synthetic_data(
        n_samples=500,
        t=1.0,
        n_permutations=3,
        cpdag_v2=cpdag_v2
    )
    
    print("Original data shape:", data.shape)
    print("Synthetic data shape:", synthetic_data.shape)
    print("CPDAG 2.0 structure:")
    for node, info in cpdag_v2.items():
        print(f"  Node {node} ({feature_names[node]}): parents={info['parents']}, undirected={info['undirected']}")
    
    # Run experiment for visualization
    experiment = GenerateSyntheticDataExperiment()
    experiment.run(
        model,
        X=torch.tensor(data, dtype=torch.float32),
        y=None,
        attribute_names=feature_names,
        temp=1.0,
        n_samples=500,
        n_permutations=3,
        cpdag_v2=cpdag_v2
    )
    
    print("\nExperiment completed. Check the generated plots to compare original vs synthetic data.")
    
    return model, data, synthetic_data, cpdag_v2


def compare_cpdag_approaches():
    """Compare original CPDAG vs CPDAG 2.0 approaches."""
    
    # Create models
    tabpfn_clf = TabPFNClassifier()
    tabpfn_reg = TabPFNRegressor()
    model = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_clf, tabpfn_reg=tabpfn_reg)
    
    # Create data
    np.random.seed(42)
    n_samples = 1000
    
    X = np.random.normal(0, 1, n_samples)
    Y = 0.5 * X + np.random.normal(0, 0.5, n_samples)
    Z = 0.3 * Y + np.random.normal(0, 0.7, n_samples)
    
    data = np.column_stack([X, Y, Z])
    feature_names = ["X", "Y", "Z"]
    
    # Define CPDAG structures
    cpdag_original = {
        0: {"parents": [], "undirected": []},
        1: {"parents": [0], "undirected": [2]},
        2: {"parents": [], "undirected": [1]}
    }
    
    cpdag_v2 = {
        0: {"parents": [], "undirected": []},
        1: {"parents": [0], "undirected": [2]},
        2: {"parents": [], "undirected": [1]}
    }
    
    # Fit model
    model.fit(data)
    
    # Generate with original CPDAG approach
    synthetic_original = model.generate_synthetic_data(
        n_samples=500,
        t=1.0,
        n_permutations=3,
        cpdag=cpdag_original
    )
    
    # Generate with CPDAG 2.0 approach
    synthetic_v2 = model.generate_synthetic_data(
        n_samples=500,
        t=1.0,
        n_permutations=3,
        cpdag_v2=cpdag_v2
    )
    
    print("Data shapes:")
    print(f"  Original: {data.shape}")
    print(f"  Synthetic (Original CPDAG): {synthetic_original.shape}")
    print(f"  Synthetic (CPDAG 2.0): {synthetic_v2.shape}")
    
    # Compare correlations
    print("\nCorrelation matrices:")
    print("Original data:")
    print(np.corrcoef(data.T))
    print("\nSynthetic data (Original CPDAG):")
    print(np.corrcoef(synthetic_original.numpy().T))
    print("\nSynthetic data (CPDAG 2.0):")
    print(np.corrcoef(synthetic_v2.numpy().T))
    
    return model, data, synthetic_original, synthetic_v2


if __name__ == "__main__":
    print("=== CPDAG 2.0 Example ===")
    model, data, synthetic_data, cpdag_v2 = create_cpdag_v2_example()
    
    print("\n=== Comparison of CPDAG Approaches ===")
    model, data, synthetic_original, synthetic_v2 = compare_cpdag_approaches()
