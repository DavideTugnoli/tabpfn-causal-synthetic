
"""CSuite dataset loader utilities.

This module provides utilities for loading CSuite benchmark datasets
that include DAG structure, variable metadata, and train/test splits.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_csuite_dataset(dataset_name: str, base_path: str | Path = None) -> dict[str, Any]:
    """Load a complete CSuite dataset with all metadata.
    
    Args:
        dataset_name: Name of the CSuite dataset directory
        base_path: Base path to csuite_datasets directory. If None, uses default location.
        
    Returns:
        Dictionary containing:
        - 'train_data': Training data as numpy array
        - 'test_data': Test data as numpy array  
        - 'dag': DAG structure as adjacency matrix (numpy array)
        - 'dag_dict': DAG as dictionary format {node: [parents]}
        - 'variables': Variable metadata from variables.json
        - 'column_names': List of variable names in order
        - 'categorical_columns': List of categorical/binary column names
        - 'n_features': Number of features
        
    Raises:
        FileNotFoundError: If dataset directory or required files don't exist
        ValueError: If dataset structure is invalid
    """
    if base_path is None:
        # Default to csuite_experiment/csuite_datasets/
        base_path = Path(__file__).parent.parent / "csuite_experiment" / "csuite_datasets"
    
    dataset_path = Path(base_path) / dataset_name
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    # Load required files
    train_file = dataset_path / "train.csv"
    test_file = dataset_path / "test.csv"
    adj_matrix_file = dataset_path / "adj_matrix.csv"
    variables_file = dataset_path / "variables.json"
    
    # Check all required files exist
    missing_files = []
    for file_path, name in [(train_file, "train.csv"), (test_file, "test.csv"), 
                           (adj_matrix_file, "adj_matrix.csv"), (variables_file, "variables.json")]:
        if not file_path.exists():
            missing_files.append(name)
    
    if missing_files:
        raise FileNotFoundError(f"Missing required files in {dataset_name}: {missing_files}")
    
    # Load train and test data (no headers in CSuite CSV files)
    train_data = np.loadtxt(str(train_file), delimiter=',')
    test_data = np.loadtxt(str(test_file), delimiter=',')
    
    # Load adjacency matrix
    adj_matrix = np.loadtxt(str(adj_matrix_file), delimiter=',', dtype=int)
    
    # Load variable metadata  
    with open(variables_file, 'r') as f:
        variables_metadata = json.load(f)
    
    # Extract variable information
    variables = variables_metadata['variables']
    n_features = len(variables)
    
    # Validate data dimensions
    if train_data.shape[1] != n_features:
        raise ValueError(f"Train data has {train_data.shape[1]} columns but {n_features} variables defined")
    if test_data.shape[1] != n_features:
        raise ValueError(f"Test data has {test_data.shape[1]} columns but {n_features} variables defined")
    if adj_matrix.shape != (n_features, n_features):
        raise ValueError(f"Adjacency matrix shape {adj_matrix.shape} doesn't match {n_features} variables")
    
    # Extract column names and types
    column_names = [var['name'] for var in variables]
    categorical_columns = []
    
    for i, var in enumerate(variables):
        var_type = var['type']
        if var_type in ['categorical', 'binary']:
            categorical_columns.append(column_names[i])
    
    # Convert adjacency matrix to DAG dictionary format
    dag_dict = adjacency_matrix_to_dag_dict(adj_matrix)
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'dag': adj_matrix,
        'dag_dict': dag_dict,
        'variables': variables,
        'column_names': column_names,
        'categorical_columns': categorical_columns,
        'n_features': n_features,
        'dataset_name': dataset_name
    }


def adjacency_matrix_to_dag_dict(adj_matrix: np.ndarray) -> dict[int, list[int]]:
    """Convert adjacency matrix to DAG dictionary format.
    
    Args:
        adj_matrix: Adjacency matrix where adj_matrix[i,j] = 1 means edge i -> j
        
    Returns:
        Dictionary mapping each node to its list of parents
        Format: {child_node: [parent1, parent2, ...]}
    """
    n_nodes = adj_matrix.shape[0]
    dag_dict = {}
    
    for child in range(n_nodes):
        parents = []
        for parent in range(n_nodes):
            if adj_matrix[parent, child] == 1:
                parents.append(parent)
        dag_dict[child] = parents
    
    return dag_dict


def get_variable_types_info(variables: list[dict]) -> dict[str, Any]:
    """Extract detailed variable type information.
    
    Args:
        variables: List of variable metadata dictionaries
        
    Returns:
        Dictionary with variable type statistics and mappings
    """
    type_counts = {'continuous': 0, 'categorical': 0, 'binary': 0}
    continuous_indices = []
    categorical_indices = []
    binary_indices = []
    
    for i, var in enumerate(variables):
        var_type = var['type']
        type_counts[var_type] += 1
        
        if var_type == 'continuous':
            continuous_indices.append(i)
        elif var_type == 'categorical':
            categorical_indices.append(i)
        elif var_type == 'binary':
            binary_indices.append(i)
    
    return {
        'type_counts': type_counts,
        'continuous_indices': continuous_indices,
        'categorical_indices': categorical_indices,
        'binary_indices': binary_indices,
        'total_variables': len(variables)
    }


def list_available_datasets(base_path: str | Path = None) -> list[str]:
    """List all available CSuite datasets.
    
    Args:
        base_path: Base path to csuite_datasets directory
        
    Returns:
        List of dataset directory names
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent / "csuite_experiment" / "csuite_datasets"
    
    base_path = Path(base_path)
    if not base_path.exists():
        return []
    
    datasets = []
    for item in base_path.iterdir():
        if item.is_dir() and (item / "variables.json").exists():
            datasets.append(item.name)
    
    return sorted(datasets)