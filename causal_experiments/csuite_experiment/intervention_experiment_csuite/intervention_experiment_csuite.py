#!/usr/bin/env python3
"""Interventional experiment for CSuite datasets with TabPFN.

This script implements an interventional experiment to test TabPFN's ability
to preserve Average Treatment Effect (ATE) in synthetic data generation using
CSuite causal datasets.

Experimental Design:
1. Load CSuite dataset with predefined interventions
2. For each repetition:
   - Sample training subset from observational data
   - Use intervention data as test set (ground truth)
   - Calculate ATE on intervention test data (ground truth)
   - Generate synthetic data using different TabPFN configurations
   - Calculate ATE on synthetic data
   - Measure |ATE_test - ATE_synthetic| as primary metric

TabPFN Configurations:
- vanilla: original/topological/reverse_topological
- dag: ground truth DAG
- cpdag_discovered, cpdag_minimal


Key Features:
- Intervention variable remapping to handle TabPFN's categorical normalization
- Supports CSuite datasets with intervention.json specifications
- Proper handling of various intervention value ranges (e.g., {-1, 1}, {0, 1})

Usage Examples:

    # Run ALL algorithms (default)
    uv run causal_experiments/csuite_experiment/intervention_experiment_csuite/intervention_experiment_csuite.py
    
    # Run in test mode (faster)
    uv run causal_experiments/csuite_experiment/intervention_experiment_csuite/intervention_experiment_csuite.py --test
    
    # Run specific algorithm
    uv run causal_experiments/csuite_experiment/intervention_experiment_csuite/intervention_experiment_csuite.py --algorithm vanilla
    
    # Run with specific dataset
    uv run causal_experiments/csuite_experiment/intervention_experiment_csuite/intervention_experiment_csuite.py --dataset csuite_lingauss
"""
from __future__ import annotations

import os
# Set environment for maximal determinism before importing torch
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from typing import Any, cast, List

# Fix imports - add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from causal_experiments.utils import (
    create_result_row,
    discover_cpdag_from_data,
    identify_vanilla_duplicates,
    copy_vanilla_duplicate_result,
    prepare_vanilla_data,
    prepare_dag_data,
    prepare_cpdag_data,
    save_results_to_csv,
    save_train_set_per_split,
    save_synthetic_data,
    save_global_test_set,
    load_global_test_set,
    # CSuite dataset utilities
    load_csuite_dataset,
    list_available_datasets,
    # Determinism utilities
    setup_determinism,
    set_experiment_seeds,
)

# Intervention remapping utilities
from causal_experiments.utils.intervention_remapping import (
    remap_interventions,
)

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised

# Global flag to print device info only once
_device_info_printed = False

_LINEAR_DATASETS = {"csuite_lingauss", "csuite_linexp"}

# Reuse cached train splits when they were generated with the independent sampling
# strategy; otherwise regenerate them to ensure consistent batches across algorithms.
_REUSE_CACHED_TRAIN_SPLITS = True


def _choose_indep_test(dataset_name: str, categorical_cols: list[str]) -> str:
    if categorical_cols:
        if dataset_name.lower() == "csuite_mixed_confounding":
            return "hybrid"
        return "gsq"
    if dataset_name.lower() in _LINEAR_DATASETS:
        return "fisherz"
    return "kci"


def apply_do_surgery_to_dag(
    dag: dict[int, list[int]],
    intervention_nodes: list[int],
) -> dict[int, list[int]]:
    """Return an interventional DAG by removing incoming edges to intervention nodes."""
    dag_copy = {int(child): [int(parent) for parent in parents] for child, parents in dag.items()}
    all_nodes = set(dag_copy.keys())
    for parents in dag_copy.values():
        all_nodes.update(parents)
    all_nodes.update(int(node) for node in intervention_nodes)

    for node in sorted(all_nodes):
        dag_copy.setdefault(node, [])

    for node in intervention_nodes:
        dag_copy[int(node)] = []

    return dag_copy


def save_crashed_seeds_to_csv(crashed_seeds_data: List[dict], output_file: Path):
    """Save crashed seeds information to CSV file.
    
    Args:
        crashed_seeds_data: List of dictionaries containing crash information
        output_file: Path to the crashed seeds CSV file
    """
    if not crashed_seeds_data:
        return
    
    df = pd.DataFrame(crashed_seeds_data)
    df.to_csv(output_file, index=False)
    print(f" Crashed seeds data saved to: {output_file}")

def load_csuite_intervention_data(dataset_name: str) -> dict:
    """Load CSuite dataset with intervention data support.
    
    This function implements the correct logic for interventional experiments:
    1. Load intervention data from interventions.json (not train.csv/test.csv)
    2. Parse intervention structure and variable types
    3. Set up proper sampling strategy for ATE estimation
    
    Args:
        dataset_name: Name of CSuite dataset
        
    Returns:
        Dictionary containing all CSuite data including intervention information
    """
    import json
    from pathlib import Path
    
    # Load base CSuite dataset for metadata (DAG, variable types, etc.)
    csuite_data = load_csuite_dataset(dataset_name)
    
    # Path to dataset directory 
    script_dir = Path(__file__).parent
    dataset_path = script_dir.parent / "csuite_datasets" / dataset_name
    
    # Load intervention metadata
    interventions_file = dataset_path / "interventions.json"
    if not interventions_file.exists():
        raise FileNotFoundError(f"Interventions file not found: {interventions_file}")
        
    with open(interventions_file, 'r') as f:
        interventions_data = json.load(f)
    
    # Parse intervention information - CSuite format:
    # {"environments": [{"intervention_idxs": [0], "effect_idxs": [1], 
    #                   "intervention_reference": [-1.0], "intervention_values": [1.0]}]}
    if 'environments' not in interventions_data or not interventions_data['environments']:
        raise ValueError(f"No intervention environments found in {interventions_file}")
    
    env = interventions_data['environments'][0]  # Use first environment
    
    # Extract intervention info
    intervention_indices = [int(idx) for idx in env.get('intervention_idxs', [])]
    intervention_idx = intervention_indices[0] if intervention_indices else None
    effect_idx = env['effect_idxs'][0] if env['effect_idxs'] else None
    
    if intervention_idx is None or effect_idx is None:
        raise ValueError("Could not determine intervention and effect variables")
    
    column_names = csuite_data['column_names']
    intervention_var = column_names[intervention_idx]
    target_var = column_names[effect_idx]
    
    # Extract intervention values from JSON specification
    intervention_reference = env.get('intervention_reference', [])
    intervention_values_list = env.get('intervention_values', [])
    
    # Handle both array and single value formats
    # Convert single values to arrays for consistency
    if isinstance(intervention_reference, (int, float)):
        intervention_reference = [intervention_reference]
    if isinstance(intervention_values_list, (int, float)):
        intervention_values_list = [intervention_values_list]
    
    # Create the binary intervention: reference vs intervention  
    if len(intervention_reference) == 1 and len(intervention_values_list) == 1:
        intervention_values = [intervention_reference[0], intervention_values_list[0]]
    else:
        raise ValueError(f"Expected single reference and intervention value, got {intervention_reference}, {intervention_values_list}")
    
    # Load the actual intervention data from JSON
    reference_data = np.array(env.get('reference_data', []))  # intervention = reference value
    intervention_data = np.array(env.get('test_data', []))    # intervention = intervention value
    
    if reference_data.size == 0 or intervention_data.size == 0:
        raise ValueError("Missing reference_data or test_data in interventions.json")
    
    print(f" Intervention data loaded:")
    print(f"   📈 Reference data: {len(reference_data)} samples with intervention={intervention_reference[0]}")
    print(f"    Intervention data: {len(intervention_data)} samples with intervention={intervention_values_list[0]}")
    print(f"   📈 Intervention variable: {intervention_var} (index {intervention_idx})")
    print(f"    Target variable: {target_var} (index {effect_idx})")
    print(f"   🔢 Intervention values: {intervention_values}")
    
    # Create intervention info structure
    intervention_info = {
        'variable': intervention_var,
        'target': target_var, 
        'values': intervention_values,
        'intervention_indices': intervention_indices,
        'intervention_idx': intervention_idx,
        'effect_idx': effect_idx,
        'reference_data': reference_data,
        'intervention_data': intervention_data
    }
    
    # Add intervention info to CSuite data
    csuite_data['intervention_info'] = intervention_info
    
    return csuite_data

def calculate_ate_csuite(data: np.ndarray, column_names: List[str], intervention_info: dict) -> float:
    """Calculate Average Treatment Effect for CSuite dataset.
    
    ATE = E[Y|do(X=intervention_high)] - E[Y|do(X=intervention_low)]
    
    Args:
        data: Data array 
        column_names: List of column names
        intervention_info: CSuite intervention information with:
            - variable: name of intervention variable
            - values: list of intervention values
            - target: name of target/outcome variable
    
    Returns:
        ATE value
    """
    try:
        # Get intervention variable and target variable indices
        intervention_var = intervention_info['variable']
        target_var = intervention_info['target']
        
        intervention_idx = column_names.index(intervention_var)
        target_idx = column_names.index(target_var)
        
        # Get intervention values - should be exactly 2 values
        intervention_values = intervention_info['values']
        if len(intervention_values) != 2:
            print(f"  Warning: Expected 2 intervention values, got {len(intervention_values)}: {intervention_values}")
            return 0.0
            
        # Use the actual intervention values from the intervention info (e.g., [-1.0, 1.0])
        intervention_values = intervention_info['values']
        
        # Calculate mean outcome for each intervention level
        outcomes_by_intervention = {}
        for intervention_val in intervention_values:
            mask = data[:, intervention_idx] == intervention_val
            if np.any(mask):
                outcomes_by_intervention[intervention_val] = np.mean(data[mask, target_idx])
        
        if len(outcomes_by_intervention) != 2:
            print(f"  Warning: Could not find both intervention levels in data")
            return 0.0
        
        # Calculate ATE: high_value - low_value
        high_val = max(intervention_values)
        low_val = min(intervention_values)
        ate = outcomes_by_intervention[high_val] - outcomes_by_intervention[low_val]
        
        return float(ate)
        
    except (ValueError, KeyError) as e:
        print(f"❌ Error calculating ATE: {e}")
        return 0.0

def create_interventional_sampling_strategy(
    intervention_info: dict,
    intervention_mappings: dict,
    train_size: int = 1000,
    test_size: int = 500,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Create sampling strategy for interventional experiment.
    
    Strategy: 
    1. Sample 500 samples from reference branch and 500 from intervention branch for training
    2. Use remaining samples as test set for ATE calculation
    3. Map intervention values in training data to categorical integers (0, 1) for TabPFN compatibility
    4. Keep original values in test data for ATE calculation
    5. Ensure same sampling across algorithms for fair comparison
    
    Args:
        intervention_info: Intervention information from CSuite
        intervention_mappings: Mapping dictionary for intervention values
        train_size: Total training size (500 + 500 = 1000 by default)
        test_size: Test set size for ATE calculation
        seed: Random seed for reproducible sampling
        
    Returns:
        Tuple of (train_data_mapped, test_data) as numpy arrays
        - train_data_mapped: Training data with intervention values mapped to 0/1
        - test_data: Test data with original intervention values
    """
    reference_data = intervention_info['reference_data']
    intervention_data = intervention_info['intervention_data']
    
    # Calculate samples per branch for training
    samples_per_branch = train_size // 2
    
    # Set random seed for reproducible sampling
    rng = np.random.default_rng(seed)
    
    # Sample from reference branch
    if len(reference_data) < samples_per_branch:
        print(f"  Warning: Only {len(reference_data)} reference samples available, using all")
        reference_train = reference_data
        reference_test = np.array([])
    else:
        # Sample without replacement
        ref_indices = rng.choice(len(reference_data), size=samples_per_branch, replace=False)
        reference_train = reference_data[ref_indices]
        # Use remaining as test
        remaining_ref = np.setdiff1d(np.arange(len(reference_data)), ref_indices)
        reference_test = reference_data[remaining_ref]
    
    # Sample from intervention branch
    if len(intervention_data) < samples_per_branch:
        print(f"  Warning: Only {len(intervention_data)} intervention samples available, using all")
        intervention_train = intervention_data
        intervention_test = np.array([])
    else:
        # Sample without replacement
        int_indices = rng.choice(len(intervention_data), size=samples_per_branch, replace=False)
        intervention_train = intervention_data[int_indices]
        # Use remaining as test
        remaining_int = np.setdiff1d(np.arange(len(intervention_data)), int_indices)
        intervention_test = intervention_data[remaining_int]
    
    # Combine training data (500 + 500 = 1000)
    train_data = np.concatenate([reference_train, intervention_train], axis=0)
    
    # Combine test data (remaining samples)
    test_data = np.concatenate([reference_test, intervention_test], axis=0)
    
    # Shuffle training data to mix reference and intervention samples
    train_indices = rng.permutation(len(train_data))
    train_data = train_data[train_indices]
    
    print(f" Sampling strategy (seed={seed}):")
    print(f"    Training: {len(reference_train)} reference + {len(intervention_train)} intervention = {len(train_data)} total")
    print(f"   📈 Test: {len(reference_test)} reference + {len(intervention_test)} intervention = {len(test_data)} total")
    print(f"   🔢 Original intervention values in train: {sorted(np.unique(train_data[:, intervention_info['intervention_idx']]))}")
    
    # Apply intervention mappings to training data for TabPFN compatibility
    # Map original intervention values (e.g., 0.5, 2.5) to categorical integers (0, 1)
    train_data_mapped = train_data.copy()
    intervention_idx = intervention_info['intervention_idx']
    
    if intervention_idx in intervention_mappings:
        mapping = intervention_mappings[intervention_idx]
        # Create reverse mapping: original_value -> categorical_integer
        reverse_mapping = {v: k for k, v in mapping.items()}
        
        print(f" Applying forward intervention mapping to training data:")
        for original_val, categorical_int in reverse_mapping.items():
            mask = train_data_mapped[:, intervention_idx] == original_val
            if np.any(mask):
                train_data_mapped[mask, intervention_idx] = categorical_int
                print(f"   📈 {original_val} → {categorical_int} ({np.sum(mask)} samples)")
        
        print(f"   🔢 Mapped intervention values in train: {sorted(np.unique(train_data_mapped[:, intervention_idx]))}")
    
    # Test data keeps original values for ATE calculation
    return train_data_mapped, test_data

def create_global_test_set_and_remaining_pool(
    intervention_info: dict,
    test_seed: int = 2000
) -> tuple[np.ndarray, dict]:
    """Create a fixed global test set and remaining pool for all experiments.
    
    This function creates:
    1. A fixed test set that will be used consistently across all experiments
    2. A remaining pool (original data - test set) for training data sampling
    
    This ensures zero data leakage between train and test sets.
    
    Args:
        intervention_info: Intervention information from CSuite
        test_seed: Fixed seed for test set generation
        
    Returns:
        Tuple of (global_test_set, remaining_pool_info):
        - global_test_set: Fixed test set as numpy array
        - remaining_pool_info: Dict with remaining reference/intervention data
    """
    reference_data = intervention_info['reference_data']
    intervention_data = intervention_info['intervention_data']
    
    # Set random seed for reproducible test set generation
    rng = np.random.default_rng(test_seed)
    
    # Calculate test samples per branch (use 50% of available data for test)
    total_reference = len(reference_data)
    total_intervention = len(intervention_data)
    
    # Use 50% of each branch for test set
    test_ref_samples = min(total_reference // 2, 1500)  # Max 1500 per branch
    test_int_samples = min(total_intervention // 2, 1500)
    
    # Sample from reference branch for test and track indices
    if total_reference < test_ref_samples:
        print(f"  Warning: Only {total_reference} reference samples available for test, using all")
        ref_test_indices = np.arange(total_reference)
        reference_test = reference_data
    else:
        ref_test_indices = rng.choice(total_reference, size=test_ref_samples, replace=False)
        reference_test = reference_data[ref_test_indices]
    
    # Sample from intervention branch for test and track indices
    if total_intervention < test_int_samples:
        print(f"  Warning: Only {total_intervention} intervention samples available for test, using all")
        int_test_indices = np.arange(total_intervention)
        intervention_test = intervention_data
    else:
        int_test_indices = rng.choice(total_intervention, size=test_int_samples, replace=False)
        intervention_test = intervention_data[int_test_indices]
    
    # Create remaining pool by excluding test indices
    ref_remaining_mask = np.ones(total_reference, dtype=bool)
    ref_remaining_mask[ref_test_indices] = False
    reference_remaining = reference_data[ref_remaining_mask]
    
    int_remaining_mask = np.ones(total_intervention, dtype=bool)
    int_remaining_mask[int_test_indices] = False
    intervention_remaining = intervention_data[int_remaining_mask]
    
    # Combine test data
    global_test_set = np.concatenate([reference_test, intervention_test], axis=0)
    
    # Create remaining pool info
    remaining_pool_info = {
        'reference_data': reference_remaining,
        'intervention_data': intervention_remaining,
        'reference_original_size': total_reference,
        'intervention_original_size': total_intervention,
        'reference_remaining_size': len(reference_remaining),
        'intervention_remaining_size': len(intervention_remaining)
    }
    
    print(f" Global test set and remaining pool created (seed={test_seed}):")
    print(f"   📈 Reference test samples: {len(reference_test)}")
    print(f"   📈 Intervention test samples: {len(intervention_test)}")
    print(f"   📈 Total test samples: {len(global_test_set)}")
    print(f"    Reference remaining for training: {len(reference_remaining)}")
    print(f"    Intervention remaining for training: {len(intervention_remaining)}")
    print(f"    Zero data leakage guaranteed!")
    
    return global_test_set, remaining_pool_info

def save_remaining_pool(
    remaining_pool_info: dict,
    base_output_dir: Path,
    column_names: list
) -> Path:
    """Save remaining pool data to .npz file for reproducibility.
    
    Args:
        remaining_pool_info: Remaining pool information
        base_output_dir: Base output directory
        column_names: Column names for the data
        
    Returns:
        Path to saved remaining pool file
    """
    datasets_dir = base_output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    remaining_pool_file = datasets_dir / "remaining_pool.npz"
    
    # Combine remaining pool data
    remaining_combined = np.concatenate([
        remaining_pool_info['reference_data'],
        remaining_pool_info['intervention_data']
    ], axis=0)
    
    # Save with metadata
    np.savez_compressed(
        remaining_pool_file,
        data=remaining_combined,
        column_names=column_names,
        reference_size=remaining_pool_info['reference_remaining_size'],
        intervention_size=remaining_pool_info['intervention_remaining_size'],
        reference_original_size=remaining_pool_info['reference_original_size'],
        intervention_original_size=remaining_pool_info['intervention_original_size']
    )
    
    print(f" Saved remaining pool: {remaining_pool_file} ({len(remaining_combined)} samples)")
    return remaining_pool_file


def load_remaining_pool(
    base_output_dir: Path,
    column_names: list[str]
) -> dict | None:
    """Load remaining pool data if previously saved."""
    remaining_file = base_output_dir / "datasets" / "remaining_pool.npz"
    if not remaining_file.exists():
        return None

    try:
        with np.load(remaining_file, allow_pickle=True) as data:
            combined = data["data"]
            saved_cols = data["column_names"].tolist()
            ref_size = int(data["reference_size"])
            int_size = int(data["intervention_size"])
            ref_original = int(data["reference_original_size"])
            int_original = int(data["intervention_original_size"])
    except Exception as exc:
        print(f"  Failed to load remaining pool from {remaining_file}: {exc}")
        return None

    if list(saved_cols) != column_names:
        print(
            "  Ignoring stored remaining pool due to column mismatch; regenerating from source data."
        )
        return None

    if ref_size + int_size != combined.shape[0]:
        print(
            "  Stored remaining pool has inconsistent sizes; regenerating from source data."
        )
        return None

    reference_data = combined[:ref_size]
    intervention_data = combined[ref_size:ref_size + int_size]
    return {
        'reference_data': reference_data,
        'intervention_data': intervention_data,
        'reference_original_size': ref_original,
        'intervention_original_size': int_original,
        'reference_remaining_size': ref_size,
        'intervention_remaining_size': int_size,
    }

def create_interventional_sampling_strategy_from_remaining_pool(
    remaining_pool_info: dict,
    intervention_info: dict,
    intervention_mappings: dict,
    global_test_set: np.ndarray,
    train_size: int = 1000,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy helper kept for backward compatibility.

    Generates a training split for a single ``train_size`` using the independent
    sampling routine, ensuring consistency with other train sizes for the same
    seed.
    """
    train_sets = generate_interventional_train_sets(
        remaining_pool_info=remaining_pool_info,
        intervention_info=intervention_info,
        intervention_mappings=intervention_mappings,
        train_sizes=[train_size],
        seed=seed,
    )
    train_data = train_sets.get(train_size)
    if train_data is None:
        raise ValueError(
            f"Failed to generate training data for train_size={train_size} and seed={seed}"
        )
    return train_data, global_test_set


def generate_interventional_train_sets(
    *,
    remaining_pool_info: dict,
    intervention_info: dict,
    intervention_mappings: dict,
    train_sizes: list[int],
    seed: int
) -> dict[int, np.ndarray]:
    """Generate independent train splits for all requested train sizes for a seed."""

    unique_sizes = sorted({int(ts) for ts in train_sizes if ts is not None})
    if not unique_sizes:
        return {}

    reference_remaining = remaining_pool_info['reference_data']
    intervention_remaining = remaining_pool_info['intervention_data']

    if len(reference_remaining) == 0 or len(intervention_remaining) == 0:
        raise ValueError("Remaining pool does not contain enough samples to generate training data")

    rng = np.random.default_rng(seed)
    intervention_idx = intervention_info['intervention_idx']

    train_sets: dict[int, np.ndarray] = {}
    for train_size in unique_sizes:
        samples_per_branch = max(train_size // 2, 0)

        available_ref = len(reference_remaining)
        available_int = len(intervention_remaining)

        samples_per_branch = min(samples_per_branch, available_ref, available_int)

        if samples_per_branch == 0:
            print(
                f"  Unable to sample train_size={train_size} for seed={seed}: not enough data per branch."
            )
            train_sets[train_size] = np.empty((0, reference_remaining.shape[1]))
            continue

        ref_indices = rng.choice(available_ref, size=samples_per_branch, replace=False)
        int_indices = rng.choice(available_int, size=samples_per_branch, replace=False)

        reference_selected = reference_remaining[ref_indices]
        intervention_selected = intervention_remaining[int_indices]

        train_data = np.concatenate([reference_selected, intervention_selected], axis=0)
        shuffle_perm = rng.permutation(len(train_data))
        train_data = train_data[shuffle_perm]

        train_data_mapped = train_data.copy()
        if intervention_idx in intervention_mappings:
            mapping = intervention_mappings[intervention_idx]
            reverse_mapping = {v: k for k, v in mapping.items()}
            for original_val, categorical_int in reverse_mapping.items():
                mask = train_data_mapped[:, intervention_idx] == original_val
                if np.any(mask):
                    train_data_mapped[mask, intervention_idx] = categorical_int

        actual_size = train_data_mapped.shape[0]
        if actual_size != train_size:
            print(
                f"  Generated train split of size {actual_size} (requested {train_size}) "
                f"for seed={seed} due to limited samples per branch."
            )

        train_sets[train_size] = train_data_mapped

    return train_sets

def run_single_experiment(
    X_train: np.ndarray,
    test_df: pd.DataFrame,
    algorithm: str,
    column_order: str,
    dag: dict | None,
    cpdag: dict | None,
    column_names: List[str],
    categorical_cols: List[str],
    intervention_info: dict,
    intervention_mappings: dict,
    column_ordering_used: List[int] | None = None,
    updated_categorical_features: List[int] | None = None,
    n_permutations: int = 3,
    temp: float = 1.0,
    n_estimators: int = 3,
    seed: int | None = None,
    causal_structures_last: bool = False,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Run a single experimental configuration for CSuite interventional experiment.

    Args:
        X_train: Training data (already prepared/processed)
        test_df: Test data as DataFrame for evaluation
        algorithm: Algorithm name
        column_order: Column ordering strategy
        dag: DAG structure (already prepared, if applicable)
        cpdag: CPDAG structure (already prepared, if applicable)
        column_names: List of column names
        categorical_cols: List of categorical column names
        intervention_info: CSuite intervention information
        intervention_mappings: Mapping for intervention variable remapping
        column_ordering_used: List of column indices used for reordering (if any)
        updated_categorical_features: List of indices of categorical features in the reordered X_train
        n_permutations: Number of permutations for generation
        temp: Temperature for sampling
        causal_structures_last: If True, generate causal structures after correlational nodes

    Returns:
        Tuple of (metrics_dict, synthetic_dataframe)
    """
    try:
        # Create fresh TabPFN models for this experiment to avoid state contamination
        clf = TabPFNClassifier(n_estimators=n_estimators)
        reg = TabPFNRegressor(n_estimators=n_estimators)
        
        # Debug: Print device info for first experiment only to avoid spam
        global _device_info_printed
        if not _device_info_printed:
            print(f" CUDA available: {torch.cuda.is_available()}")
            print(f" CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
            print(f" TabPFN Classifier device: {clf.device}")
            print(f" TabPFN Regressor device: {reg.device}")
            # Check what get_device actually returns
            from tabpfn_extensions.utils import get_device
            print(f" get_device('auto') returns: {get_device('auto')}")
            _device_info_printed = True
        
        # Initialize fresh unsupervised model 
        model_unsupervised = unsupervised.TabPFNUnsupervisedModel(
            tabpfn_clf=clf,
            tabpfn_reg=reg,
        )
        
        # Set categorical features: CSuite metadata + intervention variable
        # Start with categorical features from CSuite metadata
        if categorical_cols:
            # Use updated categorical features if available (for vanilla with reordering)
            if updated_categorical_features is not None:
                categorical_indices = updated_categorical_features
            else:
                # Convert categorical column names to indices
                categorical_indices = [column_names.index(col) for col in categorical_cols]
        else:
            categorical_indices = []
        model_unsupervised.set_categorical_features(categorical_indices)
        print(f"  Final categorical features: {categorical_indices}")
        
        # Convert to torch tensor (data is already prepared)
        X_tensor = torch.tensor(X_train, dtype=torch.float32)

        # Create and run the GenerateSyntheticDataExperiment
        exp_synthetic = unsupervised.experiments.GenerateSyntheticDataExperiment(
            task_type="unsupervised"
        )

        # CPDAG v2 removed: pass only standard CPDAG when required
        cpdag_to_pass = cpdag

        # Determine feature names to reflect the CURRENT column order seen by the model
        # If a reordering was applied (including randomized replacement for 'original'),
        # reflect it regardless of the 'column_order' label.
        if column_ordering_used is not None and (
            algorithm == "vanilla" or algorithm == "dag" or algorithm.startswith("cpdag_")
        ):
            feature_names_for_run = [column_names[i] for i in column_ordering_used]
        else:
            feature_names_for_run = column_names

        # Run the experiment using their system
        exp_synthetic.run(
            tabpfn=model_unsupervised,
            X=X_tensor,
            y=None,  # Not used in unsupervised
            attribute_names=feature_names_for_run,
            temp=temp,
            n_samples=len(test_df),  # Generate same amount as test set
            n_permutations=n_permutations,
            dag=dag,
            cpdag=cpdag_to_pass,
            causal_structures_last=causal_structures_last,
            indices=list(range(len(column_names))),  # Use number of features from dataset
        )

        # Get the real and synthetic DataFrames from the experiment
        real_df = exp_synthetic.data_real
        synthetic_df = exp_synthetic.data_synthetic
        
        # Remove the 'real_or_synthetic' column that TabPFN adds for evaluation
        if 'real_or_synthetic' in real_df.columns:
            real_df = real_df.drop('real_or_synthetic', axis=1)
        if 'real_or_synthetic' in synthetic_df.columns:
            synthetic_df = synthetic_df.drop('real_or_synthetic', axis=1)
        
        # Fix column ordering issue for algorithms with reordering (vanilla, DAG, CPDAG)
        # Restore to original order whenever reordering was applied, regardless of label.
        if column_ordering_used is not None and (
            algorithm == "vanilla" or algorithm == "dag" or algorithm.startswith("cpdag_")
        ):
            # Create reverse mapping to restore original column order
            reverse_ordering = [column_ordering_used.index(i) for i in range(len(column_ordering_used))]
            
            # Reorder synthetic data to match original column order
            synthetic_data_reordered = np.zeros_like(exp_synthetic.synthetic_X)
            for orig_idx, reordered_idx in enumerate(reverse_ordering):
                synthetic_data_reordered[:, orig_idx] = exp_synthetic.synthetic_X[:, reordered_idx]
            
            # Recreate synthetic DataFrame with correct column mapping
            synthetic_df = pd.DataFrame({
                **dict(zip(column_names, [synthetic_data_reordered[:, i] for i in range(len(column_names))])),
            })
        
        # CRITICAL: Apply intervention remapping to synthetic data
        print(f" Applying intervention remapping...")
        synthetic_data_np = synthetic_df[column_names].values
        synthetic_data_remapped = remap_interventions(synthetic_data_np, intervention_mappings)
        
        # Debug: Check synthetic intervention values before/after remapping
        print(f" Synthetic intervention values (before): {sorted(np.unique(synthetic_data_np[:, intervention_info['intervention_idx']]))[:5]}...")
        print(f" Synthetic intervention values (after): {sorted(np.unique(synthetic_data_remapped[:, intervention_info['intervention_idx']]))[:5]}...")
        
        # Update synthetic DataFrame with remapped values
        synthetic_df = pd.DataFrame({
            **dict(zip(column_names, [synthetic_data_remapped[:, i] for i in range(len(column_names))])),
        })
        
        # INTERVENTIONAL EXPERIMENT: Calculate ATE metrics
        
        # Convert test DataFrame to numpy for ATE calculation
        test_data_np = test_df[column_names].values
        
        # Calculate ATE on test data (ground truth)
        ate_test = calculate_ate_csuite(test_data_np, column_names, intervention_info)
        
        # Calculate ATE on remapped synthetic data
        ate_synthetic = calculate_ate_csuite(synthetic_data_remapped, column_names, intervention_info)
        
        # Calculate ATE difference (primary metric)
        ate_difference = abs(ate_test - ate_synthetic)
        
        # Create metrics dictionary focused on ATE
        metrics = {
            "ate_test": ate_test,
            "ate_synthetic": ate_synthetic, 
            "ate_difference": ate_difference,
            "ate_relative_error": abs(ate_difference / ate_test) if ate_test != 0 else float('inf'),
            "n_synthetic_samples": len(synthetic_data_remapped),
            "n_test_samples": len(test_data_np)
        }
        
        return metrics, synthetic_df

    except Exception as e:
        # Print the actual error for debugging
        print(f"Error in run_single_experiment: {e}")
        import traceback
        print(traceback.format_exc())
        # Return default error values for ATE metrics
        return {
            "ate_test": -1.0,
            "ate_synthetic": -1.0,
            "ate_difference": -1.0,
            "ate_relative_error": -1.0,
            "n_synthetic_samples": -1,
            "n_test_samples": -1
        }, pd.DataFrame()  # Return empty DataFrame on error
    
    finally:
        # Light cleanup - let PyTorch handle GPU memory automatically
        # Heavy cleanup every 10 experiments is handled in main loop
        pass

def main(
    test_mode: bool = False, 
    save_every: int = 100, 
    save_datasets: bool = False, 
    save_reordered: bool = False, 
    save_synthetic: bool = False, 
    algorithm_filter: str | None = None,
    dataset_name: str = "csuite_lingauss",
    column_order_filter: str | None = None,
    repetitions: int | None = None,
    train_sizes_group: str | None = None,
    skip_seeds: list[int] | None = None,
):
    """Main experimental pipeline.
    
    Args:
        test_mode: If True, run with fewer experiments for testing
        save_every: Save CSV results every N experiments
        save_datasets: If True, save train/test datasets to .npz files
        save_reordered: If True, save reordered datasets for verification
        save_synthetic: If True, save synthetic data for analysis
        algorithm_filter: If specified, run only this algorithm
        dataset_name: Name of CSuite dataset to use
        column_order_filter: For vanilla algorithm, specify column ordering
        skip_seeds: Explicit seed values to skip (their repetitions are skipped and replacements are sampled to maintain total repetitions)
    """
    print("🧪 CSuite Interventional Experiment for TabPFN")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    if test_mode:
        print("⚡ Running in TEST MODE (fewer experiments)")
    print()

    cpdag_indep_test = "kci"
    
    # ======================
    # EXPERIMENTAL PARAMETERS
    # ======================

    if test_mode:
        TRAIN_SIZES = [100, 500]  # Reduced sizes for testing
        # Small default in test mode unless overridden
        N_REPETITIONS = repetitions if repetitions is not None else 3
    else:
        TRAIN_SIZES = [20, 50, 100, 200, 500, 1000]  # All train sizes for causal effect experiments
        # Default repetitions for full runs (can be overridden)
        N_REPETITIONS = repetitions if repetitions is not None else 150

    # Optional: restrict to small/large train-size groups (for walltime safety)
    group_small = [20, 50, 100]
    group_large = [200, 500, 1000]
    train_sizes_group = locals().get('train_sizes_group', None)
    if train_sizes_group:
        if train_sizes_group == 'small':
            TRAIN_SIZES = [ts for ts in TRAIN_SIZES if ts in group_small]
        elif train_sizes_group == 'large':
            TRAIN_SIZES = [ts for ts in TRAIN_SIZES if ts in group_large]
        elif train_sizes_group == 'all':
            pass
        else:
            raise ValueError("train_sizes_group must be one of: small, large, all")
    
    # TabPFN parameters
    N_ESTIMATORS = 3
    N_PERMUTATIONS = 3
    TEMPERATURE = 1.0
    
    # Linear seed approach: 0,1,2,3,4,5... (same seed = same dataset for all algorithms = fair comparison)

    skip_seeds_set = set(skip_seeds or [])
    if skip_seeds_set:
        print(f"  Requested skip seeds: {sorted(skip_seeds_set)}")

    train_size_to_seeds: dict[int, list[int]] = {ts: [] for ts in TRAIN_SIZES}
    seed_to_metadata: dict[int, tuple[int, int]] = {}
    skipped_values: list[int] = []
    next_seed = 0

    for train_size in TRAIN_SIZES:
        for rep_idx in range(N_REPETITIONS):
            while next_seed in skip_seeds_set:
                skipped_values.append(next_seed)
                next_seed += 1
            seed = next_seed
            train_size_to_seeds[train_size].append(seed)
            seed_to_metadata[seed] = (train_size, rep_idx + 1)
            next_seed += 1

    if skipped_values:
        print(f"   • Skipped seeds (removed from schedule): {skipped_values}")

    print("  Linear seed allocation per train size:")
    for ts in TRAIN_SIZES:
        seeds = train_size_to_seeds[ts]
        if seeds:
            print(f"   train_size={ts}: {len(seeds)} seeds (range {seeds[0]}–{seeds[-1]})")
    
    # Output configuration - consolidated CSV files (no per-train-size split)
    results_root = Path(
        os.environ.get(
            "CSUITE_INTERVENTIONAL_RESULTS_ROOT",
            str(Path(__file__).parent / "results"),
        )
    )
    base_output_dir = results_root / dataset_name
    base_output_dir.mkdir(parents=True, exist_ok=True)  # Create base directory with dataset name
    
    # Function to get output directory and consolidated result file path
    def get_output_paths(train_size: int | None = None, algorithm_filter: str | None = None, column_order_filter: str | None = None) -> tuple[Path, Path]:
        """Return output directory and consolidated CSV path independent of train size."""
        out_dir = base_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"csuite_interventional_results_{dataset_name}"
        name_parts: list[str] = []
        if algorithm_filter:
            name_parts.append(algorithm_filter)
        if column_order_filter:
            name_parts.append(column_order_filter)

        suffix = "_" + "_".join(name_parts) if name_parts else ""
        output_file = out_dir / f"{base_name}{suffix}.csv"
        return out_dir, output_file

    # ======================
    # CSUITE DATASET LOADING
    # ======================
    
    print(f" Loading CSuite dataset: {dataset_name}")
    try:
        dataset_info = load_csuite_intervention_data(dataset_name)
        
        # Extract CSuite metadata
        dag_observational = dataset_info['dag_dict']
        intervention_info = dataset_info['intervention_info']  # Our parsed intervention info
        column_names = dataset_info['column_names']
        categorical_cols = dataset_info['categorical_columns']  # Note: different key name
        intervention_nodes = intervention_info.get(
            "intervention_indices",
            [intervention_info["intervention_idx"]],
        )
        dag_dict = apply_do_surgery_to_dag(dag_observational, intervention_nodes)
        
        print(f" Dataset loaded successfully:")
        print(f"    Observational DAG structure: {dag_observational}")
        print(
            f"    Interventional DAG structure (do-surgery on nodes {intervention_nodes}): {dag_dict}"
        )
        print(f"   📈 Intervention variable: {intervention_info['variable']} → {intervention_info['target']}")
        print(f"   🔢 Intervention values: {intervention_info['values']}")
        print(f"     Categorical features: {categorical_cols if categorical_cols else 'None'}")
        print(f"    Reference data available: {len(intervention_info['reference_data'])} samples")
        print(f"    Intervention data available: {len(intervention_info['intervention_data'])} samples")
        cpdag_indep_test = _choose_indep_test(dataset_name, categorical_cols)
        print(f"    Independence test for CPDAG discovery: {cpdag_indep_test}")
        alpha = 0.05

    except Exception as e:
        print(f"❌ Failed to load dataset {dataset_name}: {e}")
        print(f"💡 Available datasets: {list_available_datasets()}")
        import traceback
        print(traceback.format_exc())
        return

    # ======================
    # INTERVENTION MAPPING SETUP
    # ======================
    
    print(f"\n Setting up intervention variable mapping...")
    # Create binary mapping based on actual intervention values from JSON
    intervention_idx = intervention_info['intervention_idx']
    intervention_values = intervention_info['values']
    intervention_mappings = {
        intervention_idx: {
            0: intervention_values[0],  # TabPFN 0 → first intervention value 
            1: intervention_values[1]   # TabPFN 1 → second intervention value
        }
    }
    print(f" Intervention mappings: {intervention_mappings}")
    print(f"💡 Using binary intervention values: {intervention_info['values']}")

    # ======================
    # GLOBAL TEST SET CREATION (FIXED FOR ALL EXPERIMENTS)
    # ======================
    
    print(f"\n Creating fixed global test set and remaining pool...")
    loaded_test = load_global_test_set(base_output_dir)
    loaded_pool = load_remaining_pool(base_output_dir, column_names)
    regenerate_test_assets = True

    if loaded_test is not None and loaded_pool is not None:
        loaded_array, loaded_cols = loaded_test
        if list(loaded_cols) == column_names and loaded_array.shape[1] == len(column_names):
            global_test_set = loaded_array
            remaining_pool_info = loaded_pool
            regenerate_test_assets = False
            print(" Loaded global test set and remaining pool from disk")
        else:
            print("  Stored global test set does not match current schema; regenerating from source data.")

    if regenerate_test_assets:
        global_test_set, remaining_pool_info = create_global_test_set_and_remaining_pool(
            intervention_info=intervention_info,
            test_seed=2000  # Fixed seed for consistent test set
        )

    if len(global_test_set) != 2000:
        raise ValueError(
            f"Expected CSuite interventional test set to contain exactly 2000 samples, got {len(global_test_set)}"
        )
    
    # Calculate ground truth ATE on global test set
    test_ate_ground_truth = calculate_ate_csuite(global_test_set, column_names, intervention_info)
    print(f"Ground truth ATE on global test set: {test_ate_ground_truth:.6f}")
    
    # Note: global_test_df is not used directly, but the global_test_set is passed to the sampling function
    
    print(f" Global test set and remaining pool created for all experiments")
    print(f"   📈 Test samples: {len(global_test_set)}")
    print(f"    Ground truth ATE: {test_ate_ground_truth:.6f}")
    print(f"    Remaining pool sizes: {remaining_pool_info['reference_remaining_size']} reference, {remaining_pool_info['intervention_remaining_size']} intervention")
    print(f"    Total samples: {remaining_pool_info['reference_original_size']} + {remaining_pool_info['intervention_original_size']} = {remaining_pool_info['reference_original_size'] + remaining_pool_info['intervention_original_size']}")
    
    # ======================
    # SAVE GLOBAL TEST SET AND REMAINING POOL (FOLLOWING CUSTOM SCM MODEL)
    # ======================
    
    if regenerate_test_assets:
        print(f"\n Saving global test set and remaining pool for reproducibility...")
        save_global_test_set(global_test_set, base_output_dir, column_names)
        save_remaining_pool(remaining_pool_info, base_output_dir, column_names)
        print(f" Dataset files saved following Custom SCM model")
        print(f"   📁 Global test set: datasets/global_test_set.npz")
        print(f"   📁 Remaining pool: datasets/remaining_pool.npz")
    else:
        print(f"\n📁 Using cached global test set and remaining pool from datasets/")

    # Prepare cache for reusable training splits loaded from disk or generated on-the-fly
    train_split_cache: dict[tuple[int, int], np.ndarray] = {}
    train_dataset_path_cache: dict[tuple[int, int], str] = {}

    datasets_dir = base_output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    print("\n Preparing training splits from remaining pool...")
    splits_loaded_from_disk = 0
    splits_regenerated = 0

    for train_size in TRAIN_SIZES:
        seed_list = train_size_to_seeds[train_size]
        for seed in seed_list:
            split_key = (seed, train_size)
            dataset_file = datasets_dir / f"train_ts{train_size}_s{seed}.npz"
            loaded_successfully = False

            if _REUSE_CACHED_TRAIN_SPLITS and dataset_file.exists():
                try:
                    with np.load(str(dataset_file), allow_pickle=True) as data:
                        metadata_obj = None
                        if "metadata" in data:
                            metadata_raw = data["metadata"]
                            if isinstance(metadata_raw, np.ndarray) and metadata_raw.shape == ():
                                metadata_obj = metadata_raw.item()
                            else:
                                metadata_obj = metadata_raw
                        if (
                            isinstance(metadata_obj, dict)
                            and metadata_obj.get("sampling_strategy") == "independent_branch"
                        ):
                            array = data["X_train"]
                            cols = data["column_names"].tolist()
                            if list(cols) == column_names and array.shape[1] == len(column_names):
                                train_split_cache[split_key] = array.copy()
                                train_dataset_path_cache[split_key] = str(dataset_file)
                                loaded_successfully = True
                        else:
                            print(
                                f"  Ignoring stored train split train_size={train_size}, seed={seed} due to invalid metadata."
                            )
                except Exception as exc:
                    print(
                        f"  Failed to load cached train split train_size={train_size}, seed={seed}: {exc}"
                    )

            if loaded_successfully:
                splits_loaded_from_disk += 1
                continue

            generated_split = generate_interventional_train_sets(
                remaining_pool_info=remaining_pool_info,
                intervention_info=intervention_info,
                intervention_mappings=intervention_mappings,
                train_sizes=[train_size],
                seed=seed,
            )
            train_data = generated_split.get(train_size, np.empty((0, len(column_names))))
            train_split_cache[split_key] = train_data.copy()

            if save_datasets:
                path = save_train_set_per_split(
                    X_train=train_data,
                    train_size=train_size,
                    seed=seed,
                    output_dir=base_output_dir,
                    column_names=column_names,
                    overwrite=True,
                    metadata={
                        "sampling_strategy": "independent_branch",
                        "seed": seed,
                    },
                )
                train_dataset_path_cache[split_key] = path
            elif dataset_file.exists():
                train_dataset_path_cache[split_key] = str(dataset_file)
            else:
                train_dataset_path_cache[split_key] = ""

            splits_regenerated += 1

    if splits_loaded_from_disk:
        print(f"    Cached splits loaded from disk for {splits_loaded_from_disk} seed/train-size combos")
    if splits_regenerated:
        print(f"    Regenerated fresh splits for {splits_regenerated} seed/train-size combos")

    # ======================
    # EXPERIMENTAL CONFIGURATIONS  
    # ======================

    # Use CSuite DAG structure for experimental configurations
    # Don't use get_experimental_configs - create configurations inline like in CSuite comparison
    from causal_experiments.utils.dag_utils import get_ordering_strategies, dag_to_ideal_cpdag
    
    # Get orderings based on CSuite DAG
    orderings_dict = get_ordering_strategies(dag_dict)

    # Optional dataset-specific overrides for column permutations.
    custom_vanilla_orderings: dict[str, list[int]] | None = None

    # If 'original' equals 'topological' or 'worst', define a deterministic random
    # ordering to be used instead across all configurations in this dataset.
    def deterministic_random_ordering(n_features: int, dataset_key: str) -> list[int]:
        import hashlib
        salt = "random_original_v1"
        key = f"{dataset_key}:{salt}"
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        seed = int(h[:16], 16) % (2**32)
        rng = np.random.default_rng(seed)
        return list(rng.permutation(n_features))

    # Hard-code column permutations for datasets with multiple valid topological
    # sorts to avoid ambiguity in downstream comparisons.
    if dataset_name.lower() == "csuite_symprod_simpson":
        name_to_index = {name: idx for idx, name in enumerate(column_names)}

        def _names_to_indices(names: list[str]) -> list[int]:
            resolved: list[int] = []
            for name in names:
                canonical = name if name in name_to_index else f"{name}_0"
                if canonical not in name_to_index:
                    raise ValueError(
                        f"Column '{name}' not found in dataset '{dataset_name}' while setting custom orderings."
                    )
                resolved.append(name_to_index[canonical])
            return resolved

        forced_orders = {
            "topological": ["x0_0", "x1_0", "x2_0", "x3_0"],
            "reverse_topological": ["x3_0", "x2_0", "x1_0", "x0_0"],
            "original": ["x1_0", "x3_0", "x0_0", "x2_0"],
        }

        custom_vanilla_orderings = {
            ordering_name: _names_to_indices(ordering_columns)
            for ordering_name, ordering_columns in forced_orders.items()
        }

        for ordering_name, ordering_indices in custom_vanilla_orderings.items():
            orderings_dict[ordering_name] = ordering_indices

    random_original_ordering: list[int] | None = None
    _orig = orderings_dict.get("original")
    _topo = orderings_dict.get("topological")
    _reverse_topo = orderings_dict.get("reverse_topological")
    if _orig is not None and (_orig == _topo or _orig == _reverse_topo):
        random_original_ordering = deterministic_random_ordering(len(_orig), dataset_name)
        if random_original_ordering == _topo or random_original_ordering == _reverse_topo:
            random_original_ordering = random_original_ordering[1:] + random_original_ordering[:1]
        print("  Original ordering equals topological/reverse_topological. Using deterministic random ordering for 'original'.")
        print(f"   Random original ordering: {random_original_ordering}")
    
    # Filter orderings when a specific column order is requested
    if column_order_filter is not None:
        if column_order_filter not in orderings_dict:
            raise ValueError(f"Invalid column order '{column_order_filter}'. Available: {list(orderings_dict.keys())}")
        if custom_vanilla_orderings and column_order_filter in custom_vanilla_orderings:
            selected_ordering = custom_vanilla_orderings[column_order_filter]
        else:
            selected_ordering = orderings_dict[column_order_filter]
        vanilla_orderings = [(column_order_filter, selected_ordering)]
        print(f" Using specific column ordering for vanilla: {column_order_filter}")
    else:
        # Use all orderings, but if we have a deterministic random replacement for 'original', use it
        vanilla_orderings = []
        for name, indices in orderings_dict.items():
            if custom_vanilla_orderings and name in custom_vanilla_orderings:
                vanilla_orderings.append((name, custom_vanilla_orderings[name]))
            elif name == "original" and random_original_ordering is not None:
                vanilla_orderings.append((name, random_original_ordering))
            else:
                vanilla_orderings.append((name, indices))
        print(f" Using all column orderings for vanilla: {list(orderings_dict.keys())}")
    
    # Generate ideal CPDAG from CSuite DAG
    ideal_cpdag = dag_to_ideal_cpdag(dag_dict)
    
    # Create configurations based on CSuite structure (not hardcoded 4-variable SCM)
    configs = {
        "vanilla": {
            "orderings": vanilla_orderings,
            "dag": None,
            "cpdag": None,
            "dag_for_ordering": dag_dict  # Used for vanilla reordering
        },
        "dag": {
            "orderings": [
                ("topological", orderings_dict["topological"])
            ],
            "dag": dag_dict,
            "cpdag": None,
            "dag_for_ordering": dag_dict
        },
        # CPDAG v1 both-vanilla (discovered)
        "cpdag_discovered": {
            "orderings": [
                ("original", orderings_dict["original"])
            ],
            "dag": dag_dict,  # True DAG for discovery validation
            "cpdag": None,  # Will be discovered per training set
            "dag_for_ordering": dag_dict,
        },
        # CPDAG minimal
        "cpdag_minimal": {
            "orderings": [
                ("original", orderings_dict["original"])
            ],
            "dag": dag_dict,
            "cpdag": ideal_cpdag,
            "dag_for_ordering": dag_dict,
        },
        # v3 removed
    }
    
    print(f" Dataset structure: {len(column_names)} features, DAG: {dag_dict}")
    print(f"🔢 Generated {len(orderings_dict)} column orderings: {list(orderings_dict.keys())}")
    
    # CSuite datasets can have categorical variables
    USE_CATEGORICAL = len(categorical_cols) > 0

    # If a specific column order is requested together with a specific algorithm, filter that algorithm's orderings
    if column_order_filter is not None and 'configs' in locals():
        if algorithm_filter and algorithm_filter in configs:
            if column_order_filter not in orderings_dict:
                raise ValueError(f"Invalid column order '{column_order_filter}'. Available: {list(orderings_dict.keys())}")
            filtered = [ot for ot in configs[algorithm_filter]["orderings"] if ot[0] == column_order_filter]
            if not filtered:
                raise ValueError(f"Column order '{column_order_filter}' not found for algorithm '{algorithm_filter}'")
            configs[algorithm_filter]["orderings"] = filtered
            print(f" Running ONLY {algorithm_filter} column order: {column_order_filter}")

    # SAVE COMPLETE CONFIGURATIONS for crashed seed replacement
    # This ensures that crashed seeds are replaced for ALL algorithms, not just filtered ones

    # Filter algorithms if specified
    if algorithm_filter:
        if algorithm_filter not in configs:
            available_algorithms = list(configs.keys())
            raise ValueError(f"Algorithm '{algorithm_filter}' not found. Available: {available_algorithms}")
        configs = {algorithm_filter: configs[algorithm_filter]}
        print(f" Running ONLY algorithm: {algorithm_filter}")
    else:
        print(f" Running ALL algorithms: {list(configs.keys())}")

    # Validate configuration structure
    for config_details in configs.values():
        orderings = config_details["orderings"]
        assert isinstance(orderings, list), "Orderings must be a list"

    # ======================
    # VANILLA CONFIGURATION CACHE (OPTIMIZATION)
    # ======================

    # Identify duplicate vanilla configurations
    vanilla_duplicates = identify_vanilla_duplicates(configs)

    # Calculate total configurations dynamically AFTER filtering
    total_configs = sum(len(config["orderings"]) for config in configs.values())
    total_experiments = len(TRAIN_SIZES) * N_REPETITIONS * total_configs

    # ======================
    # LOAD EXISTING RESULTS IF ANY
    # ======================
    
    results: list[dict[str, Any]] = []
    completed_experiments: set[tuple] = set()
    
    # Resume support: scan all CSVs in dataset results dir (covers any older unique naming)
    loaded_rows = 0
    for csv_path in sorted(base_output_dir.glob("*.csv")):
        try:
            existing_df = pd.read_csv(csv_path)
        except Exception:
            continue
        existing_rows = cast(list[dict[str, Any]], [dict(row) for row in existing_df.to_dict('records')])
        for row in existing_rows:
            # Respect filters and ensure train_size is in our target set
            if algorithm_filter and row.get('algorithm') != algorithm_filter:
                continue
            if column_order_filter and row.get('column_order') != column_order_filter:
                continue
            if row.get('train_size') not in TRAIN_SIZES:
                continue
            exp_key = (row.get('algorithm'), row.get('column_order'), row.get('train_size'), row.get('seed'))
            if exp_key in completed_experiments:
                continue
            completed_experiments.add(exp_key)
            results.append(row)
        loaded_rows += len(existing_rows)
    if loaded_rows > 0:
        print(f" Resume: scanned {loaded_rows} rows from existing CSVs; {len(completed_experiments)} unique experiments will be skipped.")
    
    # ======================
    # MAIN EXPERIMENTAL LOOP
    # ======================

    start_time = time.time()

    experiment_counter = 0 + len(completed_experiments)
    crashed_seeds = set()  # Track seeds that trigger infinity bug
    crashed_seeds_data = []  # Track detailed crash information for CSV
    
    # Recovery: scan all CSVs for crashed seeds
    for csv_path in sorted(base_output_dir.glob("*.csv")):
        try:
            existing_df = pd.read_csv(csv_path)
        except Exception:
            continue
        for _, row in existing_df.iterrows():
            metric_cols = [
                'ate_test', 'ate_synthetic', 'ate_difference', 'ate_relative_error',
                'n_synthetic_samples', 'n_test_samples'
            ]
            if any(row.get(col, 0) == -1 for col in metric_cols):
                crashed_seeds.add(row.get('seed'))
    if crashed_seeds:
        print(f"Recovered {len(crashed_seeds)} crashed seeds from existing results: {sorted(crashed_seeds)}")

    # Run experiments
    for algorithm, config in configs.items():
        orderings = config["orderings"]
        dag = config["dag"]
        cpdag = config["cpdag"]
        dag_for_ordering = config.get("dag_for_ordering")

        for column_order_tuple in orderings:
            # Extract column order name from tuple
            column_order, _ = column_order_tuple
            
            for train_size in TRAIN_SIZES:
                seed_sequence = train_size_to_seeds[train_size]

                for rep_idx, seed in enumerate(seed_sequence):
                    repetition = rep_idx + 1  # 1, 2, 3, ..., N_REPETITIONS
                    
                    # Skip if already completed
                    exp_key = (algorithm, column_order, train_size, seed)
                    if exp_key in completed_experiments:
                        continue
                    
                    # Skip vanilla duplicates by copying results from original
                    copied_result = copy_vanilla_duplicate_result(
                        results, algorithm, column_order, train_size, seed, vanilla_duplicates
                    )
                    if copied_result is not None:
                        results.append(copied_result)
                        continue
                        
                    experiment_counter += 1

                    # Set seeds for this specific experiment
                    set_experiment_seeds(seed, include_cuda=True, verbose=False)

                    # Improved logging with current configuration details
                    print(f" [{experiment_counter}/{total_experiments}] {algorithm}-{column_order} | train_size={train_size} | rep={repetition}/{N_REPETITIONS} | seed={seed}")

                    # Print progress every 50 experiments
                    if experiment_counter % 50 == 0 and experiment_counter >= 10:
                        elapsed = time.time() - start_time
                        progress_pct = experiment_counter / total_experiments * 100
                        eta_seconds = (elapsed / experiment_counter) * (total_experiments - experiment_counter)

                        print(f" Progress: {experiment_counter}/{total_experiments} ({progress_pct:.1f}%), ETA: {eta_seconds/60:.1f} min")

                    try:
                        split_key = (seed, train_size)
                        cached_train = train_split_cache.get(split_key)
                        if cached_train is None:
                            raise ValueError(
                                f"Missing cached training split for seed={seed}, train_size={train_size}."
                            )
                        train_data = cached_train.copy()
                        test_data = global_test_set
                        train_set_path = train_dataset_path_cache.get(split_key, "")

                        # Sample training data from the interventional data
                        X_train_original = train_data

                        # For CPDAG algorithms, decide source (discovered vs minimal) robustly
                        if algorithm.startswith("cpdag_"):
                            if "discovered" in algorithm:
                                print(f"  → Discovering CPDAG for seed={seed}, train_size={train_size}")
                                cpdag_to_use = discover_cpdag_from_data(
                                    X_train_original,
                                    column_names,
                                    categorical_cols,
                                    USE_CATEGORICAL,
                                    true_dag=dag,
                                    alpha=alpha,
                                    indep_test=cpdag_indep_test,
                                    hybrid_params={
                                        "k": 5,
                                        "permutations": 500,
                                        "random_state": seed,
                                    } if cpdag_indep_test == "hybrid" else None,
                                )
                            elif "minimal" in algorithm:
                                cpdag_to_use = cpdag
                            else:
                                cpdag_to_use = cpdag
                        else:
                            cpdag_to_use = None

                        # Prepare data using algorithm-specific functions
                        if algorithm == "vanilla":
                            if dag_for_ordering is None:
                                raise ValueError(f"dag_for_ordering is required for vanilla algorithm but got None")

                            categorical_indices = [column_names.index(col) for col in categorical_cols] if categorical_cols else None

                            # Dataset-specific overrides take precedence
                            if custom_vanilla_orderings and column_order in custom_vanilla_orderings:
                                column_ordering_used = custom_vanilla_orderings[column_order]
                                X_train_prepared = X_train_original[:, column_ordering_used]
                                updated_categorical_features = None
                                if categorical_indices is not None:
                                    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(column_ordering_used)}
                                    updated_categorical_features = [old_to_new[i] for i in categorical_indices if i in old_to_new]
                            # If 'original' duplicates others, replace with deterministic random ordering
                            elif column_order == "original" and random_original_ordering is not None:
                                column_ordering_used = random_original_ordering
                                X_train_prepared = X_train_original[:, column_ordering_used]
                                updated_categorical_features = None
                                if categorical_indices is not None:
                                    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(column_ordering_used)}
                                    updated_categorical_features = [old_to_new[i] for i in categorical_indices if i in old_to_new]
                            else:
                                X_train_prepared, column_ordering_used, updated_categorical_features = prepare_vanilla_data(
                                    X_train_original, column_order, dag_for_ordering, column_names, categorical_indices
                                )

                            dag_prepared, cpdag_prepared = None, None

                        elif algorithm == "dag":
                                if dag is None:
                                    raise ValueError(f"DAG is required for dag algorithm but got None")
                                # Optional column reordering for DAG
                                if (column_order != "original") or (column_order == "original" and random_original_ordering is not None):
                                    if dag_dict is None:
                                        raise ValueError("dag_for_ordering (CSuite DAG) is required for DAG reordering but got None")
                                    categorical_indices = [column_names.index(col) for col in categorical_cols] if categorical_cols else None
                                    if column_order == "original" and random_original_ordering is not None:
                                        column_ordering_used = random_original_ordering
                                        X_train_reordered = X_train_original[:, column_ordering_used]
                                        updated_categorical_features = None
                                        if categorical_indices is not None:
                                            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(column_ordering_used)}
                                            updated_categorical_features = [old_to_new[i] for i in categorical_indices if i in old_to_new]
                                    else:
                                        X_train_reordered, column_ordering_used, updated_categorical_features = prepare_vanilla_data(
                                            X_train_original, column_order, dag_dict, column_names, categorical_indices
                                        )
                                    # Remap DAG indices to the new column order
                                    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(column_ordering_used)}
                                    dag_reordered = {}
                                    for old_child, old_parents in dag.items():
                                        new_child = old_to_new.get(old_child)
                                        if new_child is None:
                                            continue
                                        dag_reordered[new_child] = [old_to_new[p] for p in old_parents if p in old_to_new]
                                    X_train_prepared, dag_prepared = prepare_dag_data(
                                        X_train_reordered, dag_reordered, [column_names[i] for i in column_ordering_used]
                                    )
                                else:
                                    X_train_prepared, dag_prepared = prepare_dag_data(
                                        X_train_original, dag, column_names
                                    )
                                    column_ordering_used = None
                                    updated_categorical_features = None
                                cpdag_prepared = None

                        elif algorithm.startswith("cpdag_"):
                                if cpdag_to_use is None:
                                    raise ValueError(f"CPDAG is required for {algorithm} algorithm but got None")

                                # Handle column reordering for CPDAG (like vanilla does).
                                # For datasets with custom "original" order (e.g. csuite_symprod_simpson),
                                # apply that explicit permutation also in CPDAG modes.
                                should_reorder_cpdag = (
                                    (column_order != "original")
                                    or (column_order == "original" and random_original_ordering is not None)
                                    or (
                                        custom_vanilla_orderings is not None
                                        and column_order in custom_vanilla_orderings
                                    )
                                )
                                if should_reorder_cpdag:
                                    # Use the same DAG as vanilla does
                                    dag_for_ordering = dag_dict
                                    categorical_indices = [column_names.index(col) for col in categorical_cols] if categorical_cols else None
                                    if custom_vanilla_orderings and column_order in custom_vanilla_orderings:
                                        column_ordering_used = custom_vanilla_orderings[column_order]
                                        X_train_reordered = X_train_original[:, column_ordering_used]
                                        updated_categorical_features = None
                                        if categorical_indices is not None:
                                            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(column_ordering_used)}
                                            updated_categorical_features = [old_to_new[i] for i in categorical_indices if i in old_to_new]
                                    elif column_order == "original" and random_original_ordering is not None:
                                        column_ordering_used = random_original_ordering
                                        X_train_reordered = X_train_original[:, column_ordering_used]
                                        updated_categorical_features = None
                                        if categorical_indices is not None:
                                            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(column_ordering_used)}
                                            updated_categorical_features = [old_to_new[i] for i in categorical_indices if i in old_to_new]
                                    else:
                                        X_train_reordered, column_ordering_used, updated_categorical_features = prepare_vanilla_data(
                                            X_train_original, column_order, dag_for_ordering, column_names, categorical_indices
                                        )

                                    # Reorder CPDAG structure to match new column ordering
                                    # Create mapping from old indices to new indices
                                    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(column_ordering_used)}

                                    # Reorder CPDAG structure
                                    cpdag_reordered = {}
                                    for old_idx in range(len(column_names)):
                                        if old_idx in cpdag_to_use:
                                            new_idx = old_to_new[old_idx]
                                            cpdag_reordered[new_idx] = {
                                                "parents": [old_to_new[p] for p in cpdag_to_use[old_idx]["parents"] if p in old_to_new],
                                                "undirected": [old_to_new[u] for u in cpdag_to_use[old_idx]["undirected"] if u in old_to_new]
                                            }
                                        else:
                                            # Create empty entry for missing nodes
                                            new_idx = old_to_new[old_idx]
                                            cpdag_reordered[new_idx] = {"parents": [], "undirected": []}

                                    X_train_prepared, cpdag_prepared = prepare_cpdag_data(
                                        X_train_reordered, cpdag_reordered, [column_names[i] for i in column_ordering_used]
                                    )
                                else:
                                    # Original ordering - no changes needed
                                    X_train_prepared, cpdag_prepared = prepare_cpdag_data(
                                        X_train_original, cpdag_to_use, column_names
                                    )
                                    column_ordering_used = None
                                    updated_categorical_features = None

                                dag_prepared = None

                        else:
                            raise ValueError(f"Unknown algorithm: {algorithm}")

                        # Create test DataFrame from fixed global test set
                        test_df = pd.DataFrame({
                            **dict(zip(column_names, [test_data[:, i] for i in range(test_data.shape[1])])),
                        })

                        # Determine if causal_structures_last should be used
                        causal_structures_last = config.get("causal_structures_last", False)

                        # Run the experiment
                        metrics, synthetic_df = run_single_experiment(
                            X_train=X_train_prepared,
                            test_df=test_df,
                            algorithm=algorithm,
                            column_order=column_order,
                            dag=dag_prepared,
                            cpdag=cpdag_prepared,
                            column_names=column_names,
                            categorical_cols=categorical_cols,
                            intervention_info=intervention_info,
                            intervention_mappings=intervention_mappings,
                            column_ordering_used=column_ordering_used,
                            updated_categorical_features=updated_categorical_features,
                            n_permutations=N_PERMUTATIONS,
                            temp=TEMPERATURE,
                            n_estimators=N_ESTIMATORS,
                            seed=seed,
                            causal_structures_last=causal_structures_last,
                        )

                        # Determine which graph to use for result row
                        if algorithm == "vanilla":
                            graph_dict = None  # Vanilla never has graph structure in results
                            reordered_graph_dict = None
                        elif algorithm.startswith("cpdag_"):
                            graph_dict = cpdag_to_use
                            reordered_graph_dict = cpdag_prepared if cpdag_prepared is not None else None
                        else:  # algorithm == "dag"
                            graph_dict = dag
                            reordered_graph_dict = dag_prepared if dag_prepared is not None else None

                        # Create result row
                        result_row = create_result_row(
                            algorithm=algorithm,
                            column_order=column_order,
                            graph_dict=graph_dict,
                            train_size=train_size,
                            seed=seed,
                            repetition=repetition,
                            categorical_cols=categorical_cols,
                            column_names=column_names,
                            metrics_results=metrics,
                            reordered_graph_dict=reordered_graph_dict,
                            column_ordering_used=column_ordering_used
                        )

                        # Get consolidated output file for current configuration
                        _, current_output_file = get_output_paths(None, algorithm_filter, column_order_filter)

                        # Save training set to common datasets directory (Custom SCM model)
                        if save_datasets:
                            if not train_set_path:
                                train_set_path = save_train_set_per_split(
                                    X_train=train_data,
                                    train_size=train_size,
                                    seed=seed,
                                    output_dir=base_output_dir,
                                    column_names=column_names
                                )
                                train_dataset_path_cache[split_key] = train_set_path
                            if train_set_path:
                                result_row['train_dataset_path'] = str(Path("datasets") / Path(train_set_path).name)
                                result_row['test_dataset_path'] = 'datasets/global_test_set.npz'
                                if test_mode:
                                    csv_path = Path(train_set_path).with_suffix('.csv')
                                    pd.DataFrame(train_data, columns=column_names).to_csv(csv_path, index=False)
                                    print(f"[TEST] Training CSV saved: {csv_path}")
                        else:
                            if train_set_path:
                                result_row['train_dataset_path'] = str(Path("datasets") / Path(train_set_path).name)
                                result_row['test_dataset_path'] = 'datasets/global_test_set.npz'

                        # Save synthetic data to common datasets directory (Custom SCM model)
                        if save_synthetic:
                            # Pass base_output_dir - save_synthetic_data will add /datasets/synthetic/ automatically
                            synthetic_path = save_synthetic_data(
                                synthetic_df=synthetic_df,
                                algorithm=algorithm,
                                column_order=column_order,
                                train_size=train_size,
                                seed=seed,
                                output_dir=base_output_dir,  # Base dir - function will add /datasets/synthetic/
                                n_permutations=N_PERMUTATIONS,
                                temperature=TEMPERATURE,
                                metrics=metrics
                            )
                            result_row['synthetic_data_path'] = str(Path("datasets/synthetic") / Path(synthetic_path).name)  # Relative path
                            if test_mode:
                                csv_path = Path(synthetic_path).with_suffix('.csv')
                                synthetic_df.to_csv(csv_path, index=False)
                                print(f"[TEST] Synthetic CSV saved: {csv_path}")

                        # Get actual column order based on whether reordering was applied
                        if column_ordering_used is not None and (
                            algorithm == "vanilla" or algorithm == "dag" or algorithm.startswith("cpdag_")
                        ):
                            # For algorithms with reordering, use the ordering from preparation
                            actual_column_order = [column_names[i] for i in column_ordering_used]
                        else:
                            # For DAG or algorithms without reordering, columns stay in original order
                            actual_column_order = column_names.copy()

                        # Add actual column order used for verification
                        result_row['actual_column_order'] = ','.join(actual_column_order)

                        # Add dataset information
                        result_row['dataset_name'] = dataset_name
                        result_row['intervention_variable'] = intervention_info['variable']
                        result_row['target_variable'] = intervention_info['target']
                        result_row['intervention_values'] = ','.join(map(str, intervention_info['values']))

                        results.append(result_row)

                        # Save intermediate results every N experiments
                        if len(results) % save_every == 0:
                            # Persist all accumulated rows to the consolidated file
                            save_results_to_csv(results, str(current_output_file))

                            # Also save crashed seeds data if any exist
                            if crashed_seeds_data:
                                algorithm_suffix = f"_{algorithm_filter}" if algorithm_filter else "_all_algorithms"
                                crashed_csv_file = current_output_file.parent / f"crashed_seeds_detailed{algorithm_suffix}.csv"
                                save_crashed_seeds_to_csv(crashed_seeds_data, crashed_csv_file)

                    except Exception as e:
                        error_msg = str(e)
                        if "infinity" in error_msg.lower() or "too large for dtype" in error_msg.lower():
                            print(f" INFINITY BUG: {algorithm}-{column_order} seed={seed} | SKIPPING")
                            crashed_seeds.add(seed)
                            # Record detailed crash information
                            crashed_seeds_data.append({
                                'seed': seed,
                                'algorithm': algorithm,
                                'column_order': column_order,
                                'train_size': train_size,
                                'repetition': repetition,
                                'dataset_name': dataset_name,
                                'error_type': 'infinity_bug',
                                'error_message': error_msg[:200],  # Truncate long error messages
                                'timestamp': pd.Timestamp.now().isoformat()
                            })
                        else:
                            print(f" OTHER ERROR in experiment {algorithm}-{column_order} seed={seed}: {e}")
                            # Record other errors too
                            crashed_seeds_data.append({
                                'seed': seed,
                                'algorithm': algorithm,
                                'column_order': column_order,
                                'train_size': train_size,
                                'repetition': repetition,
                                'dataset_name': dataset_name,
                                'error_type': 'other_error',
                                'error_message': error_msg[:200],  # Truncate long error messages
                                'timestamp': pd.Timestamp.now().isoformat()
                            })
                        continue
                    
                    # Periodic memory cleanup
                    finally:
                        if experiment_counter % 10 == 0:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            import gc
                            gc.collect()

    # ======================
    # SAVE FINAL RESULTS
    # ======================

    total_time = time.time() - start_time
    print(f"\n🎉 Experiment completed in {total_time/60:.1f} minutes")

    _, output_file = get_output_paths(None, algorithm_filter, column_order_filter)

    if not results:
        print("  WARNING: No experiments produced results! Creating placeholder CSV.")
        empty_result = {
            'algorithm': 'none', 'column_order': 'none', 'train_size': TRAIN_SIZES[0] if TRAIN_SIZES else 0,
            'seed': 0, 'repetition': 0, 'ate_test': 0, 'ate_synthetic': 0,
            'ate_difference': 0, 'ate_relative_error': 0, 'dataset_name': dataset_name
        }
        results_to_save = [empty_result]
    else:
        results_to_save = results

    save_results_to_csv(results_to_save, str(output_file))
    print(f" Final results saved to: {output_file}")

    # Warn if any train size produced no results
    missing_train_sizes = [ts for ts in TRAIN_SIZES if not any(r.get('train_size') == ts for r in results)]
    for missing_ts in missing_train_sizes:
        print(f"  WARNING: No results produced for train size {missing_ts}! All experiments may have crashed.")
    
    # Report crashed seeds summary
    print("\n" + "=" * 60)
    print(" CRASHED SEEDS SUMMARY")
    print("=" * 60)
    
    if crashed_seeds:
        crashed_list = sorted(list(crashed_seeds))
        print(f"Total crashed seeds: {len(crashed_list)}")
        print(f"Crashed seeds: {crashed_list}")
        print("\n NOTE: Experiments with these seeds returned -1.0 metrics due to TabPFN infinity bug.")
        
        # Save detailed crashed seeds data to CSV
        if crashed_seeds_data:
            algorithm_suffix = f"_{algorithm_filter}" if algorithm_filter else "_all_algorithms"
            crashed_csv_file = Path("causal_experiments/csuite_experiment/intervention_experiment_csuite/results") / f"crashed_seeds_detailed{algorithm_suffix}.csv"
            save_crashed_seeds_to_csv(crashed_seeds_data, crashed_csv_file)
    else:
        print(" No crashed seeds detected! All experiments completed successfully.")
    
    print("=" * 60)

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CSuite Interventional TabPFN Experiment")
    parser.add_argument("--test", action="store_true", help="Run in test mode (faster, fewer experiments)")
    parser.add_argument("--save-every", type=int, default=100, help="Save results every N experiments (default: 100)")
    parser.add_argument("--save-datasets", action="store_true", help="Save train/test datasets to .npz files for reproducibility")
    parser.add_argument("--save-reordered", action="store_true", help="Save reordered datasets for verification (requires --save-datasets)")
    parser.add_argument("--save-synthetic", action="store_true", help="Save synthetic data for analysis and debugging")
    parser.add_argument(
        "--algorithm",
        help=(
            "Run only specific algorithm: vanilla, dag, "
            "cpdag_discovered, cpdag_minimal."
        ),
    )
    parser.add_argument("--dataset", default="csuite_lingauss", help="CSuite dataset to use (default: csuite_lingauss)")
    parser.add_argument(
        "--column-order",
        help=(
            "For vanilla, DAG, and all CPDAG variants, specify column ordering: original, topological, worst"
        ),
    )
    parser.add_argument(
        "--repetitions", type=int, default=None,
        help="Number of repetitions per (algorithm, order, train_size). Default 150 (3 in --test) if not provided."
    )
    parser.add_argument(
        "--skip-seeds",
        type=int,
        nargs='+',
        default=None,
        help="Seed values to skip; replacements are sampled so the total number of repetitions stays unchanged."
    )
    parser.add_argument(
        "--train-sizes-group", choices=["small", "large", "all"], default=None,
        help="Optionally restrict train sizes: small=[20,50,100], large=[200,500,1000]."
    )
    args = parser.parse_args()

    # Set up environment for reproducibility
    setup_determinism(
        enable_cuda_determinism=True,
        cublas_workspace_config=":4096:8",
        set_num_threads=1,
        verbose=True
    )

    # Run the experiment
    main(
        test_mode=args.test, 
        save_every=args.save_every,
        save_datasets=args.save_datasets,
        save_reordered=args.save_reordered,
        save_synthetic=args.save_synthetic,
        algorithm_filter=args.algorithm,
        dataset_name=args.dataset,
        column_order_filter=args.column_order,
        repetitions=args.repetitions,
        train_sizes_group=args.train_sizes_group,
        skip_seeds=args.skip_seeds,
    )
