
"""Interventional experiment for evaluating causal TabPFN configurations.

This script implements an interventional experiment to test TabPFN's ability
to preserve Average Treatment Effect (ATE) in synthetic data generation.

TabPFN Configurations (aligned with comparison_experiment):
- vanilla: 3 orderings (original, topological, reverse_topological)
- dag: DAG ground truth
- cpdag_discovered, cpdag_minimal

Usage Examples:

    # Run ALL algorithms (default)
    uv run causal_experiments/custom_scm_experiment/interventional_experiment/interventional_experiment.py
    
    # Run in test mode (faster)
    uv run causal_experiments/custom_scm_experiment/interventional_experiment/interventional_experiment.py --test
    
    # Run specific algorithm
    uv run causal_experiments/custom_scm_experiment/interventional_experiment/interventional_experiment.py --algorithm vanilla
    
"""
from __future__ import annotations

import os
# Set environment for maximal determinism before importing torch
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import time  # Used for timing experiments and generating timestamps
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from typing import Any, cast, List

# Fix imports - add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def to_workspace_relative(path: str | Path) -> str:
    """Return a repository-relative string for paths stored in CSV outputs."""
    if not path:
        return ""
    try:
        resolved = Path(path).resolve()
    except (TypeError, OSError):
        return str(path)
    try:
        return str(resolved.relative_to(project_root))
    except ValueError:
        return str(resolved)

from causal_experiments.utils import (
    create_result_row,
    discover_cpdag_from_data,
    get_experimental_configs,
    identify_vanilla_duplicates,
    copy_vanilla_duplicate_result,
    prepare_vanilla_data,
    prepare_dag_data,
    prepare_cpdag_data,
    save_global_test_set,
    save_train_set_per_split,
    save_reordered_dataset,
    save_synthetic_data,
    save_results_to_csv,
    # Loading functions
    load_global_test_set,
    # Determinism utilities
    setup_determinism,
    set_experiment_seeds,
)

def calculate_ate(data: np.ndarray, column_names: List[str], treatment_col: int = 3, outcome_col: int = 1) -> float:
    """Calculate Average Treatment Effect.
    
    ATE = E[Y|do(X3=1)] - E[Y|do(X3=0)]
    
    Args:
        data: Data array 
        column_names: List of column names (e.g., ['X0', 'X1', 'X2', 'X3'])
        treatment_col: Default column index for treatment (X3) - only used if no reordering
        outcome_col: Default column index for outcome (X1) - only used if no reordering
    
    Returns:
        ATE value
    """
    # Find correct column indices based on column names
    try:
        treatment_idx = column_names.index('X3')  # Treatment is always X3
        outcome_idx = column_names.index('X1')    # Outcome is always X1 (collider)
    except ValueError:
        # Fallback to provided indices if column names don't match expected
        treatment_idx = treatment_col
        outcome_idx = outcome_col
    
    # Get unique treatment values
    treatment_values = np.unique(data[:, treatment_idx])
    
    if len(treatment_values) < 2:
        # All samples have same treatment value - can't calculate ATE
        return 0.0
    
    # Calculate mean outcome for each treatment level
    outcomes_by_treatment = {}
    for treatment_val in treatment_values:
        mask = data[:, treatment_idx] == treatment_val
        if np.any(mask):
            outcomes_by_treatment[treatment_val] = np.mean(data[mask, outcome_idx])
    
    # Calculate ATE: E[Y|X3=1] - E[Y|X3=0]
    if 1.0 in outcomes_by_treatment and 0.0 in outcomes_by_treatment:
        ate = outcomes_by_treatment[1.0] - outcomes_by_treatment[0.0]
    else:
        # Fallback: use highest - lowest if exactly 1 and 0 not available
        treatment_vals = sorted(outcomes_by_treatment.keys())
        ate = outcomes_by_treatment[treatment_vals[-1]] - outcomes_by_treatment[treatment_vals[0]]
    
    return float(ate)


def create_global_test_set_and_remaining_pool(
    *,
    intervention_info: dict,
    test_size: int,
    test_seed: int = 2000,
) -> tuple[np.ndarray, dict]:
    reference_data = intervention_info["reference_data"]
    intervention_data = intervention_info["intervention_data"]

    total_reference = len(reference_data)
    total_intervention = len(intervention_data)

    test_samples_per_branch = min(test_size // 2, total_reference, total_intervention)
    if test_samples_per_branch == 0:
        raise ValueError("Not enough samples per branch to build the test set")

    rng = np.random.default_rng(test_seed)

    ref_test_indices = rng.choice(total_reference, size=test_samples_per_branch, replace=False)
    int_test_indices = rng.choice(total_intervention, size=test_samples_per_branch, replace=False)

    reference_test = reference_data[ref_test_indices]
    intervention_test = intervention_data[int_test_indices]

    ref_remaining_mask = np.ones(total_reference, dtype=bool)
    ref_remaining_mask[ref_test_indices] = False
    reference_remaining = reference_data[ref_remaining_mask]

    int_remaining_mask = np.ones(total_intervention, dtype=bool)
    int_remaining_mask[int_test_indices] = False
    intervention_remaining = intervention_data[int_remaining_mask]

    global_test_set = np.concatenate([reference_test, intervention_test], axis=0)

    remaining_pool_info = {
        "reference_data": reference_remaining,
        "intervention_data": intervention_remaining,
        "reference_original_size": total_reference,
        "intervention_original_size": total_intervention,
        "reference_remaining_size": len(reference_remaining),
        "intervention_remaining_size": len(intervention_remaining),
    }

    print(f" Global test set and remaining pool created (seed={test_seed}):")
    print(f"   📈 Reference test samples: {len(reference_test)}")
    print(f"   📈 Intervention test samples: {len(intervention_test)}")
    print(f"   📈 Total test samples: {len(global_test_set)}")
    print(f"    Reference remaining for training: {len(reference_remaining)}")
    print(f"    Intervention remaining for training: {len(intervention_remaining)}")

    return global_test_set, remaining_pool_info


def save_remaining_pool(
    remaining_pool_info: dict,
    base_output_dir: Path,
    column_names: list[str],
) -> Path:
    datasets_dir = base_output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    remaining_pool_file = datasets_dir / "remaining_pool.npz"
    remaining_combined = np.concatenate(
        [remaining_pool_info["reference_data"], remaining_pool_info["intervention_data"]],
        axis=0,
    )

    np.savez_compressed(
        remaining_pool_file,
        data=remaining_combined,
        column_names=column_names,
        reference_size=remaining_pool_info["reference_remaining_size"],
        intervention_size=remaining_pool_info["intervention_remaining_size"],
        reference_original_size=remaining_pool_info["reference_original_size"],
        intervention_original_size=remaining_pool_info["intervention_original_size"],
    )

    print(f" Saved remaining pool: {remaining_pool_file} ({len(remaining_combined)} samples)")
    return remaining_pool_file


def load_remaining_pool(
    base_output_dir: Path,
    column_names: list[str],
) -> dict | None:
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
        print("  Ignoring stored remaining pool due to column mismatch; regenerating from source data.")
        return None

    if ref_size + int_size != combined.shape[0]:
        print("  Stored remaining pool has inconsistent sizes; regenerating from source data.")
        return None

    reference_data = combined[:ref_size]
    intervention_data = combined[ref_size:ref_size + int_size]

    return {
        "reference_data": reference_data,
        "intervention_data": intervention_data,
        "reference_original_size": ref_original,
        "intervention_original_size": int_original,
        "reference_remaining_size": ref_size,
        "intervention_remaining_size": int_size,
    }


def generate_interventional_train_sets(
    *,
    remaining_pool_info: dict,
    intervention_info: dict,
    intervention_mappings: dict,
    train_sizes: list[int],
    seed: int,
) -> dict[int, np.ndarray]:
    unique_sizes = sorted({int(ts) for ts in train_sizes if ts is not None})
    if not unique_sizes:
        return {}

    reference_remaining = remaining_pool_info["reference_data"]
    intervention_remaining = remaining_pool_info["intervention_data"]

    if len(reference_remaining) == 0 or len(intervention_remaining) == 0:
        raise ValueError("Remaining pool does not contain enough samples to generate training data")

    rng = np.random.default_rng(seed)
    intervention_idx = intervention_info["intervention_idx"]

    train_sets: dict[int, np.ndarray] = {}
    for train_size in unique_sizes:
        samples_per_branch = max(train_size // 2, 0)

        available_ref = len(reference_remaining)
        available_int = len(intervention_remaining)

        samples_per_branch = min(samples_per_branch, available_ref, available_int)

        if samples_per_branch == 0:
            print(
                f"  Unable to sample train_size={train_size} for seed={seed}: "
                "not enough data per branch."
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


# =============================

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised

# Global flag to print device info only once
_device_info_printed = False


def _load_custom_intervention_data(
    *,
    mode: str = "numeric",
) -> dict[str, Any]:
    """Load static custom SCM interventional data (X3 fixed to 0 or 1)."""

    dataset_kind = "mixed" if mode == "mixed" else "numeric"
    base_dir = Path(__file__).parent / "generated_interventional"
    branch0_path = base_dir / f"custom_{dataset_kind}_intervention_x3_eq_0.csv"
    branch1_path = base_dir / f"custom_{dataset_kind}_intervention_x3_eq_1.csv"

    if not branch0_path.exists() or not branch1_path.exists():
        raise FileNotFoundError(
            "Missing interventional dataset CSVs. Expected files: "
            f"{branch0_path} and {branch1_path}. Run generate_custom_interventional_dataset.py first."
        )

    df0 = pd.read_csv(branch0_path)
    column_names = list(df0.columns)
    df1 = pd.read_csv(branch1_path)[column_names]

    reference_data = df0.to_numpy(dtype=float)
    intervention_data = df1.to_numpy(dtype=float)

    intervention_var = "X3"
    target_var = "X1"

    if intervention_var not in column_names:
        raise ValueError(f"Intervention variable '{intervention_var}' not found in columns {column_names}")
    if target_var not in column_names:
        raise ValueError(f"Target variable '{target_var}' not found in columns {column_names}")

    intervention_idx = column_names.index(intervention_var)
    effect_idx = column_names.index(target_var)

    intervention_values = [float(reference_data[0, intervention_idx]), float(intervention_data[0, intervention_idx])]

    configs = get_experimental_configs(use_categorical=(mode == "mixed"))
    dag_dict = configs["dag"]["dag"]
    cpdag_dict = configs["cpdag_minimal"]["cpdag"]

    categorical_cols = []
    if mode == "mixed":
        categorical_cols = ["X4_cat"]

    intervention_info = {
        "variable": intervention_var,
        "target": target_var,
        "values": intervention_values,
        "intervention_idx": intervention_idx,
        "effect_idx": effect_idx,
        "reference_data": reference_data,
        "intervention_data": intervention_data,
    }

    return {
        "column_names": column_names,
        "categorical_columns": categorical_cols,
        "dag_dict": dag_dict,
        "cpdag_dict": cpdag_dict,
        "intervention_info": intervention_info,
    }


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


_REUSE_CACHED_TRAIN_SPLITS = True





def run_single_experiment(
    X_train: np.ndarray,
    test_df: pd.DataFrame,
    algorithm: str,
    column_order: str,
    dag: dict | None,
    cpdag: dict | None,
    column_names: List[str],
    categorical_cols: List[str],
    column_ordering_used: List[int] | None = None,
    updated_categorical_features: List[int] | None = None,
    n_permutations: int = 3,
    temp: float = 1.0,
    n_estimators: int = 3,
    seed: int | None = None,
    causal_structures_last: bool = False,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Run a single experimental configuration using GenerateSyntheticDataExperiment.

    Args:
        X_train: Training data (already prepared/processed)
        test_df: Test data as DataFrame for evaluation
        algorithm: Algorithm name
        column_order: Column ordering strategy
        dag: DAG structure (already prepared, if applicable)
        cpdag: CPDAG structure (already prepared, if applicable)
        column_names: List of column names
        categorical_cols: List of categorical column names
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
        
        # Set categorical features if any exist
        if categorical_cols:
            # Use updated categorical features if available (for vanilla with reordering)
            if updated_categorical_features is not None:
                categorical_indices = updated_categorical_features
            else:
                # Convert categorical column names to indices
                categorical_indices = [column_names.index(col) for col in categorical_cols]
            model_unsupervised.set_categorical_features(categorical_indices)
        
        # Convert to torch tensor (data is already prepared)
        X_tensor = torch.tensor(X_train, dtype=torch.float32)

        # Create and run the GenerateSyntheticDataExperiment
        exp_synthetic = unsupervised.experiments.GenerateSyntheticDataExperiment(
            task_type="unsupervised"
        )

        # Pass CPDAG when required
        cpdag_to_pass = cpdag if algorithm.startswith("cpdag_") else None

        # Determine feature names to reflect the CURRENT column order seen by the model
        if column_order != "original" and column_ordering_used is not None and (
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
            indices=list(range(X_train.shape[1])),  # Use actual number of features from prepared data
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
        if (algorithm == "vanilla" or algorithm == "dag" or algorithm.startswith("cpdag_")) and column_order != "original" and column_ordering_used is not None:
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
        
        # INTERVENTIONAL EXPERIMENT: Calculate ATE metrics instead of data fidelity
        
        # Convert test DataFrame to numpy for ATE calculation
        test_data_np = test_df[column_names].values
        
        # Convert synthetic DataFrame to numpy for ATE calculation  
        synthetic_data_np = synthetic_df[column_names].values
        
        # Calculate ATE on test data (ground truth)
        ate_test = calculate_ate(test_data_np, column_names)
        
        # Calculate ATE on synthetic data
        ate_synthetic = calculate_ate(synthetic_data_np, column_names)
        
        # Calculate ATE difference (primary metric)
        ate_difference = abs(ate_test - ate_synthetic)
        
        # Create metrics dictionary focused on ATE
        metrics = {
            "ate_test": ate_test,
            "ate_synthetic": ate_synthetic, 
            "ate_difference": ate_difference,
            "ate_relative_error": abs(ate_difference / ate_test) if ate_test != 0 else float('inf'),
            "n_synthetic_samples": len(synthetic_data_np),
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


def main(test_mode: bool = False, save_every: int = 100, save_datasets: bool = False, save_reordered: bool = False, save_synthetic: bool = False, algorithm_filter: str | None = None, column_order_filter: str | None = None, train_sizes_group: str | None = None):
    """Main experimental pipeline.
    
    Args:
        test_mode: If True, run with fewer experiments for testing
        save_every: Save CSV results every N experiments
        save_datasets: If True, save train/test datasets to .npz files
        save_reordered: If True, save reordered datasets for verification
        save_synthetic: If True, save synthetic data for analysis
    """
    # ======================
    # EXPERIMENTAL PARAMETERS
    # ======================

    # INTERVENTIONAL EXPERIMENT PARAMETERS
    if test_mode:
        TRAIN_SIZES = [50, 100]  # Multiple training sizes for testing
        TEST_SIZE = 200   # Remaining samples from interventional data
        N_REPETITIONS = 3
    else:
        TRAIN_SIZES = [20, 50, 100, 200, 500, 1000]  # Multiple training sizes for full experiment
        TEST_SIZE = 2000   # Remaining samples from interventional data
        N_REPETITIONS = 130

    # Optional: restrict to small/large train-size groups
    if train_sizes_group:
        group_small = [20, 50, 100]
        group_large = [200, 500, 1000]
        if train_sizes_group == "small":
            TRAIN_SIZES = [ts for ts in TRAIN_SIZES if ts in group_small]
        elif train_sizes_group == "large":
            TRAIN_SIZES = [ts for ts in TRAIN_SIZES if ts in group_large]
        elif train_sizes_group == "all":
            pass
        else:
            raise ValueError("train_sizes_group must be one of: small, large, all")
    
    # Total interventional data: max(TRAIN_SIZES) + TEST_SIZE (split 50/50 for X3=0/1)
    # TOTAL_INTERVENTIONAL_SIZE = max(TRAIN_SIZES) + TEST_SIZE  # Not used in this experiment

    # TabPFN parameters
    N_ESTIMATORS = 3
    N_PERMUTATIONS = 3
    TEMPERATURE = 1.0
    
    # Seed configuration - automatic seed management for fair comparison
    # Linear seed approach: 0,1,2,3,4,5... (same seed = same dataset for all algorithms = fair comparison)
    # No BASE_SEED needed - use simple linear counter like in comparison_experiment

    # Output configuration - consolidated CSV files (no per-train-size split)
    script_dir = Path(__file__).parent
    base_output_dir = script_dir / "results"
    base_output_dir.mkdir(parents=True, exist_ok=True)  # Create base directory
    
    # Function to get output directory and consolidated results file path
    def get_output_paths(train_size: int | None = None, algorithm_filter: str | None = None, column_order_filter: str | None = None) -> tuple[Path, Path]:
        """Return output directory and consolidated CSV path (independent of train size)."""
        out_dir = base_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build filename base
        base_name = "custom_scm_interventional_results"
        name_parts: list[str] = []
        if algorithm_filter:
            name_parts.append(algorithm_filter)
        if column_order_filter:
            name_parts.append(column_order_filter)

        suffix = "_" + "_".join(name_parts) if name_parts else ""
        output_file = out_dir / f"{base_name}{suffix}.csv"
        return out_dir, output_file


    # ======================
    # INTERVENTIONAL DATA GENERATION
    # ======================

    # For interventional experiment, default to numeric SCM (no categorical support)
    USE_CATEGORICAL = False
    dataset_info = _load_custom_intervention_data(mode="mixed" if USE_CATEGORICAL else "numeric")

    column_names = dataset_info["column_names"]
    categorical_cols = dataset_info["categorical_columns"]
    dag_dict_static = dataset_info["dag_dict"]
    cpdag_static = dataset_info["cpdag_dict"]
    intervention_info = dataset_info["intervention_info"]

    cpdag_indep_test = "hybrid" if USE_CATEGORICAL else "fisherz"
    print(f" Independence test for CPDAG discovery: {cpdag_indep_test}")


    # ======================
    # MODEL INITIALIZATION
    # ======================


    # Note: TabPFN models will be created fresh for each experiment to avoid state contamination

    # ======================
    # EXPERIMENTAL CONFIGURATIONS
    # ======================

    configs = get_experimental_configs(use_categorical=USE_CATEGORICAL)

    if dag_dict_static != configs["dag"]["dag"]:
        raise ValueError("Static dataset DAG does not match experimental configuration DAG")
    if cpdag_static != configs["cpdag_minimal"]["cpdag"]:
        raise ValueError("Static dataset CPDAG does not match experimental configuration CPDAG")

    # Filter algorithms if specified (for accurate total_experiments calculation)
    if algorithm_filter:
        if algorithm_filter not in configs:
            available_algorithms = list(configs.keys())
            raise ValueError(f"Algorithm '{algorithm_filter}' not found. Available: {available_algorithms}")
        configs = {algorithm_filter: configs[algorithm_filter]}
        print(f" Running ONLY algorithm: {algorithm_filter}")
        
        # Filter by column order if specified for the selected algorithm
        if column_order_filter:
            if column_order_filter not in ["original", "topological", "reverse_topological"]:
                raise ValueError(f"Column order '{column_order_filter}' not valid. Available: original, topological, reverse_topological")
            filtered_orderings = [ot for ot in configs[algorithm_filter]["orderings"] if ot[0] == column_order_filter]
            if not filtered_orderings:
                raise ValueError(f"No orderings found for column order '{column_order_filter}' in algorithm '{algorithm_filter}'")
            configs[algorithm_filter]["orderings"] = filtered_orderings
            print(f" Running ONLY column order: {column_order_filter}")
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
    total_experiments = len(TRAIN_SIZES) * N_REPETITIONS * total_configs  # Include TRAIN_SIZES in interventional experiment


    # ======================
    # LOAD EXISTING RESULTS IF ANY
    # ======================
    
    results: list[dict[str, Any]] = []
    completed_experiments: set[tuple] = set()
    
    # Resume support: scan all CSVs in results dir (covers any older unique naming)
    loaded_rows = 0
    for csv_path in sorted(base_output_dir.glob("*.csv")):
        try:
            existing_df = pd.read_csv(csv_path)
        except Exception:
            continue
        existing_rows = cast(list[dict[str, Any]], [dict(row) for row in existing_df.to_dict('records')])
        for row in existing_rows:
            # Respect CLI filters and ensure train_size is in our target set
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

    experiment_counter = 0
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

    # ======================
    # INTERVENTIONAL DATASET GENERATION
    # ======================
    
    intervention_mappings = {
        intervention_info["intervention_idx"]: {
            0: intervention_info["values"][0],
            1: intervention_info["values"][1],
        }
    }

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
            test_size=TEST_SIZE,
            test_seed=2000,
        )
        save_global_test_set(global_test_set, base_output_dir, column_names)
        save_remaining_pool(remaining_pool_info, base_output_dir, column_names)
        print(" Global test set and remaining pool saved for reproducibility")

    global_test_df = pd.DataFrame({
        **dict(zip(column_names, [global_test_set[:, i] for i in range(global_test_set.shape[1])])),
    })

    test_ate_ground_truth = calculate_ate(global_test_set, column_names)
    print(f"Ground truth ATE on test set: {test_ate_ground_truth:.6f}")

    all_splits: dict[int, np.ndarray] = {}
    train_dataset_paths: dict[int, str] = {}
    cached_datasets = 0
    regenerated_datasets = 0

    print("Loading/generating interventional train sets...")
    total_experiments_needed = len(TRAIN_SIZES) * N_REPETITIONS
    print(f"Will generate {total_experiments_needed} unique datasets using seeds 0 to {total_experiments_needed-1}")

    datasets_dir = base_output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    linear_seed = 0
    for train_size in TRAIN_SIZES:
        for rep_idx in range(N_REPETITIONS):
            seed = linear_seed
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
                            and metadata_obj.get("sampling_strategy") == "interventional_branch_subset"
                        ):
                            X_loaded = data["X_train"]
                            cols_loaded = data["column_names"].tolist()
                            if list(cols_loaded) == column_names and X_loaded.shape[1] == len(column_names):
                                all_splits[seed] = X_loaded
                                train_dataset_paths[seed] = str(dataset_file)
                                cached_datasets += 1
                                loaded_successfully = True
                            else:
                                print(
                                    f"  Ignoring stored train set {dataset_file.name} due to schema mismatch."
                                )
                        else:
                            print(
                                f"  Ignoring stored train set {dataset_file.name} due to missing/invalid metadata."
                            )
                except Exception as exc:
                    print(f"  Failed to load cached train set {dataset_file.name}: {exc}")

            if not loaded_successfully:
                train_sets = generate_interventional_train_sets(
                    remaining_pool_info=remaining_pool_info,
                    intervention_info=intervention_info,
                    intervention_mappings=intervention_mappings,
                    train_sizes=[train_size],
                    seed=seed,
                )
                X_train = train_sets.get(train_size, np.empty((0, len(column_names))))
                all_splits[seed] = X_train
                regenerated_datasets += 1

                if save_datasets:
                    train_path = save_train_set_per_split(
                        X_train=X_train,
                        train_size=train_size,
                        seed=seed,
                        output_dir=base_output_dir,
                        column_names=column_names,
                        overwrite=True,
                        metadata={
                            "sampling_strategy": "interventional_branch_subset",
                            "seed": seed,
                        },
                    )
                    train_dataset_paths[seed] = train_path
                elif dataset_file.exists():
                    train_dataset_paths[seed] = str(dataset_file)
                else:
                    train_dataset_paths[seed] = ""
            else:
                X_train = all_splits[seed]

            linear_seed += 1

            if linear_seed % 20 == 0:
                print(f"Processed {linear_seed} train splits")

    if cached_datasets:
        print(f"    Cached datasets loaded for {cached_datasets} seeds")
    if regenerated_datasets:
        print(f"    Regenerated datasets for {regenerated_datasets} seeds")

    # Now run interventional experiments with consistent seed management
    for algorithm, config in configs.items():
        orderings = config["orderings"]
        dag = config["dag"]
        cpdag = config["cpdag"]
        dag_for_ordering = config.get("dag_for_ordering")

        for column_order_tuple in orderings:
            # Extract column order name from tuple (order_indices pre-calculated but not used)
            column_order, _ = column_order_tuple

            # Linear seed counter approach - use same seed sequence for all algorithms
            linear_seed = 0  # Reset to 0 for each algorithm to ensure same seeds across algorithms

            for train_size in TRAIN_SIZES:
                for rep_idx in range(N_REPETITIONS):
                    seed = linear_seed  # 0, 1, 2, 3, 4, 5, ... (same for all algorithms!)
                    linear_seed += 1  # Always move to the next dataset, even if we skip below
                    repetition = rep_idx + 1  # 1-based repetition counter for this train_size
                    
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
                    # CRITICAL: Same seed = same train data + same TabPFN randomness
                    # This enables fair comparison between algorithms (vanilla vs dag vs cpdag)
                    set_experiment_seeds(seed, include_cuda=True, verbose=False)

                    # Improved logging with current configuration details
                    print(f" [{experiment_counter}/{total_experiments}] {algorithm}-{column_order} | train_size={train_size} | rep={repetition}/{N_REPETITIONS} | seed={seed}")

                    # Print progress every 50 experiments with optional memory monitoring
                    if experiment_counter % 50 == 0 and experiment_counter >= 10:
                            elapsed = time.time() - start_time
                            progress_pct = experiment_counter / total_experiments * 100
                            eta_seconds = (elapsed / experiment_counter) * (total_experiments - experiment_counter)
                            
                            print(f" Progress: {experiment_counter}/{total_experiments} ({progress_pct:.1f}%), ETA: {eta_seconds/60:.1f} min")

                    try:
                        # Get pre-generated interventional training data
                        if seed not in all_splits:
                            raise RuntimeError(f"Missing split for seed={seed}, train_size={train_size}")
                        X_train_original = all_splits[seed]

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

                        # Prepare data using algorithm-specific functions (cleaner approach)
                        if algorithm == "vanilla":
                            if dag_for_ordering is None:
                                raise ValueError(f"dag_for_ordering is required for vanilla algorithm but got None")
                            # Convert categorical column names to indices for prepare_vanilla_data
                            categorical_indices = [column_names.index(col) for col in categorical_cols] if categorical_cols else None
                            X_train_prepared, column_ordering_used, _ = prepare_vanilla_data(
                                X_train_original, column_order, dag_for_ordering, column_names, categorical_indices
                            )
                            dag_prepared, cpdag_prepared = None, None
                            
                        elif algorithm == "dag":
                            if dag is None:
                                raise ValueError(f"DAG is required for dag algorithm but got None")
                            # Optional column reordering for DAG
                            if column_order != "original":
                                if dag_for_ordering is None:
                                    raise ValueError(f"dag_for_ordering is required for DAG algorithm but got None")
                                categorical_indices = [column_names.index(col) for col in categorical_cols] if categorical_cols else None
                                X_train_reordered, column_ordering_used, _ = prepare_vanilla_data(
                                    X_train_original, column_order, dag_for_ordering, column_names, categorical_indices
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
                            cpdag_prepared = None
                            
                        elif algorithm.startswith("cpdag_"):
                            if cpdag_to_use is None:
                                raise ValueError(f"CPDAG is required for {algorithm} algorithm but got None")
                            
                            # Handle column reordering for CPDAG (like vanilla does)
                            if column_order != "original":
                                # Use the same DAG as vanilla does for computing ordering
                                if dag_for_ordering is None:
                                    raise ValueError(f"dag_for_ordering is required for CPDAG algorithm but got None")
                                categorical_indices = [column_names.index(col) for col in categorical_cols] if categorical_cols else None
                                X_train_reordered, column_ordering_used, _ = prepare_vanilla_data(
                                    X_train_original, column_order, dag_for_ordering, column_names, categorical_indices
                                )
                                
                                # Reorder CPDAG structure to match new column ordering
                                old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(column_ordering_used)}
                                cpdag_reordered = {}
                                for old_idx in range(len(column_names)):
                                    if old_idx in cpdag_to_use:
                                        new_idx = old_to_new[old_idx]
                                        cpdag_reordered[new_idx] = {
                                            "parents": [old_to_new[p] for p in cpdag_to_use[old_idx]["parents"] if p in old_to_new],
                                            "undirected": [old_to_new[u] for u in cpdag_to_use[old_idx]["undirected"] if u in old_to_new]
                                        }
                                    else:
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
                            
                            dag_prepared = None
                            
                        else:
                            raise ValueError(f"Unknown algorithm: {algorithm}")

                        # Use global test DataFrame for consistent evaluation (same for all)
                        test_df = global_test_df
                        
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
                            column_ordering_used=column_ordering_used,
                            updated_categorical_features=None,  # Not used in interventional experiment
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
                            repetition=repetition,  # Using proper repetition counter (1-100 per train_size)
                            categorical_cols=categorical_cols,
                            column_names=column_names,
                            metrics_results=metrics,
                            reordered_graph_dict=reordered_graph_dict,
                            column_ordering_used=column_ordering_used
                        )
                        
                        # Get actual column order based on algorithm and ordering strategy
                        if (algorithm == "vanilla" or algorithm == "dag" or algorithm.startswith("cpdag_")) and column_order != "original" and column_ordering_used is not None:
                            # For algorithms with reordering, use the ordering from preparation
                            actual_column_order = [column_names[i] for i in column_ordering_used]
                        else:
                            # For DAG or algorithms without reordering, columns stay in original order
                            actual_column_order = column_names.copy()
                        
                        # Add actual column order used for verification
                        result_row['actual_column_order'] = ','.join(actual_column_order)

                        train_path = train_dataset_paths.get(seed, '')
                        if train_path and not Path(train_path).exists():
                            train_path = ''
                        result_row['train_dataset_path'] = to_workspace_relative(train_path)

                        test_dataset_path = base_output_dir / 'datasets/global_test_set.npz'
                        result_row['test_dataset_path'] = (
                            to_workspace_relative(test_dataset_path)
                            if test_dataset_path.exists()
                            else ''
                        )

                        # Save reordered datasets if requested and training data was modified
                        if save_datasets:
                            # Optionally save reordered dataset for verification
                            if save_reordered and column_ordering_used is not None:
                                train_data_modified = not np.array_equal(X_train_prepared, X_train_original)
                                if train_data_modified:
                                    reordered_path = save_reordered_dataset(
                                        X_train_reordered=X_train_prepared,
                                        algorithm=algorithm,
                                        column_order=column_order,
                                        train_size=train_size,
                                        seed=seed,
                                        output_dir=base_output_dir,
                                        reordered_column_names=actual_column_order,
                                        base_train_path=train_dataset_paths[seed]
                                    )
                                    result_row['reordered_dataset_path'] = to_workspace_relative(reordered_path)
                        
                        # Save synthetic data if requested
                        if save_synthetic:
                            synthetic_path = save_synthetic_data(
                                synthetic_df=synthetic_df,
                                algorithm=algorithm,
                                column_order=column_order,
                                train_size=train_size,
                                seed=seed,
                                output_dir=base_output_dir,
                                n_permutations=N_PERMUTATIONS,
                                temperature=TEMPERATURE,
                                metrics=metrics
                            )
                            result_row['synthetic_data_path'] = to_workspace_relative(synthetic_path)

                        results.append(result_row)

                        # Save intermediate results every N experiments
                        if len(results) % save_every == 0:
                            # Get consolidated output file and persist all accumulated rows
                            _, current_output_file = get_output_paths(None, algorithm_filter, column_order_filter)
                            save_results_to_csv(results, str(current_output_file))
                            
                            # Also save crashed seeds data if any exist
                            if crashed_seeds_data:
                                algorithm_suffix = f"_{algorithm_filter}" if algorithm_filter else "_all_algorithms"
                                column_suffix = f"_{column_order_filter}" if column_order_filter else ""
                                crashed_csv_file = current_output_file.parent / f"crashed_seeds_detailed{algorithm_suffix}{column_suffix}.csv"
                                save_crashed_seeds_to_csv(crashed_seeds_data, crashed_csv_file)

                    except Exception as e:
                        error_msg = str(e)
                        if "infinity" in error_msg.lower() or "too large for dtype" in error_msg.lower():
                            print(f" INFINITY BUG: {algorithm}-{column_order} train_size={train_size} seed={seed} | SKIPPING")
                            crashed_seeds.add(seed)
                            # Record detailed crash information
                            crashed_seeds_data.append({
                                'seed': seed,
                                'algorithm': algorithm,
                                'column_order': column_order,
                                'train_size': train_size,
                                'repetition': repetition,
                                'error_type': 'infinity_bug',
                                'error_message': error_msg[:200],  # Truncate long error messages
                                'timestamp': pd.Timestamp.now().isoformat()
                            })
                        else:
                            print(f" OTHER ERROR in experiment {algorithm}-{column_order} train_size={train_size} seed={seed}: {e}")
                            # Record other errors too
                            crashed_seeds_data.append({
                                'seed': seed,
                                'algorithm': algorithm,
                                'column_order': column_order,
                                'train_size': train_size,
                                'repetition': repetition,
                                'error_type': 'other_error',
                                'error_message': error_msg[:200],  # Truncate long error messages
                                'timestamp': pd.Timestamp.now().isoformat()
                            })
                        continue
                    
                    # Periodic memory cleanup to prevent OOM without killing performance
                    finally:
                        # Heavy cleanup every 10 experiments instead of every single one
                        if experiment_counter % 10 == 0:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            import gc
                            gc.collect()
                
                # SCIENTIFIC IMPROVEMENT: Do NOT increment base_seed
                # We want same seeds across train_sizes to isolate effect of sample size
                # base_seed += N_REPETITIONS  # REMOVED

    # ======================
    # SAVE FINAL RESULTS
    # ======================

    total_time = time.time() - start_time
    print(f"\nExperiment completed in {total_time/60:.1f} minutes")

    _, output_file = get_output_paths(None, algorithm_filter, column_order_filter)

    if not results:
        print("  WARNING: No experiments produced results! Creating placeholder CSV.")
        empty_result = {
            'algorithm': 'none', 'column_order': 'none', 'train_size': TRAIN_SIZES[0] if TRAIN_SIZES else 0,
            'seed': 0, 'repetition': 0, 'ate_test': 0, 'ate_synthetic': 0,
            'ate_difference': 0, 'ate_relative_error': 0
        }
        results_to_save = [empty_result]
    else:
        results_to_save = results

    save_results_to_csv(results_to_save, str(output_file))
    print(f" Final results saved to: {output_file}")

    # Warn if any train_size has zero completed runs
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
        print("   To fix corrupted results, re-run experiments with replacement seeds outside the")
        print(f"   used range (e.g., seeds {max(crashed_list) + 1000} and above).")
        
        # Save crashed seeds to file for post-processing (algorithm-specific to avoid conflicts)
        algorithm_suffix = f"_{algorithm_filter}" if algorithm_filter else "_all_algorithms"
        crashed_seeds_file = base_output_dir / f"crashed_seeds{algorithm_suffix}.txt"
        with open(crashed_seeds_file, 'w') as f:
            f.write("# Seeds that triggered TabPFN infinity bug\n")
            f.write("# These experiments returned -1.0 metrics and need replacement\n")
            f.write(f"# Total crashed seeds: {len(crashed_list)}\n")
            f.write("# Crashed seeds (one per line):\n")
            for seed in crashed_list:
                f.write(f"{seed}\n")
        print(f"   Crashed seeds list saved to: {crashed_seeds_file}")
        
        # Save detailed crashed seeds data to CSV
        if crashed_seeds_data:
            crashed_csv_file = base_output_dir / f"crashed_seeds_detailed{algorithm_suffix}.csv"
            save_crashed_seeds_to_csv(crashed_seeds_data, crashed_csv_file)
    else:
        print(" No crashed seeds detected! All experiments completed successfully.")
    
    print("=" * 60)

    # ======================
    # SUMMARY STATISTICS
    # ======================


    # Count experiments by configuration
    from collections import Counter

    config_counts = Counter()
    train_size_counts = Counter()

    for result in results:
        config_key = f"{result['algorithm']}-{result['column_order']}"
        config_counts[config_key] += 1
        train_size_counts[result["train_size"]] += 1

    # Configuration counts available for debugging
    _ = config_counts
    _ = train_size_counts



if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Causal TabPFN Interventional Experiment (custom SCM)")
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
    parser.add_argument(
        "--column-order",
        help=(
            "For vanilla, DAG, and all CPDAG variants, specify column ordering: original, topological, reverse_topological"
        ),
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
        column_order_filter=args.column_order,
        train_sizes_group=args.train_sizes_group,
    )
