
"""Comprehensive comparison experiment for causal TabPFN configurations.

This script runs a systematic comparison of:
1. Vanilla TabPFN with different column orderings (original, topological, worst)
2. DAG-aware TabPFN with causal graph constraints  
3. CPDAG-aware TabPFN with hybrid causal/correlational constraints (discovered)
4. CPDAG-ideal TabPFN with ideal causal structure (ground truth)

The experiment tests all configurations across multiple training sizes,
seeds, and repetitions to provide robust statistical comparisons.

IMPORTANT: Uses consistent seeds across train_sizes to isolate the effect
of sample size. Each seed samples a maximum dataset from CSuite, with smaller 
train_sizes using subsets to ensure fair comparison and proper causal inference.

Usage Examples:

    # Run ALL algorithms and configurations (default)
    uv run causal_experiments/csuite_experiment/comparison_experiment_csuite/csuite_comparison_experiment.py
    
    # Run ALL algorithms in test mode (faster)
    uv run causal_experiments/csuite_experiment/comparison_experiment_csuite/csuite_comparison_experiment.py --test
    
    # Run ONLY vanilla algorithm (all 3 orderings: original, topological, worst)
    uv run causal_experiments/csuite_experiment/comparison_experiment_csuite/csuite_comparison_experiment.py --algorithm vanilla
    
    # Run ONLY vanilla algorithm in test mode
    uv run causal_experiments/csuite_experiment/comparison_experiment_csuite/csuite_comparison_experiment.py --algorithm vanilla --test
    
    # Run ONLY vanilla with specific column ordering (useful for splitting heavy jobs)
    uv run causal_experiments/csuite_experiment/comparison_experiment_csuite/csuite_comparison_experiment.py --algorithm vanilla --column-order original
    uv run causal_experiments/csuite_experiment/comparison_experiment_csuite/csuite_comparison_experiment.py --algorithm vanilla --column-order topological
    uv run causal_experiments/csuite_experiment/comparison_experiment_csuite/csuite_comparison_experiment.py --algorithm vanilla --column-order worst
    
    # Run ONLY DAG-aware algorithm (original ordering only)
    uv run causal_experiments/csuite_experiment/comparison_experiment_csuite/csuite_comparison_experiment.py --algorithm dag
    
    # Run CPDAG v1 both-vanilla discovered/minimal
    uv run causal_experiments/csuite_experiment/comparison_experiment_csuite/csuite_comparison_experiment.py --algorithm cpdag_discovered
    uv run causal_experiments/csuite_experiment/comparison_experiment_csuite/csuite_comparison_experiment.py --algorithm cpdag_minimal
    
    # Other options work with --algorithm filter
    uv run causal_experiments/csuite_experiment/comparison_experiment_csuite/csuite_comparison_experiment.py --algorithm vanilla --save-datasets
    uv run causal_experiments/csuite_experiment/comparison_experiment_csuite/csuite_comparison_experiment.py --algorithm dag --save-every 50
    uv run causal_experiments/csuite_experiment/comparison_experiment_csuite/csuite_comparison_experiment.py --dataset csuite_lingauss --algorithm vanilla
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
    FaithfulDataEvaluator,
    create_result_row,
    discover_cpdag_from_data,
    identify_vanilla_duplicates,
    copy_vanilla_duplicate_result,
    prepare_vanilla_data,
    prepare_dag_data,
    prepare_cpdag_data,
    _get_true_dag_for_ordering,
    save_global_test_set,
    save_train_set_per_split,
    load_global_test_set,
    # CSuite dataset utilities
    load_csuite_dataset,
    list_available_datasets,
    save_reordered_dataset,
    save_synthetic_data,
    save_results_to_csv,
    # Determinism utilities
    setup_determinism,
    set_experiment_seeds,
)
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised

# Global flag to print device info only once
_device_info_printed = False


_LINEAR_DATASETS = {"csuite_lingauss", "csuite_linexp"}

# Enable reuse of cached train splits when they were generated with the new
# independent observational sampling strategy. Otherwise, regenerate them.
_REUSE_CACHED_TRAIN_SPLITS = True


def _choose_indep_test(dataset_name: str, categorical_cols: list[str]) -> str:
    if categorical_cols:
        if dataset_name.lower() == "csuite_mixed_confounding":
            return "hybrid"
        return "gsq"
    if dataset_name.lower() in _LINEAR_DATASETS:
        return "fisherz"
    return "kci"


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

        cpdag_to_pass = cpdag if algorithm.startswith("cpdag_") else None

        # Determine feature names to reflect the CURRENT column order seen by the model
        # If we applied any reordering (including randomized replacement of 'original'),
        # reflect that regardless of the column_order label.
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
        
        # Calculate metrics using the evaluator
        evaluator = FaithfulDataEvaluator()
        metrics = evaluator.evaluate(
            real_data=test_df,  # Use test data as real reference
            synthetic_data=synthetic_df,  # Use synthetic data
            categorical_columns=categorical_cols,
            k_for_kmarginal=2,
            random_seed=seed  # Use experiment seed for reproducible metrics
        )

        # Flatten propensity metrics if they exist
        if "propensity_metrics" in metrics:
            propensity = metrics.pop("propensity_metrics")
            if isinstance(propensity, dict):
                for key, value in propensity.items():
                    metrics[f"propensity_{key}"] = value

        return metrics, synthetic_df

    except Exception as e:
        # Print the actual error for debugging
        print(f"Error in run_single_experiment: {e}")
        import traceback
        print(traceback.format_exc())
        # Return default error values for all metrics
        return {
            "correlation_matrix_difference": -1.0,
            "k_marginal_tvd": -1.0,
            "nnaa": -1.0,
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
    dataset_name: str = "csuite_mixed_confounding",
    algorithm_filter: str | None = None,
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
        dataset_name: Name of CSuite dataset to use
        algorithm_filter: If provided, run only the specified algorithm (vanilla, dag, cpdag_v1_both_vanilla_*)
        column_order_filter: If provided, run only the specified column ordering for vanilla (original, topological, worst)
        skip_seeds: Explicit seed values to skip (their repetitions are skipped and replacements are sampled to maintain total repetitions)
    """
    # ======================
    # EXPERIMENTAL PARAMETERS
    # ======================

    if test_mode:
        TRAIN_SIZES = [20, 50]
        TEST_SIZE = 100
        # Small default in test mode unless overridden
        N_REPETITIONS = repetitions if repetitions is not None else 2
    else:
        TRAIN_SIZES = [20, 50, 100, 200, 500]
        TEST_SIZE = 2000
        # Default repetitions for full runs (can be overridden)
        N_REPETITIONS = repetitions if repetitions is not None else 150

    # Optional: restrict to small/large train-size groups (helps stay under 24h)
    if train_sizes_group:
        group_small = [20, 50, 100]
        group_large = [200, 500]
        if train_sizes_group == "small":
            TRAIN_SIZES = [ts for ts in TRAIN_SIZES if ts in group_small]
        elif train_sizes_group == "large":
            TRAIN_SIZES = [ts for ts in TRAIN_SIZES if ts in group_large]
        elif train_sizes_group == "all":
            pass
        else:
            raise ValueError("train_sizes_group must be one of: small, large, all")

    # TabPFN parameters
    N_ESTIMATORS = 3
    N_PERMUTATIONS = 3
    TEMPERATURE = 1.0
    
    # Linear seed approach: 0,1,2,3,4,5... (same seed = same dataset for all algorithms = fair comparison)
    # Each repetition gets a unique seed: 0, 1, 2, ..., total_experiments-1
    # This ensures: (1) Same seed = same data across algorithms, (2) Reproducible datasets

    skip_seeds_set = set(skip_seeds or [])
    if skip_seeds_set:
        print(f"  Requested skip seeds: {sorted(skip_seeds_set)}")

    # Linear seed sequence - create mapping across ALL experiments
    total_datasets_needed = len(TRAIN_SIZES) * N_REPETITIONS
    selected_seeds: list[int] = []
    skipped_schedule: list[tuple[int, int]] = []

    linear_seed = 0  # Start from 0, increment linearly
    while len(selected_seeds) < total_datasets_needed:
        if linear_seed in skip_seeds_set:
            skipped_schedule.append((linear_seed, len(selected_seeds) + 1))
            linear_seed += 1
            continue
        selected_seeds.append(linear_seed)
        linear_seed += 1

    if skipped_schedule:
        for seed_value, repetition_idx in skipped_schedule:
            print(f"   • Skipping seed {seed_value} (would have been experiment {repetition_idx})")
        if selected_seeds:
            print(
                f"  Effective seed schedule covers {len(selected_seeds)} experiments: "
                f"first={selected_seeds[0]} last={selected_seeds[-1]}"
            )
    else:
        if selected_seeds:
            print(
                f"  Using linear seeds starting at 0 "
                f"for {len(selected_seeds)} experiments"
            )

    # Output configuration
    output_dir = Path("causal_experiments/csuite_experiment/comparison_experiment_csuite/results") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)  # Create directory with dataset name
    
    # Create unique output filename based on filters (no special v2 naming)
    if algorithm_filter and column_order_filter:
        output_file = output_dir / f"csuite_{dataset_name}_{algorithm_filter}_{column_order_filter}_results.csv"
    elif algorithm_filter:
        output_file = output_dir / f"csuite_{dataset_name}_{algorithm_filter}_results.csv"
    elif column_order_filter:
        # When running all algorithms but a specific order globally, include the order in filename
        output_file = output_dir / f"csuite_{dataset_name}_{column_order_filter}_results.csv"
    else:
        output_file = output_dir / f"csuite_{dataset_name}_results.csv"


    # ======================
    # CSUITE DATA LOADING
    # ======================

    print(f"Loading CSuite dataset: {dataset_name}")
    available_datasets = list_available_datasets()
    print(f"Available datasets: {available_datasets}")
    
    if dataset_name not in available_datasets:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available_datasets}")
    
    # Load CSuite dataset
    csuite_data = load_csuite_dataset(dataset_name)
    
    # Extract dataset components
    full_train_data = csuite_data['train_data']  # Shape: (N_train, n_features)  
    global_test_set = csuite_data['test_data']   # Shape: (N_test, n_features)
    dag_dict = csuite_data['dag_dict']           # DAG structure
    column_names = csuite_data['column_names']   # Variable names
    categorical_cols = csuite_data['categorical_columns']  # Categorical variables
    n_features = csuite_data['n_features']
    
    # Update experimental parameters based on CSuite data
    TEST_SIZE = len(global_test_set)  # Dynamic test size
    if TEST_SIZE != 2000:
        raise ValueError(
            f"Expected CSuite observational test set to contain exactly 2000 samples, got {TEST_SIZE}"
        )
    USE_CATEGORICAL = len(categorical_cols) > 0  # Auto-detect categorical features
    
    datasets_dir = output_dir / "datasets"

    # Attempt to reuse persisted global test set if available
    loaded_global = load_global_test_set(output_dir)
    if loaded_global is not None:
        loaded_test, loaded_cols = loaded_global
        if list(loaded_cols) == column_names and loaded_test.shape[1] == len(column_names):
            global_test_set = loaded_test
            print(" Loaded global test set from existing datasets/global_test_set.npz")
        else:
            print("  Saved global test set has mismatched schema; regenerating from source data.")

    print(f"Dataset loaded successfully:")
    print(f"  - Features: {n_features}")
    print(f"  - Train samples available: {len(full_train_data)}")
    print(f"  - Test samples: {TEST_SIZE}")
    print(f"  - Column names: {column_names}")
    print(f"  - Categorical columns: {categorical_cols}")
    print(f"  - DAG structure: {dag_dict}")
    print(f"  - Has categorical features: {USE_CATEGORICAL}")

    cpdag_indep_test = _choose_indep_test(dataset_name, categorical_cols)
    print(f"  - Independence test for CPDAG discovery: {cpdag_indep_test}")
    # Default significance level for PC discovery
    alpha = 0.05


    # ======================
    # MODEL INITIALIZATION
    # ======================


    # Note: TabPFN models will be created fresh for each experiment to avoid state contamination

    # ======================
    # EXPERIMENTAL CONFIGURATIONS  
    # ======================

    # Use CSuite DAG structure for experimental configurations
    # Temporarily create configurations inline (will need to modify get_experimental_configs later)
    from causal_experiments.utils.dag_utils import get_ordering_strategies, dag_to_ideal_cpdag
    
    # Get orderings based on CSuite DAG
    orderings_dict = get_ordering_strategies(dag_dict)

    # Optional dataset-specific overrides for column permutations.
    custom_vanilla_orderings: dict[str, list[int]] | None = None

    # Hard-code column permutations for datasets with multiple valid
    # topological sorts to avoid ambiguity in downstream comparisons.
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

    # If original equals topological or worst, create a deterministic random ordering to use instead.
    def deterministic_random_ordering(n_features: int, dataset_key: str) -> list[int]:
        import hashlib
        salt = "random_original_v1"
        key = f"{dataset_key}:{salt}"
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        seed = int(h[:16], 16) % (2**32)
        rng = np.random.default_rng(seed)
        return list(rng.permutation(n_features))

    random_original_ordering: list[int] | None = None
    orig_ord = orderings_dict.get("original")
    topo_ord = orderings_dict.get("topological")
    reverse_topo_ord = orderings_dict.get("reverse_topological")
    if orig_ord is not None and (orig_ord == topo_ord or orig_ord == reverse_topo_ord):
        random_original_ordering = deterministic_random_ordering(len(orig_ord), dataset_name)
        # Ensure distinct from topo/reverse_topological (extremely unlikely to collide)
        if random_original_ordering == topo_ord or random_original_ordering == reverse_topo_ord:
            random_original_ordering = random_original_ordering[1:] + random_original_ordering[:1]
        print("  Original ordering equals topological/reverse_topological. Using deterministic random ordering for 'original'.")
        print(f"   Random original ordering: {random_original_ordering}")
    
    # Generate ideal CPDAG from CSuite DAG
    ideal_cpdag = dag_to_ideal_cpdag(dag_dict)
    
    # CPDAG format for CSuite DAG (will be discovered dynamically in experiments)
    # For now, create a placeholder that will be replaced during experiments
    csuite_cpdag = {node: {"parents": parents, "undirected": []} for node, parents in dag_dict.items()}
    
    configs = {
        "vanilla": {
            "orderings": [
                ("original", random_original_ordering if random_original_ordering is not None else orderings_dict["original"]),
                ("topological", orderings_dict["topological"]),
                ("reverse_topological", orderings_dict["reverse_topological"])
            ],
            "dag": None,  # Vanilla never uses DAG for causal constraints
            "cpdag": None,
            "dag_for_ordering": dag_dict  # Used only for vanilla column reordering
        },
        "dag": {
            "orderings": [
                ("topological", orderings_dict["topological"])
            ],
            "dag": dag_dict,
            "cpdag": None,
            "dag_for_ordering": dag_dict  # Used to compute column orderings for DAG
        },
        # CPDAG v1 both-vanilla (discovered)
        "cpdag_discovered": {
            "orderings": [
                ("original", orderings_dict["original"])
            ],
            "dag": None,
            "cpdag": csuite_cpdag,
            "dag_for_ordering": None,
        },
        # CPDAG minimal
        "cpdag_minimal": {
            "orderings": [
                ("original", orderings_dict["original"])
            ],
            "dag": None,
            "cpdag": ideal_cpdag,
            "dag_for_ordering": None,
        },
    }

    
    # Filter column orderings if specified (must be done before algorithm filtering)
    if column_order_filter:
        available_orderings = ["original", "topological", "reverse_topological"]
        if column_order_filter not in available_orderings:
            raise ValueError(f"Column order '{column_order_filter}' not found. Available: {available_orderings}")
        
        # If a specific algorithm is selected, filter its orderings
        if algorithm_filter and algorithm_filter in configs:
            filtered = [ot for ot in configs[algorithm_filter]["orderings"] if ot[0] == column_order_filter]
            if not filtered:
                raise ValueError(f"Column order '{column_order_filter}' not found in orderings for algorithm '{algorithm_filter}'")
            configs[algorithm_filter]["orderings"] = filtered
            print(f" Running ONLY {algorithm_filter} column order: {column_order_filter}")
        else:
            # Otherwise, filter vanilla orderings by default (historical behavior)
            if "vanilla" in configs:
                vanilla_orderings = configs["vanilla"]["orderings"]
                filtered_orderings = [
                    (order_name, order_list) for order_name, order_list in vanilla_orderings 
                    if order_name == column_order_filter
                ]
                if not filtered_orderings:
                    raise ValueError(f"Column order '{column_order_filter}' not found in vanilla orderings")
                configs["vanilla"]["orderings"] = filtered_orderings
                print(f" Running ONLY vanilla column order: {column_order_filter}")
            else:
                print(f"  Column order filter '{column_order_filter}' ignored (no vanilla algorithm)")

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

    # Resume support: scan all CSVs in dataset results dir (handles any prior unique suffix files)
    loaded_rows = 0
    for csv_path in sorted(output_dir.glob("*.csv")):
        try:
            existing_df = pd.read_csv(csv_path)
        except Exception:
            continue
        existing_rows = cast(list[dict[str, Any]], [dict(row) for row in existing_df.to_dict('records')])
        for row in existing_rows:
            # Respect filters if provided
            if algorithm_filter and row.get('algorithm') != algorithm_filter:
                continue
            if column_order_filter and row.get('column_order') != column_order_filter:
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
    for csv_path in sorted(output_dir.glob("*.csv")):
        try:
            existing_df = pd.read_csv(csv_path)
        except Exception:
            continue
        for _, row in existing_df.iterrows():
            metric_cols = [
                'correlation_matrix_difference', 'k_marginal_tvd', 'nnaa'
            ]
            if any(row.get(col, 0) == -1 for col in metric_cols):
                crashed_seeds.add(row.get('seed'))
    if crashed_seeds:
        print(f"Recovered {len(crashed_seeds)} crashed seeds from existing results: {sorted(crashed_seeds)}")

    # ======================
    # CSUITE DATASET PREPARATION AND SAMPLING
    # ======================
    
    # Convert CSuite data to DataFrame format for consistency
    global_test_df = pd.DataFrame(global_test_set, columns=column_names)
    full_train_df = pd.DataFrame(full_train_data, columns=column_names)
    
    print(f"CSuite test set ready: {len(global_test_df)} samples")
    print(f"CSuite train pool ready: {len(full_train_df)} samples")
    
    # Save global test set if datasets saving is enabled
    if save_datasets:
        save_global_test_set(global_test_set, output_dir, column_names)
    
    # Step 2: Linear seed approach for CSuite sampling - ultra simple!
    # SCIENTIFIC PRINCIPLE: Same seed = same training data for all algorithms = fair comparison
    # Generate datasets with linear counter: 0, 1, 2, 3, 4, ... across all (train_size, repetition) combinations
    all_splits = {}  # {seed: X_train} - one dataset per seed, indexed by linear counter
    train_dataset_paths = {}  # {seed: train_dataset_path}
    seed_to_metadata = {}  # {seed: (train_size, rep_idx)} - for logging/debugging

    print("Pre-sampling train sets from CSuite data with linear seed management...")
    
    # Calculate maximum train_size to sample full datasets
    max_train_size = max(TRAIN_SIZES)
    
    if max_train_size > len(full_train_df):
        raise ValueError(f"Requested max_train_size={max_train_size} but CSuite dataset only has {len(full_train_df)} samples")
    
    print(f"Will generate {len(selected_seeds)} unique datasets using linear seeds")

    # Linear mapping: create datasets with linear seed counter
    cached_datasets = 0
    regenerated_datasets = 0
    linear_seed = 0
    for train_size in TRAIN_SIZES:
        for rep_idx in range(N_REPETITIONS):
            if linear_seed >= len(selected_seeds):
                break
            seed = selected_seeds[linear_seed]
            seed_to_metadata[seed] = (train_size, rep_idx + 1)  # For logging

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
                            and metadata_obj.get("sampling_strategy") == "observational_random_subset"
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

            if loaded_successfully:
                linear_seed += 1
                continue

            # Generate new dataset by sampling from CSuite with linear seed
            rng = np.random.default_rng(seed)
            sampled_indices = rng.choice(len(full_train_df), size=train_size, replace=False).astype(int)
            X_train = full_train_df.iloc[sampled_indices].values

            all_splits[seed] = X_train

            # Save dataset if requested
            expected_path = datasets_dir / f"train_ts{train_size}_s{seed}.npz"
            if save_datasets:
                train_path = save_train_set_per_split(
                    X_train=X_train,
                    train_size=train_size,
                    seed=seed,
                    output_dir=output_dir,
                    column_names=column_names,
                    dataset_seed=seed,
                    overwrite=True,
                    metadata={
                        "sampling_strategy": "observational_random_subset",
                        "seed": seed,
                    },
                )
                train_dataset_paths[seed] = train_path
            elif expected_path.exists():
                train_dataset_paths[seed] = str(expected_path)
            else:
                train_dataset_paths[seed] = ""

            regenerated_datasets += 1
            linear_seed += 1

            if linear_seed % 20 == 0:
                print(f"Processed {linear_seed} datasets...")

    if cached_datasets:
        print(f"    Cached datasets loaded for {cached_datasets} seeds")
    if regenerated_datasets:
        print(f"    Regenerated datasets for {regenerated_datasets} seeds")

    # Now run experiments with proper seed management and fair comparison
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
                    if linear_seed >= len(selected_seeds):
                        break
                    seed = selected_seeds[linear_seed]
                    repetition = rep_idx + 1  # 1-based repetition counter for this train_size
                    linear_seed += 1  # Always move to next dataset
                    
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
                    # Simplified logging with linear seed approach
                    train_size_from_metadata, rep_from_metadata = seed_to_metadata.get(
                        seed, (train_size, repetition)
                    )
                    print(f" [{experiment_counter}/{total_experiments}] {algorithm}-{column_order} | train_size={train_size} | rep={repetition}/{N_REPETITIONS} | seed={seed} | (dataset: train_size={train_size_from_metadata}, rep={rep_from_metadata})")

                    # Print progress every 50 experiments with memory monitoring
                    if experiment_counter % 50 == 0 and experiment_counter >= 10:
                        elapsed = time.time() - start_time
                        progress_pct = experiment_counter / total_experiments * 100
                        eta_seconds = (elapsed / experiment_counter) * (total_experiments - experiment_counter)
                        
                        # Memory monitoring for HPC debugging
                        try:
                            import psutil
                            process = psutil.Process()
                            memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
                            print(f" Progress: {experiment_counter}/{total_experiments} ({progress_pct:.1f}%), ETA: {eta_seconds/60:.1f} min, RAM: {memory_gb:.1f}GB")
                        except ImportError:
                            print(f" Progress: {experiment_counter}/{total_experiments} ({progress_pct:.1f}%), ETA: {eta_seconds/60:.1f} min")

                    try:
                        # Get pre-validated training data
                        # Get pre-loaded dataset using linear seed (much simpler!)
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

                        # Prepare data using algorithm-specific functions (cleaner approach)
                        updated_categorical_features = None
                        if algorithm == "vanilla":
                            if dag_for_ordering is None:
                                raise ValueError(f"dag_for_ordering is required for vanilla algorithm but got None")
                            # Convert categorical column names to indices for prepare_vanilla_data
                            categorical_indices = [column_names.index(col) for col in categorical_cols] if categorical_cols else None
                            # Dataset-specific overrides take precedence
                            if custom_vanilla_orderings and column_order in custom_vanilla_orderings:
                                column_ordering_used = custom_vanilla_orderings[column_order]
                                X_train_prepared = X_train_original[:, column_ordering_used]
                                updated_categorical_features = None
                                if categorical_indices is not None:
                                    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(column_ordering_used)}
                                    updated_categorical_features = [old_to_new[i] for i in categorical_indices if i in old_to_new]
                            # If 'original' duplicates others, replace with deterministic random
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
                            
                            # Handle column reordering for CPDAG (like vanilla does)
                            if (column_order != "original") or (column_order == "original" and random_original_ordering is not None):
                                # Use the same DAG as vanilla does
                                dag_for_ordering = dag_dict
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
                        # Test data is NEVER processed - always stays in original order for fair comparison
                        # Note: X_test_original is not used directly; global_test_df is used instead

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
                            repetition=repetition,  # Using proper repetition counter (1-100 per train_size)
                            categorical_cols=categorical_cols,
                            column_names=column_names,
                            metrics_results=metrics,
                            reordered_graph_dict=reordered_graph_dict,
                            column_ordering_used=column_ordering_used
                        )
                        
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

                        train_path = train_dataset_paths.get(seed, '')
                        if train_path and not Path(train_path).exists():
                            train_path = ''
                        result_row['train_dataset_path'] = train_path

                        test_rel_path = 'datasets/global_test_set.npz'
                        result_row['test_dataset_path'] = (
                            test_rel_path
                            if (output_dir / test_rel_path).exists()
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
                                        output_dir=output_dir,
                                        reordered_column_names=actual_column_order,
                                        base_train_path=train_dataset_paths[seed]
                                    )
                                    result_row['reordered_dataset_path'] = reordered_path
                        
                        # Save synthetic data if requested
                        if save_synthetic:
                            synthetic_path = save_synthetic_data(
                                synthetic_df=synthetic_df,
                                algorithm=algorithm,
                                column_order=column_order,
                                train_size=train_size,
                                seed=seed,
                                output_dir=output_dir,
                                n_permutations=N_PERMUTATIONS,
                                temperature=TEMPERATURE,
                                metrics=metrics
                            )
                            result_row['synthetic_data_path'] = synthetic_path

                        results.append(result_row)

                        # Save intermediate results every N experiments
                        if len(results) % save_every == 0:
                            save_results_to_csv(results, str(output_file))
                            
                            # Also save crashed seeds data if any exist
                            if crashed_seeds_data:
                                algorithm_suffix = f"_{algorithm_filter}" if algorithm_filter else "_all_algorithms"
                                crashed_csv_file = output_dir / f"crashed_seeds_detailed{algorithm_suffix}.csv"
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
                                'error_type': 'other_error',
                                'error_message': error_msg[:200],  # Truncate long error messages
                                'timestamp': pd.Timestamp.now().isoformat()
                            })
                        continue
                    
                    # Periodic memory cleanup to prevent OOM without killing performance
                    finally:
                        if experiment_counter % 10 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Seeds are shared across train_sizes through selected_seeds; no manual increment needed

    # ======================
    # SAVE FINAL RESULTS
    # ======================

    total_time = time.time() - start_time
    print(f"\nExperiment completed in {total_time/60:.1f} minutes")

    # Save final results (original behavior: persist whatever was generated)
    save_results_to_csv(results, str(output_file))
    
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
        crashed_seeds_file = output_dir / f"crashed_seeds{algorithm_suffix}.txt"
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
            crashed_csv_file = output_dir / f"crashed_seeds_detailed{algorithm_suffix}.csv"
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
    parser = argparse.ArgumentParser(description="Causal TabPFN Comparison Experiment")
    parser.add_argument("--test", action="store_true", help="Run in test mode (faster, fewer experiments)")
    parser.add_argument("--save-every", type=int, default=100, help="Save results every N experiments (default: 100)")
    parser.add_argument("--save-datasets", action="store_true", help="Save train/test datasets to .npz files for reproducibility")
    parser.add_argument("--save-reordered", action="store_true", help="Save reordered datasets for verification (requires --save-datasets)")
    parser.add_argument("--save-synthetic", action="store_true", help="Save synthetic data for analysis and debugging")
    parser.add_argument("--datasets", type=str, default="csuite_mixed_confounding", help="Name of CSuite dataset to use (default: csuite_mixed_confounding)")
    parser.add_argument(
        "--repetitions", type=int, default=None,
        help="Number of repetitions per (algorithm, order, train_size). Default 150 (2 in --test) if not provided."
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
        help="Optionally restrict train sizes: small=[20,50,100], large=[200,500]."
    )
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
            "Run only specific column ordering (vanilla, DAG, CPDAG): original, topological, worst"
        ),
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
        dataset_name=args.datasets,
        algorithm_filter=args.algorithm,
        column_order_filter=getattr(args, 'column_order', None),
        repetitions=args.repetitions,
        train_sizes_group=args.train_sizes_group,
        skip_seeds=args.skip_seeds,
    )
