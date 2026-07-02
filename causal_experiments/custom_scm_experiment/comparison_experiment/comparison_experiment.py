
"""Comprehensive comparison experiment for causal TabPFN configurations.

This script runs a systematic comparison of:
1. Vanilla TabPFN with different column orderings (original, topological, worst)
2. DAG-aware TabPFN with causal graph constraints
3. CPDAG: hybrid approach; for undirected edges both nodes use vanilla (condition on all previous nodes)

The experiment tests all configurations across multiple training sizes,
seeds, and repetitions to provide robust statistical comparisons.

Algorithm Details:
- vanilla: tests 3 orderings (original, topological, worst)
- dag: uses only original (ordering is imposed by DAG)
- cpdag_discovered|minimal: symmetric, CPDAG from discovery or ideal, causal structures generated first
 

Usage Examples:

    # Run ALL algorithms and configurations (default)
    uv run causal_experiments/custom_scm_experiment/comparison_experiment/comparison_experiment.py
    
    # Run ALL algorithms in test mode (faster)
    uv run causal_experiments/custom_scm_experiment/comparison_experiment/comparison_experiment.py --test
    
    # Run ONLY vanilla algorithm (all 3 orderings: original, topological, worst)
    uv run causal_experiments/custom_scm_experiment/comparison_experiment/comparison_experiment.py --algorithm vanilla
    
    # Run ONLY vanilla algorithm in test mode
    uv run causal_experiments/custom_scm_experiment/comparison_experiment/comparison_experiment.py --algorithm vanilla --test
    
    # Run ONLY DAG-aware algorithm (original ordering only)
    uv run causal_experiments/custom_scm_experiment/comparison_experiment/comparison_experiment.py --algorithm dag
    
    # Run CPDAG discovered (causal structures first)
    uv run causal_experiments/custom_scm_experiment/comparison_experiment/comparison_experiment.py --algorithm cpdag_discovered

    # Run CPDAG minimal (ground truth, causal structures first)
    uv run causal_experiments/custom_scm_experiment/comparison_experiment/comparison_experiment.py --algorithm cpdag_minimal
    
    # Other options work with --algorithm filter
    uv run causal_experiments/custom_scm_experiment/comparison_experiment/comparison_experiment.py --algorithm vanilla --save-datasets
    uv run causal_experiments/custom_scm_experiment/comparison_experiment/comparison_experiment.py --algorithm dag --save-every 50
    
"""
from __future__ import annotations

import os
# Set environment for maximal determinism before importing torch
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
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

from causal_experiments.utils import (
    FaithfulDataEvaluator,
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
    # Determinism utilities
    setup_determinism,
    set_experiment_seeds,
)
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised

# Global flag to print device info only once
_device_info_printed = False


def _format_noise_tag(noise_level: float) -> str:
    """Format a compact, filesystem-friendly noise tag (e.g., noise1e-2)."""
    if noise_level == 0:
        return "noise0"
    sci = f"{noise_level:.0e}"
    sci = sci.replace("e-0", "e-").replace("e+0", "e+")
    return f"noise{sci}"


def _load_custom_static_dataset(
    *,
    use_categorical: bool,
    n_samples: int = 6000,
    noise_level: float | None = None,
) -> tuple[np.ndarray, list[str], list[str], dict[int, list[int]], dict[int, dict[str, list[int]]]]:
    """Load the pre-generated custom SCM dataset and graph metadata."""

    dataset_kind = "mixed" if use_categorical else "numeric"
    base_name = f"custom_{dataset_kind}_scm_{n_samples}"
    if noise_level is not None:
        base_name = f"{base_name}_{_format_noise_tag(noise_level)}"

    base_dir = Path(__file__).parent / "generated_static_scm"
    csv_path = base_dir / f"{base_name}.csv"
    graphs_path = base_dir / f"{base_name}.graphs.json"

    if not csv_path.exists() or not graphs_path.exists():
        noise_hint = f" --noise-level {noise_level:g}" if noise_level is not None else ""
        raise FileNotFoundError(
            f"Missing static dataset files for custom SCM ({dataset_kind}). "
            f"Expected: {csv_path} and {graphs_path}. Run generate_custom_scm_dataset.py{noise_hint} first."
        )

    with graphs_path.open("r", encoding="utf-8") as f:
        graphs = json.load(f)

    column_names: list[str] = graphs["column_names"]
    categorical_cols: list[str] = graphs.get("categorical_columns", [])

    dag_raw: dict[str, list[int]] = graphs["dag_dict"]
    dag_dict: dict[int, list[int]] = {
        int(k): [int(v) for v in vals] for k, vals in dag_raw.items()
    }

    cpdag_raw: dict[str, dict[str, list[int]]] = graphs["cpdag_dict"]
    cpdag_dict: dict[int, dict[str, list[int]]] = {}
    for key, value in cpdag_raw.items():
        cpdag_dict[int(key)] = {
            "parents": [int(x) for x in value.get("parents", [])],
            "undirected": [int(x) for x in value.get("undirected", [])],
        }

    df = pd.read_csv(csv_path)
    df = df[column_names]
    X_all = df.to_numpy(dtype=float)

    return X_all, column_names, categorical_cols, dag_dict, cpdag_dict

# Reuse cached train splits generated with the standard observational sampling
# strategy. If metadata is missing or incompatible, regenerate them.
_REUSE_CACHED_TRAIN_SPLITS = True


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
    print(f"Crashed seeds data saved to: {output_file}")






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
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
            print(f"TabPFN Classifier device: {clf.device}")
            print(f"TabPFN Regressor device: {reg.device}")
            # Check what get_device actually returns
            from tabpfn_extensions.utils import get_device
            print(f"get_device('auto') returns: {get_device('auto')}")
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

        cpdag_to_pass = cpdag

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
        
        # Fix column ordering issue for algorithms with reordering (vanilla and CPDAG)
        if column_order != "original" and column_ordering_used is not None and (
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
    algorithm_filter: str | None = None,
    column_order_filter: str | None = None,
    noise_level: float | None = None,
    train_sizes: list[int] | None = None,
    repetitions: int | None = None,
    seed_start: int = 0,
):
    """Main experimental pipeline.
    
    Args:
        test_mode: If True, run with fewer experiments for testing
        save_every: Save CSV results every N experiments
        save_datasets: If True, save train/test datasets to .npz files
        save_reordered: If True, save reordered datasets for verification
        save_synthetic: If True, save synthetic data for analysis
        algorithm_filter: If provided, run only the specified algorithm
        column_order_filter: If provided, run only the specified column ordering (for vanilla and CPDAG algorithms)
        train_sizes: Optional train-size override
        repetitions: Optional number of repetitions per train size
        seed_start: Starting seed for the linear seed schedule
    """
    # ======================
    # EXPERIMENTAL PARAMETERS
    # ======================

    if test_mode:
        ALL_TRAIN_SIZES_ORDERED = [20, 50, 100]
        TRAIN_SIZES = [20, 50, 100]
        TEST_SIZE = 100
        N_REPETITIONS = 10  # Number of repetitions per (algorithm, column_order, train_size)
    else:
        ALL_TRAIN_SIZES_ORDERED = [20, 50, 100, 200, 500]
        TRAIN_SIZES = [20, 50, 100, 200, 500]
        TEST_SIZE = 2000
        N_REPETITIONS = 120  # Number of repetitions per (algorithm, column_order, train_size)

    if train_sizes is not None:
        requested_train_sizes = sorted(set(train_sizes))
        invalid_sizes = [ts for ts in requested_train_sizes if ts not in ALL_TRAIN_SIZES_ORDERED]
        if invalid_sizes:
            raise ValueError(
                f"Invalid train sizes {invalid_sizes}; expected subset of {ALL_TRAIN_SIZES_ORDERED}"
            )
        TRAIN_SIZES = sorted(requested_train_sizes, key=ALL_TRAIN_SIZES_ORDERED.index)
        print(f"Custom train sizes requested: {TRAIN_SIZES}")
    if repetitions is not None:
        if repetitions <= 0:
            raise ValueError("repetitions must be positive")
        N_REPETITIONS = repetitions
    if seed_start < 0:
        raise ValueError("seed_start must be non-negative")

    # TabPFN parameters
    N_ESTIMATORS = 3
    N_PERMUTATIONS = 3
    TEMPERATURE = 1.0
    
    # Seed configuration - automatic seed management for fair comparison
    # Each repetition gets a unique seed: 0, 1, 2, ..., N_REPETITIONS-1
    # This ensures: (1) Same seed = same data across algorithms, (2) Reproducible datasets

    # Output configuration (stable filenames, resume-friendly)
    base_results_dir = Path("causal_experiments/custom_scm_experiment/comparison_experiment")
    results_dir_name = "results"
    if noise_level is not None:
        results_dir_name = f"results_{_format_noise_tag(noise_level)}"
    output_dir = base_results_dir / results_dir_name
    train_size_suffix = f"_ts{TRAIN_SIZES[0]}" if len(TRAIN_SIZES) == 1 else ""
    if algorithm_filter and column_order_filter:
        output_file = output_dir / f"results_{algorithm_filter}_{column_order_filter}{train_size_suffix}.csv"
    elif algorithm_filter:
        output_file = output_dir / f"results_{algorithm_filter}{train_size_suffix}.csv"
    elif column_order_filter:
        output_file = output_dir / f"comparison_results_{column_order_filter}{train_size_suffix}.csv"
    else:
        output_file = output_dir / f"comparison_results{train_size_suffix}.csv"
    print(f"Results will be saved to: {output_file}")


    # ======================
    # DATA GENERATION
    # ======================

    # Choose data type: numeric or mixed (with categorical)
    USE_CATEGORICAL = False  # Set to True for mixed data with categorical features
    default_noise_level = 0.3 if USE_CATEGORICAL else 1e-5
    noise_level_used = default_noise_level if noise_level is None else noise_level
    print(f"Using SCM noise level: {noise_level_used:g}")

    X_all, column_names, categorical_cols, dag_static, cpdag_static = _load_custom_static_dataset(
        use_categorical=USE_CATEGORICAL,
        n_samples=6000,
        noise_level=noise_level,
    )

    cpdag_indep_test = "hybrid" if USE_CATEGORICAL else "fisherz"
    print(f"Independence test for CPDAG discovery: {cpdag_indep_test}")


    # ======================
    # MODEL INITIALIZATION
    # ======================


    # Note: TabPFN models will be created fresh for each experiment to avoid state contamination

    # ======================
    # EXPERIMENTAL CONFIGURATIONS
    # ======================

    configs = get_experimental_configs(use_categorical=USE_CATEGORICAL)
    if dag_static != configs["dag"]["dag"]:
        raise ValueError("Static dataset DAG does not match experimental configuration DAG")
    if cpdag_static != configs["cpdag_minimal"]["cpdag"]:
        raise ValueError("Static dataset CPDAG does not match experimental configuration CPDAG")


    # Filter algorithms if specified (for accurate total_experiments calculation)
    if algorithm_filter:
        if algorithm_filter not in configs:
            available_algorithms = list(configs.keys())
            raise ValueError(f"Algorithm '{algorithm_filter}' not found. Available: {available_algorithms}")
        configs = {algorithm_filter: configs[algorithm_filter]}
        print(f"Running ONLY algorithm: {algorithm_filter}")
    else:
        print(f"Running ALL algorithms: {list(configs.keys())}")
    
    # Print column order filter info
    if column_order_filter:
        print(f"Using specific column ordering for vanilla/CPDAG: {column_order_filter}")
    else:
        print(f"Using ALL column orderings for vanilla/CPDAG algorithms")

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
    total_configs = 0
    for config in configs.values():
        orderings_count = len(config["orderings"])
        # If column_order_filter is specified, count only that ordering
        if column_order_filter:
            # Check if the filtered ordering exists in this config
            ordering_names = [order_tuple[0] for order_tuple in config["orderings"]]
            if column_order_filter in ordering_names:
                orderings_count = 1
            else:
                orderings_count = 0
        total_configs += orderings_count
    
    total_experiments = len(TRAIN_SIZES) * N_REPETITIONS * total_configs


    # ======================
    # LOAD EXISTING RESULTS IF ANY
    # ======================
    
    results: list[dict[str, Any]] = []
    completed_experiments: set[tuple] = set()

    # Resume support: scan all CSVs in results dir (including old timestamped ones)
    loaded_rows = 0
    for csv_path in sorted(output_dir.glob("*.csv")):
        try:
            existing_df = pd.read_csv(csv_path)
        except Exception:
            continue
        existing_rows = cast(list[dict[str, Any]], [dict(row) for row in existing_df.to_dict('records')])
        for row in existing_rows:
            # Optional filters: respect CLI filters when present
            if algorithm_filter and row.get('algorithm') != algorithm_filter:
                continue
            if column_order_filter and row.get('column_order') != column_order_filter:
                continue
            if row.get('train_size') not in TRAIN_SIZES:
                continue
            seed_for_key_raw = row.get('seed')
            try:
                seed_for_key = int(seed_for_key_raw)
            except (TypeError, ValueError):
                seed_for_key = seed_for_key_raw
            exp_key = (row.get('algorithm'), row.get('column_order'), row.get('train_size'), seed_for_key)
            if exp_key in completed_experiments:
                continue
            completed_experiments.add(exp_key)
            results.append(row)
        loaded_rows += len(existing_rows)
    if loaded_rows > 0:
        print(f"Resume: scanned {loaded_rows} rows from existing CSVs; {len(completed_experiments)} unique experiments will be skipped.")
    
    # ======================
    # MAIN EXPERIMENTAL LOOP
    # ======================

    start_time = time.time()

    experiment_counter = 0
    crashed_seeds = set()  # Track seeds that trigger infinity bug
    crashed_seeds_data = []  # Track detailed crash information for CSV
    
    # Recovery: scan all CSVs for crashed seeds as well
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
                seed_for_crash_raw = row.get('seed')
                try:
                    seed_for_crash = int(seed_for_crash_raw)
                except (TypeError, ValueError):
                    seed_for_crash = seed_for_crash_raw
                crashed_seeds.add(seed_for_crash)
    if crashed_seeds:
        print(f"Recovered {len(crashed_seeds)} crashed seeds from existing results: {sorted(crashed_seeds)}")

    # ======================
    # DATASET PREPARATION AND STORAGE  
    n_total = X_all.shape[0]
    if TEST_SIZE >= n_total:
        raise ValueError("TEST_SIZE must be smaller than dataset size")

    TEST_SEED = 2000
    rng = np.random.default_rng(TEST_SEED)
    test_indices = rng.choice(n_total, size=TEST_SIZE, replace=False)
    mask = np.ones(n_total, dtype=bool)
    mask[test_indices] = False
    train_pool = X_all[mask]
    global_test_set = X_all[test_indices]

    global_test_df = pd.DataFrame({
        **dict(zip(column_names, [global_test_set[:, i] for i in range(global_test_set.shape[1])])),
    })

    print(f"Global test set selected with seed {TEST_SEED}: {len(global_test_set)} samples")

    save_global_test_set(global_test_set, output_dir, column_names)

    # ======================
    # PRE-GENERATE TRAIN SUBSETS PER LINEAR SEED
    # ======================
    all_splits: dict[int, np.ndarray] = {}
    train_dataset_paths: dict[int, str] = {}
    seed_to_metadata: dict[int, tuple[int, int]] = {}
    cached_datasets = 0
    regenerated_datasets = 0

    print("Generating train sets with linear seed counter...")

    train_size_to_seeds = {
        train_size: [
            seed_start + ALL_TRAIN_SIZES_ORDERED.index(train_size) * N_REPETITIONS + rep_idx
            for rep_idx in range(N_REPETITIONS)
        ]
        for train_size in TRAIN_SIZES
    }
    total_datasets_needed = sum(len(seeds) for seeds in train_size_to_seeds.values())
    min_seed = min(seed for seeds in train_size_to_seeds.values() for seed in seeds)
    max_seed = max(seed for seeds in train_size_to_seeds.values() for seed in seeds)
    print(f"Will generate {total_datasets_needed} unique datasets using seeds {min_seed} to {max_seed}")

    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    for train_size in TRAIN_SIZES:
        for rep_idx, seed in enumerate(train_size_to_seeds[train_size]):
            seed_to_metadata[seed] = (train_size, rep_idx + 1)

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
                                loaded_successfully = True
                            else:
                                print(
                                    f"Ignoring stored train set {dataset_file.name} due to schema mismatch."
                                )
                        else:
                            print(
                                f"Ignoring stored train set {dataset_file.name} due to missing/invalid metadata."
                            )
                except Exception as exc:
                    print(f"Failed to load cached train set {dataset_file.name}: {exc}")

            if loaded_successfully:
                cached_datasets += 1
                continue

            rng_split = np.random.default_rng(seed)
            if train_size > len(train_pool):
                sampled_indices = rng_split.choice(len(train_pool), size=train_size, replace=True)
            else:
                sampled_indices = rng_split.choice(len(train_pool), size=train_size, replace=False)
            X_train = train_pool[sampled_indices]
            all_splits[seed] = X_train

            if save_datasets:
                train_path = save_train_set_per_split(
                    X_train=X_train,
                    train_size=train_size,
                    seed=seed,
                    output_dir=output_dir,
                    column_names=column_names,
                    overwrite=True,
                    dataset_seed=seed,
                    metadata={
                        "sampling_strategy": "observational_random_subset",
                        "seed": seed,
                    },
                )
                train_dataset_paths[seed] = train_path
            elif dataset_file.exists():
                train_dataset_paths[seed] = str(dataset_file)
            else:
                train_dataset_paths[seed] = ""

            regenerated_datasets += 1

            generated_count = cached_datasets + regenerated_datasets
            if generated_count % 20 == 0:
                print(f"Generated {generated_count} splits")

    if cached_datasets:
        print(f"   Cached datasets loaded for {cached_datasets} seeds")
    if regenerated_datasets:
        print(f"   Regenerated datasets for {regenerated_datasets} seeds")

    # Now run experiments with proper seed management and fair comparison
    for algorithm, config in configs.items():
        orderings = config["orderings"]
        dag = config["dag"]
        cpdag = config["cpdag"]
        dag_for_ordering = config.get("dag_for_ordering")

        for column_order_tuple in orderings:
            # Extract column order name from tuple (order_indices pre-calculated but not used)
            column_order, _ = column_order_tuple
            
            # Filter column order if specified
            if column_order_filter and column_order != column_order_filter:
                continue
            
            for train_size in TRAIN_SIZES:
                for rep_idx, seed in enumerate(train_size_to_seeds[train_size]):
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

                    # Simplified logging with linear seed approach
                    train_size_from_metadata, rep_from_metadata = seed_to_metadata.get(
                        seed, (train_size, repetition)
                    )
                    print(f"[{experiment_counter}/{total_experiments}] {algorithm}-{column_order} | train_size={train_size} | rep={repetition}/{N_REPETITIONS} | seed={seed} | (dataset: train_size={train_size_from_metadata}, rep={rep_from_metadata})")

                    # Print progress every 50 experiments with optional memory monitoring
                    if experiment_counter % 50 == 0 and experiment_counter >= 10:
                        elapsed = time.time() - start_time
                        progress_pct = experiment_counter / total_experiments * 100
                        eta_seconds = (elapsed / experiment_counter) * (total_experiments - experiment_counter)
                        
                        # Optional memory monitoring (doesn't interrupt if psutil unavailable)
                        try:
                            import psutil
                            process = psutil.Process()
                            memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
                            print(f" Progress: {experiment_counter}/{total_experiments} ({progress_pct:.1f}%), ETA: {eta_seconds/60:.1f} min, RAM: {memory_gb:.1f}GB")
                        except ImportError:
                            print(f" Progress: {experiment_counter}/{total_experiments} ({progress_pct:.1f}%), ETA: {eta_seconds/60:.1f} min")

                    try:
                        # Get pre-loaded dataset using linear seed (much simpler!)
                        X_train_original = all_splits[seed]

                        # Determine CPDAG source robustly by algorithm name
                        if algorithm.startswith("cpdag_"):
                            if "discovered" in algorithm:
                                print(
                                    f"  → Discovering CPDAG for seed={seed}, "
                                    f"train_size={train_size}"
                                )
                                cpdag_to_use = discover_cpdag_from_data(
                                    X_train_original,
                                    column_names,
                                    categorical_cols,
                                    USE_CATEGORICAL,
                                    true_dag=dag,  # Pass the DAG from config
                                    indep_test=cpdag_indep_test,
                                    hybrid_params={
                                        "k": 5,
                                        "permutations": 500,
                                        "random_state": seed,
                                    } if cpdag_indep_test == "hybrid" else None,
                                )
                            elif "minimal" in algorithm:
                                # Use CPDAG provided by config (ground truth / ideal)
                                cpdag_to_use = cpdag
                            else:
                                # Default fallback: use CPDAG from config if present
                                cpdag_to_use = cpdag
                        else:
                            cpdag_to_use = None  # No CPDAG for vanilla or dag

                        # Prepare data using algorithm-specific functions (cleaner approach)
                        updated_categorical_features = None
                        if algorithm == "vanilla":
                            if dag_for_ordering is None:
                                raise ValueError(f"dag_for_ordering is required for vanilla algorithm but got None")
                            # Convert categorical column names to indices for prepare_vanilla_data
                            categorical_indices = [column_names.index(col) for col in categorical_cols] if categorical_cols else None
                            X_train_prepared, column_ordering_used, updated_categorical_features = prepare_vanilla_data(
                                X_train_original, column_order, dag_for_ordering, column_names, categorical_indices
                            )
                            dag_prepared, cpdag_prepared = None, None
                            
                        elif algorithm == "dag":
                            if dag is None:
                                raise ValueError(f"DAG is required for dag algorithm but got None")
                            
                            # Handle optional column reordering for DAG like vanilla/CPDAG
                            if column_order != "original":
                                if dag_for_ordering is None:
                                    raise ValueError(f"dag_for_ordering is required for DAG algorithm but got None")
                                # We reuse vanilla's reordering utility to compute the new column order
                                categorical_indices = [column_names.index(col) for col in categorical_cols] if categorical_cols else None
                                X_train_reordered, column_ordering_used, updated_categorical_features = prepare_vanilla_data(
                                    X_train_original, column_order, dag_for_ordering, column_names, categorical_indices
                                )
                                # Remap DAG indices to the new column order
                                if column_ordering_used is None:
                                    raise ValueError("column_ordering_used should not be None when reordering is requested")
                                old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(column_ordering_used)}
                                dag_reordered = {}
                                for old_child, old_parents in dag.items():
                                    new_child = old_to_new.get(old_child)
                                    if new_child is None:
                                        continue
                                    dag_reordered[new_child] = [old_to_new[p] for p in old_parents if p in old_to_new]
                                # Prepare data for DAG run
                                if column_ordering_used is None:
                                    raise ValueError("column_ordering_used should not be None when reordering is requested")
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
                            if column_order != "original":
                                # Use the same DAG as vanilla does
                                if dag_for_ordering is None:
                                    raise ValueError(f"dag_for_ordering is required for CPDAG algorithm but got None")
                                categorical_indices = [column_names.index(col) for col in categorical_cols] if categorical_cols else None
                                X_train_reordered, column_ordering_used, updated_categorical_features = prepare_vanilla_data(
                                    X_train_original, column_order, dag_for_ordering, column_names, categorical_indices
                                )
                                
                                # Reorder CPDAG structure to match new column ordering
                                # Create mapping from old indices to new indices
                                if column_ordering_used is None:
                                    raise ValueError("column_ordering_used should not be None when reordering is requested")
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
                                
                                if column_ordering_used is None:
                                    raise ValueError("column_ordering_used should not be None when reordering is requested")
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
                            reordered_graph_dict = cpdag_prepared if column_order != "original" and cpdag_prepared is not None else None
                        else:  # algorithm == "dag"
                            graph_dict = dag
                            reordered_graph_dict = dag_prepared if column_order != "original" and dag_prepared is not None else None

                        # Create result row
                        result_row = create_result_row(
                            algorithm=algorithm,
                            column_order=column_order,
                            graph_dict=graph_dict,
                            train_size=train_size,
                            seed=seed,  # Use linear seed directly - much simpler!
                            repetition=repetition,  # Using proper repetition counter (1-N_REPETITIONS per train_size)
                            categorical_cols=categorical_cols,
                            column_names=column_names,
                            metrics_results=metrics,
                            reordered_graph_dict=reordered_graph_dict,
                            column_ordering_used=column_ordering_used
                        )
                        result_row["noise_level"] = noise_level_used

                        
                        # Get actual column order based on algorithm and ordering strategy
                        if column_order != "original" and column_ordering_used is not None and (
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
                            # Always reference the base train dataset using linear seed
                            # Optionally save reordered dataset for verification
                            if save_reordered and column_ordering_used is not None:
                                train_data_modified = not np.array_equal(X_train_prepared, X_train_original)
                                if train_data_modified:
                                    base_train_path = train_dataset_paths.get(
                                        seed,
                                        str(output_dir / "datasets" / f"train_ts{train_size}_s{seed}.npz")
                                    )
                                    reordered_path = save_reordered_dataset(
                                        X_train_reordered=X_train_prepared,
                                        algorithm=algorithm,
                                        column_order=column_order,
                                        train_size=train_size,
                                        seed=seed,
                                        output_dir=output_dir,
                                        reordered_column_names=actual_column_order,
                                        base_train_path=base_train_path  # Use linear seed directly
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
                            print(
                                f" INFINITY BUG: {algorithm}-{column_order} "
                                f"seed={seed} | SKIPPING"
                            )
                            crashed_seeds.add(seed)
                            # Record detailed crash information
                            crashed_seeds_data.append({
                                'seed': seed,  # Linear seed - simple!
                                'algorithm': algorithm,
                                'column_order': column_order,
                                'train_size': train_size,
                                'repetition': repetition,
                                'error_type': 'infinity_bug',
                                'error_message': error_msg[:200],  # Truncate long error messages
                                'timestamp': pd.Timestamp.now().isoformat()
                            })
                        else:
                            print(
                                f" OTHER ERROR in experiment {algorithm}-{column_order} "
                                f"seed={seed}: {e}"
                            )
                            # Record other errors too
                            crashed_seeds_data.append({
                                'seed': seed,  # Linear seed - simple!
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

                    # No additional seed increment here; already advanced at loop start
                
                # SCIENTIFIC IMPROVEMENT: Do NOT increment base_seed
                # We want same seeds across train_sizes to isolate effect of sample size
                # base_seed += N_REPETITIONS  # REMOVED

    # ======================
    # SAVE FINAL RESULTS
    # ======================

    total_time = time.time() - start_time
    print(f"\nExperiment completed in {total_time/60:.1f} minutes")

    # Save final results
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
            "For vanilla, DAG, and all CPDAG variants, specify column ordering: original, topological, worst"
        ),
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=None,
        help=(
            "Noise level for the custom SCM (e.g., 1e-2). "
            "If set, expects noise-tagged datasets and writes results to a noise-specific directory."
        ),
    )
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        help="Override train sizes with one or more positive integers. Example: --train-sizes 20 50 100",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Number of repetitions per train size. Defaults to 120 (or 10 in --test).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Starting seed for the linear seed schedule (default: 0).",
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
        noise_level=args.noise_level,
        train_sizes=args.train_sizes,
        repetitions=args.repetitions,
        seed_start=args.seed_start,
    )
