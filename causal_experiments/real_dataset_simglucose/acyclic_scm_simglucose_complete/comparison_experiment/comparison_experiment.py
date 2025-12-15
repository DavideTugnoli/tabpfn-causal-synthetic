"""Comparison experiment for the real SimGlucose static SCM dataset.

This script mirrors the custom SCM comparison experiment but runs on the
SimGlucose static SCM snapshots and uses the provided DAG/CPDAG structures
stored alongside the dataset.

Dataset and graphs:
- CSV: causal_experiments/real_dataset_simglucose/acyclic_scm_simglucose_complete/generated_static_scm/simglucose_static_scm_with_params_6000.csv
- DAG/CPDAG JSON: causal_experiments/real_dataset_simglucose/acyclic_scm_simglucose_complete/generated_static_scm/simglucose_static_scm_with_params.graphs.json

Algorithms (only these):
- vanilla
- dag
- cpdag_discovered  (CPDAG discovered from train split)
- cpdag_minimal     (CPDAG minimal from JSON)

Orderings for all algorithms: original, topological, worst

Parameters match the custom experiment (train sizes, test size, repetitions).

Usage examples:
    uv run causal_experiments/real_dataset_simglucose/acyclic_scm_simglucose_complete/comparison_experiment/comparison_experiment.py
    uv run causal_experiments/real_dataset_simglucose/acyclic_scm_simglucose_complete/comparison_experiment/comparison_experiment.py --test
    uv run causal_experiments/real_dataset_simglucose/acyclic_scm_simglucose_complete/comparison_experiment/comparison_experiment.py --algorithm vanilla
    uv run causal_experiments/real_dataset_simglucose/acyclic_scm_simglucose_complete/comparison_experiment/comparison_experiment.py --algorithm cpdag_discovered --column-order topological

"""
from __future__ import annotations

import os
# Set environment for maximal determinism before importing torch
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import time
from pathlib import Path
import json as _json
import inspect
from typing import Any, List, cast

import numpy as np
import pandas as pd
import torch

# Fix imports - add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from causal_experiments.utils import (
    create_result_row,
    discover_cpdag_from_data,
    identify_vanilla_duplicates,
    copy_vanilla_duplicate_result,
    prepare_vanilla_data,
    prepare_dag_data,
    prepare_cpdag_data,
    save_global_test_set,
    save_train_set_per_split,
    load_global_test_set,
    save_reordered_dataset,
    save_synthetic_data,
    save_results_to_csv,
    # Determinism utilities
    setup_determinism,
    set_experiment_seeds,
)
from causal_experiments.utils.metrics import FaithfulDataEvaluator
from causal_experiments.utils.dag_utils import get_ordering_strategies
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised


def _verify_nested_train_sets(train_sets: dict[int, np.ndarray]) -> bool:
    """Check that smaller train splits are prefixes of the largest split."""
    if not train_sets:
        return False
    ordered_sizes = sorted(train_sets.keys())
    max_size = ordered_sizes[-1]
    max_data = train_sets[max_size]
    if max_data.shape[0] != max_size:
        return False
    for size in ordered_sizes[:-1]:
        current = train_sets[size]
        if current.shape[0] != size:
            return False
        if not np.allclose(current, max_data[:size]):
            return False
    return True


# Global flag to print device info only once
_device_info_printed = False

# Reuse cached train splits when they were generated with the independent sampling
# strategy. Otherwise, regenerate them to ensure consistent batches across
# configurations.
_REUSE_CACHED_TRAIN_SPLITS = True


def _load_simglucose_dataset_and_graphs() -> tuple[np.ndarray, list[str], dict[int, list[int]], dict[int, dict[str, list[int]]]]:
    """Load dataset CSV and graphs JSON, returning arrays and canonical graph dicts.

    Returns:
        X_all: Full dataset as numpy array
        column_names: Column names in canonical order
        dag_dict: DAG as {child: [parents]}
        cpdag_minimal: CPDAG minimal as {node: {"parents": [...], "undirected": [...]}}
    """
    base_dir = Path(
        "causal_experiments/real_dataset_simglucose/acyclic_scm_simglucose_complete/generated_static_scm"
    )
    csv_path = base_dir / "simglucose_static_scm_with_params_6000.csv"
    graphs_path = base_dir / "simglucose_static_scm_with_params.graphs.json"

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at {csv_path}")
    if not graphs_path.exists():
        raise FileNotFoundError(f"Graphs JSON not found at {graphs_path}")

    with graphs_path.open("r") as f:
        graphs = _json.load(f)

    column_names: list[str] = graphs["column_names"]

    # Convert DAG keys/values to integers
    dag_raw: dict[str, list[int]] = graphs["dag_dict"]
    dag_dict: dict[int, list[int]] = {int(k): [int(v) for v in vals] for k, vals in dag_raw.items()}

    # Convert CPDAG minimal to canonical typing with int keys
    cpdag_raw: dict[str, dict[str, list[int]]] = graphs["cpdag_dict"]
    cpdag_minimal: dict[int, dict[str, list[int]]] = {
        int(k): {
            "parents": [int(x) for x in v.get("parents", [])],
            "undirected": [int(x) for x in v.get("undirected", [])],
        }
        for k, v in cpdag_raw.items()
    }

    # Load CSV and ensure column order matches graphs.json
    df = pd.read_csv(csv_path)
    # Reorder/select columns to exactly match the JSON
    df = df[column_names]
    X_all = df.to_numpy(dtype=float)

    return X_all, column_names, dag_dict, cpdag_minimal


def save_crashed_seeds_to_csv(crashed_seeds_data: List[dict], output_file: Path):
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
    """Run a single experimental configuration using GenerateSyntheticDataExperiment."""
    try:
        clf = TabPFNClassifier(n_estimators=n_estimators)
        reg = TabPFNRegressor(n_estimators=n_estimators)

        # If we have patient_id as categorical with many classes, increase the threshold
        # so that unsupervised uses the classification path for that column.
        try:
            if hasattr(clf, 'max_num_classes_'):
                if clf.max_num_classes_ < 512:
                    clf.max_num_classes_ = 512
            else:
                setattr(clf, 'max_num_classes_', 512)
            print(f"🔧 TabPFNClassifier max_num_classes_ set to {getattr(clf, 'max_num_classes_', 'n/a')}")
        except Exception as _e:
            print(" Unable to set max_num_classes_ on classifier; proceeding with default.")

        global _device_info_printed
        if not _device_info_printed:
            print(f" CUDA available: {torch.cuda.is_available()}")
            print(f" CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
            print(f" TabPFN Classifier device: {clf.device}")
            print(f" TabPFN Regressor device: {reg.device}")
            from tabpfn_extensions.utils import get_device
            print(f" get_device('auto') returns: {get_device('auto')}")
            _device_info_printed = True

        model_unsupervised = unsupervised.TabPFNUnsupervisedModel(
            tabpfn_clf=clf,
            tabpfn_reg=reg,
        )

        categorical_indices_for_run: list[int] = []
        if categorical_cols:
            if updated_categorical_features is not None:
                categorical_indices_for_run = list(updated_categorical_features)
            else:
                categorical_indices_for_run = [column_names.index(col) for col in categorical_cols]
            model_unsupervised.set_categorical_features(categorical_indices_for_run)
        elif updated_categorical_features is not None:
            categorical_indices_for_run = list(updated_categorical_features)

        X_tensor = torch.tensor(X_train, dtype=torch.float32)

        exp_synthetic = unsupervised.experiments.GenerateSyntheticDataExperiment(
            task_type="unsupervised"
        )

        # CPDAG v1 only in this experiment
        cpdag_to_pass = cpdag

        if column_order != "original" and column_ordering_used is not None and (
            algorithm == "vanilla" or algorithm == "dag" or algorithm.startswith("cpdag_")
        ):
            feature_names_for_run = [column_names[i] for i in column_ordering_used]
        else:
            feature_names_for_run = column_names

        exp_synthetic.run(
            tabpfn=model_unsupervised,
            X=X_tensor,
            y=None,
            attribute_names=feature_names_for_run,
            temp=temp,
            n_samples=len(test_df),
            n_permutations=n_permutations,
            dag=dag,
            cpdag=cpdag_to_pass,
            causal_structures_last=causal_structures_last,
            indices=list(range(X_train.shape[1])),
            categorical_features=categorical_indices_for_run,
        )

        real_df = exp_synthetic.data_real
        synthetic_df = exp_synthetic.data_synthetic
        if 'real_or_synthetic' in real_df.columns:
            real_df = real_df.drop('real_or_synthetic', axis=1)
        if 'real_or_synthetic' in synthetic_df.columns:
            synthetic_df = synthetic_df.drop('real_or_synthetic', axis=1)

        if column_order != "original" and column_ordering_used is not None and (
            algorithm == "vanilla" or algorithm == "dag" or algorithm.startswith("cpdag_")
        ):
            reverse_ordering = [column_ordering_used.index(i) for i in range(len(column_ordering_used))]
            synthetic_data_reordered = np.zeros_like(exp_synthetic.synthetic_X)
            for orig_idx, reordered_idx in enumerate(reverse_ordering):
                synthetic_data_reordered[:, orig_idx] = exp_synthetic.synthetic_X[:, reordered_idx]
            synthetic_df = pd.DataFrame({
                **dict(zip(column_names, [synthetic_data_reordered[:, i] for i in range(len(column_names))])),
            })

        # Optional: remap categorical action back to original label space (e.g., 0/1/2 -> 30/60/90)
        if "action_CHO_g" in column_names and "action_CHO_g" in synthetic_df.columns:
            try:
                # Derive canonical label order from real data (test_df) in original value space
                label_values = sorted(pd.Series(test_df["action_CHO_g"]).unique().tolist())
                index_to_label = {i: label_values[i] for i in range(len(label_values))}
                # Map predicted class indices (0..K-1) back to original labels
                def _remap_action_vals(s: pd.Series) -> pd.Series:
                    vals = pd.to_numeric(s, errors="coerce").fillna(-1).astype(int)
                    return vals.map(index_to_label).fillna(s).astype(float)
                synthetic_df["action_CHO_g"] = _remap_action_vals(synthetic_df["action_CHO_g"])
            except Exception as _e:
                # Non-fatal: proceed without remapping
                pass

        evaluator = FaithfulDataEvaluator()
        metrics = evaluator.evaluate(
            real_data=test_df,
            synthetic_data=synthetic_df,
            categorical_columns=categorical_cols,
            k_for_kmarginal=2,
            random_seed=seed,
        )

        if "propensity_metrics" in metrics:
            propensity = metrics.pop("propensity_metrics")
            if isinstance(propensity, dict):
                for key, value in propensity.items():
                    metrics[f"propensity_{key}"] = value

        return metrics, synthetic_df

    except Exception as e:
        print(f"Error in run_single_experiment: {e}")
        import traceback
        print(traceback.format_exc())
        return {
            "correlation_matrix_difference": -1.0,
            "k_marginal_tvd": -1.0,
            "nnaa": -1.0,
        }, pd.DataFrame()


def main(
    test_mode: bool = False,
    save_every: int = 100,
    save_datasets: bool = False,
    save_reordered: bool = False,
    save_synthetic: bool = False,
    algorithm_filter: str | None = None,
    column_order_filter: str | None = None,
    skip_seeds: set[int] | None = None,
    train_sizes: list[int] | None = None,
    repetitions: int | None = None,
):
    """Run comparison experiment on real SimGlucose dataset."""
    # ======================
    # PARAMETERS
    # ======================
    if repetitions is not None:
        if repetitions <= 0:
            raise ValueError("`--repetitions` must be a positive integer.")

    if train_sizes is not None:
        if len(train_sizes) == 0:
            raise ValueError("`--train-sizes` requires at least one positive integer.")
        if any(size <= 0 for size in train_sizes):
            raise ValueError("`--train-sizes` values must be positive integers.")
        TRAIN_SIZES = sorted(set(train_sizes))
        print(f" Custom train sizes requested: {TRAIN_SIZES}")
        TEST_SIZE = 2000 if not test_mode else 100
        default_reps = 2 if test_mode else 155
        N_REPETITIONS = repetitions if repetitions is not None else default_reps
    elif test_mode:
        TRAIN_SIZES = [20]
        TEST_SIZE = 100
        default_reps = 2
        N_REPETITIONS = repetitions if repetitions is not None else default_reps
    else:
        TRAIN_SIZES = [20, 50, 100, 200, 500]
        TEST_SIZE = 2000
        default_reps = 155
        N_REPETITIONS = repetitions if repetitions is not None else default_reps

    N_ESTIMATORS = 3
    N_PERMUTATIONS = 3
    TEMPERATURE = 1.0

    # Linear seed approach: 0,1,2,3,4,5... (same seed = same dataset for all algorithms = fair comparison)
    # No BASE_SEED needed - use simple linear counter starting from 0
    skip_seeds_set: set[int] = set(skip_seeds or [])
    if skip_seeds_set:
        print(f"  Skip seeds requested: {sorted(skip_seeds_set)}")

    # ======================
    # LOAD DATASET + GRAPHS
    # ======================
    X_all, column_names, dag_dict, cpdag_minimal = _load_simglucose_dataset_and_graphs()
    # Categorical columns inferred by schema (align evaluator with TabPFN's internal handling)
    categorical_cols: list[str] = []
    for cat in ["patient_id", "action_CHO_g"]:
        if cat in column_names:
            categorical_cols.append(cat)
    if categorical_cols:
        print(f" categorical_cols set to {categorical_cols}")
    else:
        print(" No categorical columns declared explicitly in evaluator.")
    cpdag_indep_test = "kci"
    print(f" Independence test for CPDAG discovery: {cpdag_indep_test}")

    # ======================
    # OUTPUT CONFIG
    # ======================
    output_dir = Path(
        "causal_experiments/real_dataset_simglucose/acyclic_scm_simglucose_complete/comparison_experiment/results"
    )
    
    # Build base filename without train_size suffix
    if algorithm_filter and column_order_filter:
        base_filename = f"results_{algorithm_filter}_{column_order_filter}"
    elif algorithm_filter:
        base_filename = f"results_{algorithm_filter}"
    elif column_order_filter:
        base_filename = f"comparison_results_{column_order_filter}"
    else:
        base_filename = "comparison_results"
    
    # For single train_size: check if base file exists and train_size is 20, if so use it (for extension)
    # Otherwise use train_size suffix to avoid conflicts with parallel jobs
    if len(TRAIN_SIZES) == 1:
        train_size = TRAIN_SIZES[0]
        base_file = output_dir / f"{base_filename}.csv"
        # Only use base file for train_size=20 to extend existing results
        # Other train sizes use suffixed files to avoid conflicts
        if train_size == 20 and base_file.exists():
            # Use existing file to extend it (only for train_size=20)
            output_file = base_file
            print(f" Found existing CSV: {output_file} - will extend it for train_size=20")
        else:
            # Use train_size suffix for new file (parallel jobs or train_size != 20)
            output_file = output_dir / f"{base_filename}_ts{train_size}.csv"
            print(f" Results will be saved to new file: {output_file}")
    else:
        # Multiple train sizes: use base filename
        output_file = output_dir / f"{base_filename}.csv"
        print(f" Results will be saved to: {output_file}")

    # ======================
    # EXPERIMENTAL CONFIGS (subset only)
    # ======================
    orderings_dict = get_ordering_strategies(dag_dict)
    configs: dict[str, dict[str, Any]] = {
        "vanilla": {
            "orderings": [
                ("original", orderings_dict["original"]),
                ("topological", orderings_dict["topological"]),
                ("reverse_topological", orderings_dict["reverse_topological"]),
            ],
            "dag": None,
            "cpdag": None,
            "dag_for_ordering": dag_dict,
        },
        "dag": {
            "orderings": [
                ("topological", orderings_dict["topological"]),
            ],
            "dag": dag_dict,
            "cpdag": None,
            "dag_for_ordering": dag_dict,
        },
        "cpdag_discovered": {
            "orderings": [
                ("original", orderings_dict["original"]),
            ],
            "dag": None,
            # Discovery is per-split; cpdag set at runtime
            "cpdag": None,
            "dag_for_ordering": dag_dict,
        },
        "cpdag_minimal": {
            "orderings": [
                ("original", orderings_dict["original"]),
            ],
            "dag": None,
            "cpdag": cpdag_minimal,
            "dag_for_ordering": dag_dict,
        },
    }

    # Filter algorithms if requested
    if algorithm_filter:
        if algorithm_filter not in configs:
            available_algorithms = list(configs.keys())
            raise ValueError(
                f"Algorithm '{algorithm_filter}' not found. Available: {available_algorithms}"
            )
        configs = {algorithm_filter: configs[algorithm_filter]}
        print(f" Running ONLY algorithm: {algorithm_filter}")
    else:
        print(f" Running algorithms: {list(configs.keys())}")

    if column_order_filter:
        print(f" Using specific column ordering: {column_order_filter}")
    else:
        print(" Using ALL column orderings")

    # Vanilla duplicate optimization
    vanilla_duplicates = identify_vanilla_duplicates(configs)

    # Count configs after filtering
    total_configs = 0
    for config in configs.values():
        orderings_count = len(config["orderings"])
        if column_order_filter:
            ordering_names = [order_tuple[0] for order_tuple in config["orderings"]]
            orderings_count = 1 if column_order_filter in ordering_names else 0
        total_configs += orderings_count

    total_experiments = len(TRAIN_SIZES) * N_REPETITIONS * total_configs

    # ======================
    # RESUME SUPPORT + CRASHED SEEDS RECOVERY
    # ======================
    results: list[dict[str, Any]] = []
    completed_experiments: set[tuple] = set()

    loaded_rows = 0
    for csv_path in sorted(output_dir.glob("*.csv")):
        if csv_path.name.startswith("crashed_seeds"):
            continue
        filename = csv_path.name
        if filename.startswith("crashed_seeds"):
            continue
        try:
            existing_df = pd.read_csv(csv_path)
        except Exception:
            continue
        existing_rows = cast(list[dict[str, Any]], [dict(row) for row in existing_df.to_dict('records')])
        for row in existing_rows:
            if 'reordered_graph_structure' not in row or row.get('reordered_graph_structure') in ("", None):
                row['reordered_graph_structure'] = 'no_reorder'
            if algorithm_filter and row.get('algorithm') != algorithm_filter:
                continue
            if column_order_filter and row.get('column_order') != column_order_filter:
                continue
            # Filter by train_size: only load results for train_sizes we're actually running
            row_train_size = row.get('train_size')
            if row_train_size is not None and row_train_size not in TRAIN_SIZES:
                continue
            exp_key = (row.get('algorithm'), row.get('column_order'), row.get('train_size'), row.get('seed'))
            if exp_key in completed_experiments:
                continue
            completed_experiments.add(exp_key)
            results.append(row)
        loaded_rows += len(existing_rows)
    if loaded_rows > 0:
        print(
            f" Resume: scanned {loaded_rows} rows; {len(completed_experiments)} unique experiments will be skipped."
        )

    crashed_seeds = set()
    crashed_seeds_data: list[dict] = []
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
    # BUILD GLOBAL TEST SET (fixed across all runs)
    # ======================
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

    if len(global_test_set) != TEST_SIZE:
        raise ValueError(
            f"Expected SimGlucose test set to contain exactly {TEST_SIZE} samples, got {len(global_test_set)}"
        )

    global_test_df = pd.DataFrame({
        **dict(zip(column_names, [global_test_set[:, i] for i in range(global_test_set.shape[1])])),
    })
    print(f"Global test set created with seed {TEST_SEED}: {len(global_test_set)} samples")

    loaded_test = load_global_test_set(output_dir)
    if loaded_test is not None:
        loaded_array, loaded_cols = loaded_test
        if list(loaded_cols) == column_names and loaded_array.shape[1] == len(column_names):
            global_test_set = loaded_array
            global_test_df = pd.DataFrame({
                **dict(zip(column_names, [global_test_set[:, i] for i in range(global_test_set.shape[1])])),
            })
            print(" Loaded global test set from existing datasets/global_test_set.npz")
        else:
            print("  Existing global test set schema mismatch; regenerating from source data.")

    save_global_test_set(global_test_set, output_dir, column_names)

    # ======================
    # PRE-GENERATE TRAIN SUBSETS PER LINEAR SEED
    # ======================
    all_splits: dict[int, np.ndarray] = {}
    train_dataset_paths: dict[int, str] = {}
    train_size_to_seeds: dict[int, list[int]] = {}
    seed_to_metadata: dict[int, tuple[int, int]] = {}

    # Define complete ordered list of all train sizes (used for seed offset calculation)
    # This allows parallelization without seed overlap between different train_size groups
    # Note: comparison_experiment uses [20, 50, 100, 200, 500] (no 1000)
    ALL_TRAIN_SIZES_ORDERED = [20, 50, 100, 200, 500]
    
    # Allocate unique linear seeds for each (train_size, repetition)
    # Each train_size gets seeds based on its position in the complete ordered list
    for train_size in TRAIN_SIZES:
        # Find position of this train_size in complete ordered list
        if train_size in ALL_TRAIN_SIZES_ORDERED:
            position = ALL_TRAIN_SIZES_ORDERED.index(train_size)
            # Calculate starting seed for this train_size: position * N_REPETITIONS
            start_seed = position * N_REPETITIONS
            seeds_for_size: list[int] = []
            for rep_idx in range(N_REPETITIONS):
                seed = start_seed + rep_idx
                seed_to_metadata[seed] = (train_size, rep_idx + 1)
                seeds_for_size.append(seed)
            train_size_to_seeds[train_size] = seeds_for_size
        else:
            print(f"  Warning: train_size {train_size} not found in complete list {ALL_TRAIN_SIZES_ORDERED}, using sequential allocation")
            # Fallback: sequential allocation for train sizes not in the complete list
            max_seed_used = max((max(seeds) for seeds in train_size_to_seeds.values()), default=-1)
            start_seed = max_seed_used + 1
            seeds_for_size: list[int] = []
            for rep_idx in range(N_REPETITIONS):
                seed = start_seed + rep_idx
                seed_to_metadata[seed] = (train_size, rep_idx + 1)
                seeds_for_size.append(seed)
            train_size_to_seeds[train_size] = seeds_for_size

    # Calculate total experiments needed
    all_seeds = [seed for seeds in train_size_to_seeds.values() for seed in seeds]
    total_experiments_needed = len(all_seeds)
    min_seed = min(all_seeds) if all_seeds else 0
    max_seed = max(all_seeds) if all_seeds else 0
    print(f"Using dynamic seed allocation based on position in complete list")
    print(f"   Complete ordered list: {ALL_TRAIN_SIZES_ORDERED}")
    print(f"   Seed range: {min_seed}..{max_seed} for {total_experiments_needed} experiments")
    print(f"   Seed allocation per train size:")
    for ts in TRAIN_SIZES:
        seeds = train_size_to_seeds[ts]
        if seeds:
            print(f"      train_size={ts}: {len(seeds)} seeds (range {seeds[0]}–{seeds[-1]})")

    if skip_seeds_set:
        total_valid_seeds = sum(
            sum(1 for seed in seeds if seed not in skip_seeds_set)
            for seeds in train_size_to_seeds.values()
        )
        total_experiments = total_valid_seeds * total_configs
        print(f"   ➖ Effective seeds per configuration after skips: {total_valid_seeds}")

    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    cached_datasets = 0
    regenerated_datasets = 0
    processed_datasets = 0

    for train_size in TRAIN_SIZES:
        for seed in train_size_to_seeds[train_size]:
            if skip_seeds_set and seed in skip_seeds_set:
                continue
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
                            loaded_array = data["X_train"]
                            loaded_cols = data["column_names"].tolist()
                            if list(loaded_cols) == column_names and loaded_array.shape[1] == len(column_names):
                                all_splits[seed] = loaded_array
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
                processed_datasets += 1
                continue

            # Generate new split
            rng = np.random.default_rng(seed)
            if train_size > len(train_pool):
                indices = rng.choice(len(train_pool), size=train_size, replace=True)
            else:
                indices = rng.choice(len(train_pool), size=train_size, replace=False)
            X_train = train_pool[indices]
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
            processed_datasets += 1

            if processed_datasets % 20 == 0:
                print(f"Generated {processed_datasets} splits")

    if cached_datasets:
        print(f"    Cached datasets loaded for {cached_datasets} seeds")
    if regenerated_datasets:
        print(f"    Regenerated datasets for {regenerated_datasets} seeds")

    # ======================
    # MAIN LOOP
    # ======================
    start_time = time.time()
    experiment_counter = len(completed_experiments)

    for algorithm, config in configs.items():
        orderings = config["orderings"]
        dag = config["dag"]
        cpdag_config = config["cpdag"]
        dag_for_ordering = config.get("dag_for_ordering")

        for column_order_tuple in orderings:
            column_order, _ = column_order_tuple
            if column_order_filter and column_order != column_order_filter:
                continue

            for train_size in TRAIN_SIZES:
                seeds_for_size = train_size_to_seeds[train_size]

                for rep_idx, seed in enumerate(seeds_for_size):
                    repetition = rep_idx + 1

                    exp_key = (algorithm, column_order, train_size, seed)
                    if exp_key in completed_experiments:
                        continue

                    if skip_seeds_set and seed in skip_seeds_set:
                        print(f"  Skipping {algorithm}-{column_order} seed={seed} (requested).")
                        continue

                    copied_result = copy_vanilla_duplicate_result(
                        results, algorithm, column_order, train_size, seed, vanilla_duplicates
                    )
                    if copied_result is not None:
                        results.append(copied_result)
                        continue

                    experiment_counter += 1
                    set_experiment_seeds(seed, include_cuda=True, verbose=False)
                    train_size_from_meta, rep_from_meta = seed_to_metadata.get(seed, (train_size, repetition))
                    print(
                        f" [{experiment_counter}/{total_experiments}] {algorithm}-{column_order} | train_size={train_size} | rep={repetition}/{N_REPETITIONS} | seed={seed} | (dataset: train_size={train_size_from_meta}, rep={rep_from_meta})"
                    )

                    try:
                        if seed not in all_splits:
                            raise RuntimeError(f"Missing split for seed={seed}, train_size={train_size}")
                        X_train_original = all_splits[seed]

                        # Prepare per algorithm
                        updated_categorical_features = None
                        cpdag_used = None  # track actual CPDAG used for logging
                        if algorithm == "vanilla":
                            if column_order != "original":
                                categorical_indices = [column_names.index(col) for col in categorical_cols] if categorical_cols else None
                                X_train_prepared, column_ordering_used, updated_categorical_features = prepare_vanilla_data(
                                    X_train_original, column_order, dag_for_ordering, column_names, categorical_indices
                                )
                            else:
                                X_train_prepared = X_train_original
                                column_ordering_used = None
                                updated_categorical_features = None
                            dag_prepared = None
                            cpdag_prepared = None

                        elif algorithm == "dag":
                            if dag is None:
                                raise ValueError("DAG is required for dag algorithm but got None")
                            if column_order != "original":
                                if dag_for_ordering is None:
                                    raise ValueError("dag_for_ordering is required for DAG reordering but got None")
                                categorical_indices = [column_names.index(col) for col in categorical_cols] if categorical_cols else None
                                X_train_reordered, column_ordering_used, updated_categorical_features = prepare_vanilla_data(
                                    X_train_original, column_order, dag_for_ordering, column_names, categorical_indices
                                )

                                old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(column_ordering_used)}
                                dag_reordered: dict[int, list[int]] = {}
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
                            # Determine CPDAG to use: discovered per split or minimal from config
                            if algorithm == "cpdag_discovered":
                                cpdag_to_use = discover_cpdag_from_data(
                                    X_train_original,
                                    column_names,
                                    categorical_cols,
                                    use_categorical=False,
                                    true_dag=None,
                                    indep_test=cpdag_indep_test,
                                    hybrid_params={
                                        "k": 5,
                                        "permutations": 500,
                                        "random_state": seed,
                                    } if cpdag_indep_test == "hybrid" else None,
                                )
                            else:
                                cpdag_to_use = cpdag_config
                            if cpdag_to_use is None:
                                raise ValueError(
                                    f"CPDAG is required for {algorithm} algorithm but got None"
                                )
                            cpdag_used = cpdag_to_use

                            if column_order != "original":
                                if dag_for_ordering is None:
                                    raise ValueError("dag_for_ordering is required for CPDAG reordering but got None")
                                categorical_indices = [column_names.index(col) for col in categorical_cols] if categorical_cols else None
                                X_train_reordered, column_ordering_used, updated_categorical_features = prepare_vanilla_data(
                                    X_train_original, column_order, dag_for_ordering, column_names, categorical_indices
                                )

                                old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(column_ordering_used)}
                                cpdag_reordered: dict[int, dict[str, list[int]]] = {}
                                for old_idx in range(len(column_names)):
                                    if old_idx in cpdag_to_use:
                                        new_idx = old_to_new[old_idx]
                                        cpdag_reordered[new_idx] = {
                                            "parents": [old_to_new[p] for p in cpdag_to_use[old_idx]["parents"] if p in old_to_new],
                                            "undirected": [old_to_new[u] for u in cpdag_to_use[old_idx]["undirected"] if u in old_to_new],
                                        }
                                    else:
                                        new_idx = old_to_new[old_idx]
                                        cpdag_reordered[new_idx] = {"parents": [], "undirected": []}

                                X_train_prepared, cpdag_prepared = prepare_cpdag_data(
                                    X_train_reordered, cpdag_reordered, [column_names[i] for i in column_ordering_used]
                                )
                            else:
                                X_train_prepared, cpdag_prepared = prepare_cpdag_data(
                                    X_train_original, cpdag_to_use, column_names
                                )
                                column_ordering_used = None
                                updated_categorical_features = None

                            dag_prepared = None

                        else:
                            raise ValueError(f"Unknown algorithm: {algorithm}")

                        test_df = global_test_df
                        causal_structures_last = config.get("causal_structures_last", False)

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

                        # Graph to log (for CPDAG, log the actual structure used)
                        if algorithm == "vanilla":
                            graph_dict = None
                        elif algorithm.startswith("cpdag_"):
                            graph_dict = cpdag_used
                        else:
                            graph_dict = dag

                        reordered_graph_dict = None
                        if column_order != "original" and column_ordering_used is not None:
                            if algorithm == "dag" and dag_prepared is not None:
                                reordered_graph_dict = dag_prepared
                            elif algorithm.startswith("cpdag_") and cpdag_prepared is not None:
                                reordered_graph_dict = cpdag_prepared

                        create_result_kwargs: dict[str, Any] = {
                            "algorithm": algorithm,
                            "column_order": column_order,
                            "graph_dict": graph_dict,
                            "train_size": train_size,
                            "seed": seed,
                            "repetition": repetition,
                            "categorical_cols": categorical_cols,
                            "column_names": column_names,
                            "metrics_results": metrics,
                        }
                        if "reordered_graph_dict" in inspect.signature(create_result_row).parameters:
                            create_result_kwargs["reordered_graph_dict"] = reordered_graph_dict
                        if "column_ordering_used" in inspect.signature(create_result_row).parameters:
                            create_result_kwargs["column_ordering_used"] = column_ordering_used
                        result_row = create_result_row(**create_result_kwargs)

                        # Resolve actual column order used
                        if column_order != "original" and column_ordering_used is not None and (
                            algorithm == "vanilla" or algorithm == "dag" or algorithm.startswith("cpdag_")
                        ):
                            actual_column_order = [column_names[i] for i in column_ordering_used]
                        else:
                            actual_column_order = column_names.copy()
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

                        if save_datasets:
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
                                        base_train_path=train_dataset_paths[seed],
                                    )
                                    result_row['reordered_dataset_path'] = reordered_path

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
                                metrics=metrics,
                            )
                            result_row['synthetic_data_path'] = synthetic_path

                        results.append(result_row)

                        if len(results) % save_every == 0:
                            # Filter results to only include those for the current train_sizes before saving
                            filtered_results = [
                                row for row in results
                                if row.get('train_size') is None or row.get('train_size') in TRAIN_SIZES
                            ]
                            save_results_to_csv(filtered_results, str(output_file))
                            if crashed_seeds_data:
                                algorithm_suffix = f"_{algorithm_filter}" if algorithm_filter else "_all_algorithms"
                                crashed_csv_file = output_dir / f"crashed_seeds_detailed{algorithm_suffix}.csv"
                                save_crashed_seeds_to_csv(crashed_seeds_data, crashed_csv_file)

                    except Exception as e:
                        error_msg = str(e)
                        if "infinity" in error_msg.lower() or "too large for dtype" in error_msg.lower():
                            print(f" INFINITY BUG: {algorithm}-{column_order} seed={seed} | SKIPPING")
                            crashed_seeds.add(seed)
                            crashed_seeds_data.append({
                                'seed': seed,
                                'algorithm': algorithm,
                                'column_order': column_order,
                                'train_size': train_size,
                                'repetition': repetition,
                                'error_type': 'infinity_bug',
                                'error_message': error_msg[:200],
                                'timestamp': pd.Timestamp.now().isoformat(),
                            })
                        else:
                            print(f" OTHER ERROR in experiment {algorithm}-{column_order} seed={seed}: {e}")
                            crashed_seeds_data.append({
                                'seed': seed,
                                'algorithm': algorithm,
                                'column_order': column_order,
                                'train_size': train_size,
                                'repetition': repetition,
                                'error_type': 'other_error',
                                'error_message': error_msg[:200],
                                'timestamp': pd.Timestamp.now().isoformat(),
                            })
                        continue

                    finally:
                        if experiment_counter % 10 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()

    # ======================
    # SAVE FINAL RESULTS
    # ======================
    total_time = time.time() - start_time
    print(f"\nExperiment completed in {total_time/60:.1f} minutes")
    
    # Filter results to only include those for the current train_sizes before saving
    filtered_results = [
        row for row in results
        if row.get('train_size') is None or row.get('train_size') in TRAIN_SIZES
    ]
    save_results_to_csv(filtered_results, str(output_file))

    print("\n" + "=" * 60)
    print(" CRASHED SEEDS SUMMARY")
    print("=" * 60)
    if crashed_seeds:
        crashed_list = sorted(list(crashed_seeds))
        print(f"Total crashed seeds: {len(crashed_list)}")
        print(f"Crashed seeds: {crashed_list}")
        print("\n NOTE: Experiments with these seeds returned -1.0 metrics.")
        algorithm_suffix = f"_{algorithm_filter}" if algorithm_filter else "_all_algorithms"
        crashed_seeds_file = output_dir / f"crashed_seeds{algorithm_suffix}.txt"
        with open(crashed_seeds_file, 'w') as f:
            f.write("# Seeds that triggered TabPFN infinity bug\n")
            for seed in crashed_list:
                f.write(f"{seed}\n")
        print(f"   Crashed seeds list saved to: {crashed_seeds_file}")
        if crashed_seeds_data:
            crashed_csv_file = output_dir / f"crashed_seeds_detailed{algorithm_suffix}.csv"
            save_crashed_seeds_to_csv(crashed_seeds_data, crashed_csv_file)
    else:
        print(" No crashed seeds detected! All experiments completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SimGlucose Comparison Experiment (acyclic SCM)")
    parser.add_argument("--test", action="store_true", help="Run in test mode (faster, fewer experiments)")
    parser.add_argument("--save-every", type=int, default=100, help="Save results every N experiments (default: 100)")
    parser.add_argument("--save-datasets", action="store_true", help="Save train/test datasets to .npz files")
    parser.add_argument("--save-reordered", action="store_true", help="Save reordered train datasets (requires --save-datasets)")
    parser.add_argument("--save-synthetic", action="store_true", help="Save synthetic data for analysis")
    parser.add_argument(
        "--algorithm",
        help=(
            "Only: vanilla | dag | "
            "cpdag_discovered | cpdag_minimal"
        ),
    )
    parser.add_argument(
        "--column-order",
        help=(
            "Specify column ordering: original | topological | reverse_topological"
        ),
    )
    parser.add_argument(
        "--skip-seeds",
        type=str,
        default="",
        help="Comma-separated list of seed values to skip entirely.",
    )
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        help="Override the default train sizes with one or more positive integers.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        help="Override the default number of repetitions per configuration.",
    )
    args = parser.parse_args()

    setup_determinism(
        enable_cuda_determinism=True,
        cublas_workspace_config=":4096:8",
        set_num_threads=1,
        verbose=True,
    )

    skip_seeds_arg = args.skip_seeds.split(",") if args.skip_seeds else []
    skip_seeds_set = {
        int(seed_str.strip())
        for seed_str in skip_seeds_arg
        if seed_str.strip()
    }

    main(
        test_mode=args.test,
        save_every=args.save_every,
        save_datasets=args.save_datasets,
        save_reordered=args.save_reordered,
        save_synthetic=args.save_synthetic,
        algorithm_filter=args.algorithm,
        column_order_filter=args.column_order,
        skip_seeds=skip_seeds_set,
        train_sizes=args.train_sizes,
        repetitions=args.repetitions,
    )
