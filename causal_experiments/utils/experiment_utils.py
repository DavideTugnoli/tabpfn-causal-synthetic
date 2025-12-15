
"""Experiment utilities for causal comparison experiments.

This module provides utilities for managing experimental configurations,
data splitting, seeding, and CSV output formatting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .dag_utils import (
    dag_to_ideal_cpdag,
    format_graph_structure_string,
    get_graph_edge_counts,
    get_graph_nodes_count,
    get_ordering_strategies,
)
# Import done locally to avoid circular import
# from .run_pc_discovery import run_pc_discovery_on_dataset


def discover_cpdag_from_data(
    X_train: np.ndarray,
    column_names: list[str],
    categorical_cols: list[str],
    use_categorical: bool = False,
    true_dag: dict[int, list[int]] | None = None,
    alpha: float = 0.05,  # Default alpha for PC algorithm
    indep_test: str | None = None,
    hybrid_params: dict[str, Any] | None = None,
) -> dict | None:
    """Discover CPDAG structure from training data using PC algorithm.
    
    Args:
        X_train: Training data for causal discovery
        column_names: List of column names
        categorical_cols: List of categorical column names  
        use_categorical: Whether data includes categorical features
        true_dag: True DAG structure for comparison/validation (optional)
        alpha: Significance level for PC algorithm (default: 0.05)
        
    Returns:
        CPDAG dictionary in the expected format, or None if discovery fails
    """
    try:
        # Convert categorical column names to indices
        categorical_indices = [column_names.index(col) for col in categorical_cols if col in column_names]
        
        # Determine dataset type for discovery
        dataset_name = "mixed" if use_categorical and categorical_indices else "continuous"

        if indep_test is None:
            selected_test = "hybrid" if dataset_name == "mixed" else "fisherz"
        else:
            selected_test = indep_test
        
        # Import locally to avoid circular import
        from .run_pc_discovery import run_pc_discovery_on_dataset

        # Run PC discovery with provided alpha
        if selected_test == "hybrid":
            params = hybrid_params.copy() if hybrid_params else {}
        else:
            params = None

        cpdag_object = run_pc_discovery_on_dataset(
            dataset_name=dataset_name,
            data=X_train,
            true_dag=true_dag,  # For comparison/validation only (can be None)
            col_names=column_names,
            categorical_cols=categorical_indices,
            verbose=False,
            output_dir=None,  # No plot saving
            alpha=alpha,  # Use provided alpha for sensitivity
            indep_test=selected_test,
            hybrid_params=params,
        )
        
        # Extract adjacency matrix from CausalGraph object
        if isinstance(cpdag_object, np.ndarray):
            # Fallback for empty CPDAG
            cpdag_matrix = cpdag_object
        else:
            # Extract matrix from CausalGraph object
            cpdag_matrix = cpdag_object.G.graph
        
        # Convert adjacency matrix to CPDAG dictionary format
        from tabpfn_extensions.unsupervised.causal_utils import parse_cpdag_adjacency_matrix
        cpdag_dict = parse_cpdag_adjacency_matrix(cpdag_matrix)
        
        return cpdag_dict
        
    except Exception as e:
        # Re-raise the exception instead of falling back silently
        raise RuntimeError(f"CPDAG discovery failed: {e}") from e




def get_experimental_configs(use_categorical: bool = False) -> dict[str, dict[str, Any]]:
    """Get all experimental configurations for comparison experiment.
    
    Args:
        use_categorical: If True, use mixed SCM configurations, else numeric SCM

    Returns:
        Dictionary of experimental configurations with pre-calculated orderings
    """
    if use_categorical:
        # Mixed SCM: X3 → X2 → X1 ← X0, X4_cat ← X1
        scm_dag = {
            0: [],      # X0 has no parents
            1: [0, 2],  # X1 has parents X0 and X2 (collider)
            2: [3],     # X2 has parent X3
            3: [],      # X3 has no parents
            4: [1]      # X4_cat has parent X1
        }
        
        # Mixed CPDAG: X3 - X2 → X1 ← X0, X4_cat ← X1
        scm_cpdag = {
            0: {"parents": [], "undirected": []},        # X0
            1: {"parents": [0, 2], "undirected": []},    # X1 (collider)
            2: {"parents": [], "undirected": [3]},       # X2
            3: {"parents": [], "undirected": [2]},       # X3
            4: {"parents": [1], "undirected": []}        # X4_cat
        }
    else:
        # Numeric SCM: X3 → X2 → X1 ← X0 (collider at X1)
        scm_dag = {
            0: [],      # X0 has no parents
            1: [0, 2],  # X1 has parents X0 and X2 (collider)
            2: [3],     # X2 has parent X3
            3: []       # X3 has no parents
        }

        # Numeric CPDAG: X3 - X2 → X1 ← X0
        scm_cpdag = {
            0: {"parents": [], "undirected": []},        # X0 has no parents or undirected edges
            1: {"parents": [0, 2], "undirected": []},    # X1 has directed parents X0 and X2 (collider)
            2: {"parents": [], "undirected": [3]},       # X2 has undirected edge with X3
            3: {"parents": [], "undirected": [2]}        # X3 has undirected edge with X2
        }

    # Pre-calculate all orderings for consistency and performance
    from causal_experiments.utils.dag_utils import get_ordering_strategies
    orderings_dict = get_ordering_strategies(scm_dag)

    # Generate ideal CPDAG from true DAG (preserving only V-structures)
    ideal_cpdag = dag_to_ideal_cpdag(scm_dag)
    
    # Ensure ideal_cpdag is not None or empty
    if not ideal_cpdag:
        print("Warning: ideal_cpdag is empty, using scm_cpdag as fallback")
        ideal_cpdag = scm_cpdag
    
    configs = {
        "vanilla": {
            "orderings": [
                ("original", orderings_dict["original"]),
                ("topological", orderings_dict["topological"]),
                ("reverse_topological", orderings_dict["reverse_topological"])
            ],
            "dag": None,  # Vanilla never uses DAG for causal constraints
            "cpdag": None,
            "dag_for_ordering": scm_dag  # Only used for calculating column orderings
        },
        "dag": {
            "orderings": [
                ("topological", orderings_dict["topological"])
            ],
            "dag": scm_dag,
            "cpdag": None,
            # Use the true DAG to compute topological column ordering consistently
            "dag_for_ordering": scm_dag
        },
        # CPDAG with vanilla conditioning on both ends of undirected edges
        # Causal structures generated first (before correlational structures)
        "cpdag_discovered": {
            "orderings": [
                ("original", orderings_dict["original"])
            ],
            "dag": None,
            "cpdag": scm_cpdag,
            "dag_for_ordering": scm_dag,
            "causal_structures_last": False
        },
        # CPDAG on ideal/minimal CPDAG (only v-structures)
        # Causal structures generated first (before correlational structures)
        "cpdag_minimal": {
            "orderings": [
                ("original", orderings_dict["original"])
            ],
            "dag": None,
            "cpdag": ideal_cpdag,
            "dag_for_ordering": scm_dag,
            "causal_structures_last": False
        },
    }

    return configs


def create_train_test_splits(
    X: np.ndarray,
    train_sizes: list[int],
    test_size: int,
    seed: int
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Create train/test splits for different training sizes with fixed seed.

    Args:
        X: Full dataset
        train_sizes: List of training set sizes to create
        test_size: Size of test set (fixed)
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping train_size -> (X_train, X_test)
    """
    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    splits = {}

    for train_size in train_sizes:
        total_needed = train_size + test_size

        if total_needed > len(X):
            # If we need more data than available, sample with replacement
            indices = np.random.choice(len(X), size=total_needed, replace=True)
            X_subset = X[indices]
        else:
            # Sample without replacement
            indices = np.random.choice(len(X), size=total_needed, replace=False)
            X_subset = X[indices]

        # Split into train and test
        X_train, X_test = train_test_split(
            X_subset,
            train_size=train_size,
            test_size=test_size,
            random_state=seed,
            shuffle=True
        )

        splits[train_size] = (X_train, X_test)

    return splits


def validate_splits_consistency(
    splits_dict: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
    train_size: int,
    seed: int,
    tolerance: float = 1e-10
) -> None:
    """Validate that train/test splits are identical across all algorithms for the same seed and train_size.
    
    Args:
        splits_dict: Dictionary mapping algorithm -> {train_size: (X_train, X_test)}
        train_size: Training size to validate
        seed: Seed to validate
        tolerance: Numerical tolerance for array comparison
        
    Raises:
        ValueError: If splits are not consistent across algorithms
    """
    algorithms = list(splits_dict.keys())
    if len(algorithms) < 2:
        return  # Nothing to compare
    
    # Get reference splits from first algorithm
    reference_alg = algorithms[0]
    if train_size not in splits_dict[reference_alg]:
        return  # No splits for this train_size
        
    ref_train, ref_test = splits_dict[reference_alg][train_size]
    
    # Compare with all other algorithms
    for alg in algorithms[1:]:
        if train_size not in splits_dict[alg]:
            continue
            
        curr_train, curr_test = splits_dict[alg][train_size]
        
        # Check train set consistency
        if not np.allclose(ref_train, curr_train, atol=tolerance):
            raise ValueError(
                f"Train set inconsistency detected! "
                f"Algorithm '{reference_alg}' vs '{alg}' for seed={seed}, train_size={train_size}. "
                f"Train sets must be identical across algorithms for scientific validity."
            )
        
        # Check test set consistency  
        if not np.allclose(ref_test, curr_test, atol=tolerance):
            raise ValueError(
                f"Test set inconsistency detected! "
                f"Algorithm '{reference_alg}' vs '{alg}' for seed={seed}, train_size={train_size}. "
                f"Test sets must be identical across algorithms for scientific validity."
            )


def save_base_datasets_npz(
    X_train: np.ndarray,
    X_test: np.ndarray,
    train_size: int,
    seed: int,
    repetition: int,
    output_dir: str | Path,
    column_names: list[str]
) -> str:
    """Save base train/test datasets (shared across algorithms) to .npz format.
    
    This saves the raw datasets that are identical across all algorithms for
    the same seed/train_size combination, avoiding redundant storage.
    
    Args:
        X_train: Training data
        X_test: Test data  
        train_size: Training set size
        seed: Random seed
        repetition: Repetition number
        output_dir: Directory to save files
        column_names: List of column names
        
    Returns:
        Path to the saved .npz file
    """
    # Create filename with only the shared parameters
    filename = f"base_datasets_ts{train_size}_s{seed}_r{repetition}.npz"
    filepath = Path(output_dir) / "datasets" / filename
    
    # Skip if already exists
    if filepath.exists():
        return str(filepath)
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with metadata
    np.savez_compressed(
        str(filepath),
        X_train=X_train,
        X_test=X_test,
        train_size=train_size,
        seed=seed,
        repetition=repetition,
        column_names=column_names
    )
    
    return str(filepath)


def save_processed_datasets_npz(
    X_train_processed: np.ndarray,
    X_test_processed: np.ndarray,
    algorithm: str,
    column_order: str,
    train_size: int,
    seed: int,
    repetition: int,
    output_dir: str | Path,
    column_names: list[str],
    base_dataset_path: str
) -> str:
    """Save algorithm-specific processed datasets with reference to base dataset.
    
    Args:
        X_train_processed: Processed training data (after reordering/transforms)
        X_test_processed: Processed test data  
        algorithm: Algorithm name
        column_order: Column ordering strategy
        train_size: Training set size
        seed: Random seed
        repetition: Repetition number
        output_dir: Directory to save files
        column_names: List of column names after processing
        base_dataset_path: Path to the base dataset file
        
    Returns:
        Path to the saved .npz file
    """
    # Create filename with experiment parameters
    filename = f"processed_{algorithm}_{column_order}_ts{train_size}_s{seed}_r{repetition}.npz"
    filepath = Path(output_dir) / "datasets" / filename
    
    # Skip if already exists
    if filepath.exists():
        return str(filepath)
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with metadata
    np.savez_compressed(
        str(filepath),
        X_train_processed=X_train_processed,
        X_test_processed=X_test_processed,
        algorithm=algorithm,
        column_order=column_order,
        train_size=train_size,
        seed=seed,
        repetition=repetition,
        column_names_processed=column_names,
        base_dataset_path=base_dataset_path
    )
    
    return str(filepath)


def prepare_vanilla_data(
    X_train: np.ndarray,
    column_order: str,
    dag_for_ordering: dict[int, list[int]],
    column_names: list[str],
    categorical_features: list[int] | None = None
) -> tuple[np.ndarray, list[int] | None, list[int] | None]:
    """Prepare data for vanilla algorithm (no causal graph constraints).
    
    Args:
        X_train: Training data
        column_order: Column ordering strategy ("original", "topological", "reverse_topological")
        dag_for_ordering: DAG structure used ONLY for calculating orderings
        column_names: List of column names
        categorical_features: List of categorical feature indices (optional)
        
    Returns:
        Tuple of (X_reordered, column_ordering_used, updated_categorical_features)
    """
    if column_order == "original":
        return X_train, None, categorical_features
    
    # Use DAG structure only for calculating orderings
    orderings = get_ordering_strategies(dag_for_ordering)
    # Some experiments expect a deterministic shuffle of the original order
    if column_order not in orderings:
        if column_order == "original_2" and "original" in orderings:
            rng = np.random.default_rng(314159)
            shuffled = orderings["original"].copy()
            for _ in range(len(shuffled)):
                rng.shuffle(shuffled)
                if not np.array_equal(shuffled, orderings["original"]):
                    break
            orderings[column_order] = shuffled
        else:
            raise KeyError(column_order)
    new_ordering = orderings[column_order]
    
    # Reorder data columns only (DAG is not used by vanilla algorithm)
    X_reordered = X_train[:, new_ordering]
    
    # Update categorical feature indices based on new column ordering
    updated_categorical_features = None
    if categorical_features is not None:
        # Create mapping from old index to new index
        old_to_new_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(new_ordering)}
        
        # Update categorical indices
        updated_categorical_features = []
        for cat_idx in categorical_features:
            if cat_idx in old_to_new_mapping:
                updated_categorical_features.append(old_to_new_mapping[cat_idx])
        
        print(f"🔧 [COLUMN REORDERING] Updated categorical features: {categorical_features} → {updated_categorical_features}")
        print(f"   Column ordering: {list(range(len(new_ordering)))} → {new_ordering}")
    
    return X_reordered, new_ordering, updated_categorical_features


def prepare_dag_data(
    X_train: np.ndarray,
    dag: dict[int, list[int]],
    column_names: list[str]
) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Prepare data for DAG algorithm (strict causal ordering).
    
    Args:
        X_train: Training data
        dag: DAG structure for causal constraints
        column_names: List of column names
        
    Returns:
        Tuple of (X_train, dag) - data stays in original order
    """
    return X_train, dag


def prepare_cpdag_data(
    X_train: np.ndarray,
    cpdag: dict,
    column_names: list[str]
) -> tuple[np.ndarray, dict]:
    """Prepare data for CPDAG algorithm (hybrid causal/correlational).
    
    Args:
        X_train: Training data
        cpdag: CPDAG structure for hybrid constraints
        column_names: List of column names
        
    Returns:
        Tuple of (X_train, cpdag) - data stays in original order
    """
    return X_train, cpdag


def prepare_data_and_graph(
    X_train: np.ndarray,
    algorithm: str,
    column_order: str,
    dag: dict[int, list[int]] | None,
    cpdag: dict | None,
    column_names: list[str]
) -> tuple[np.ndarray, dict | None, dict | None, list[int] | None]:
    """Prepare data and graph structures based on algorithm and column order.
    
    DEPRECATED: Use specific prepare_*_data() functions instead.
    This function is kept for backward compatibility but will be removed.

    Args:
        X_train: Training data
        algorithm: Algorithm name ("vanilla", "dag", "cpdag")
        column_order: Column ordering strategy ("original", "topological", "reverse_topological")
        dag: DAG structure (if applicable)
        cpdag: CPDAG structure (if applicable)
        column_names: List of column names

    Returns:
        Tuple of (X_reordered, dag_reordered, cpdag_reordered, column_ordering_used)
    """
    if algorithm == "vanilla":
        # Use the true DAG for ordering calculations
        dag_for_ordering = _get_true_dag_for_ordering(X_train.shape[1])
        X_reordered, column_ordering_used = prepare_vanilla_data(
            X_train, column_order, dag_for_ordering, column_names
        )
        return X_reordered, None, None, column_ordering_used
    
    elif algorithm == "dag":
        if dag is None:
            raise ValueError("DAG structure is required for dag algorithm")
        X_prepared, dag_prepared = prepare_dag_data(X_train, dag, column_names)
        return X_prepared, dag_prepared, None, None
    
    elif algorithm.startswith("cpdag_"):
        if cpdag is None:
            raise ValueError("CPDAG structure is required for CPDAG algorithms")
        X_prepared, cpdag_prepared = prepare_cpdag_data(X_train, cpdag, column_names)
        return X_prepared, None, cpdag_prepared, None
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def _get_true_dag_for_ordering(n_features: int) -> dict[int, list[int]]:
    """Get the true DAG structure based on number of features (internal helper).
    
    Args:
        n_features: Number of features in the dataset
        
    Returns:
        True DAG structure for ordering calculations
    """
    if n_features == 5:
        # Mixed SCM: X4 → X3 → X2 ← X1, X5_cat ← X2
        return {
            0: [],      # X1 has no parents
            1: [0, 2],  # X2 has parents X1 and X3 (collider)
            2: [3],     # X3 has parent X4
            3: [],      # X4 has no parents
            4: [1]      # X5_cat has parent X2
        }
    else:
        # Numeric SCM: X4 → X3 → X2 ← X1 (collider at X2)
        return {
            0: [],      # X1 has no parents
            1: [0, 2],  # X2 has parents X1 and X3 (collider)
            2: [3],     # X3 has parent X4
            3: []       # X4 has no parents
        }


def create_result_row(
    algorithm: str,
    column_order: str,
    graph_dict: dict | None,
    train_size: int,
    seed: int,
    repetition: int,
    categorical_cols: list[str],
    column_names: list[str],
    metrics_results: dict[str, float],
    reordered_graph_dict: dict | None = None,
    column_ordering_used: list[int] | None = None,
) -> dict[str, Any]:
    """Create a single result row for CSV output.

    Args:
        algorithm: Algorithm name
        column_order: Column ordering strategy
        graph_dict: Graph structure (DAG or CPDAG) - indices correspond to original column order
        train_size: Training set size
        seed: Random seed used to generate the dataset
        repetition: Repetition number
        categorical_cols: List of categorical column names
        column_names: List of all column names in original order
        metrics_results: Dictionary of calculated metrics
        reordered_graph_dict: Graph structure after reordering (if any) - indices correspond to reordered column order
        column_ordering_used: List of column indices in the order used (if reordering was applied)

    Returns:
        Dictionary representing one CSV row
    """
    # Basic experiment information
    row = {
        "algorithm": algorithm,
        "column_order": column_order,
        "train_size": train_size,
        "seed": seed,
        "repetition": repetition,
        "categorical_cols": ";".join(categorical_cols) if categorical_cols else ""
    }

    # Graph structure information (uses the graph in the actual order used)
    # If reordered_graph_dict is available, use it (it reflects the actual order used)
    # Otherwise use graph_dict (original order)
    graph_to_use = reordered_graph_dict if reordered_graph_dict is not None else graph_dict
    
    if graph_to_use is not None:
        row["graph_structure"] = format_graph_structure_string(graph_to_use, column_names)
        row["graph_nodes"] = get_graph_nodes_count(graph_to_use)

        edge_counts = get_graph_edge_counts(graph_to_use)
        row["graph_edges"] = edge_counts["directed"]
        row["undirected_edges"] = edge_counts["undirected"]
    else:
        row["graph_structure"] = "no_graph"
        row["graph_nodes"] = 0
        row["graph_edges"] = 0
        row["undirected_edges"] = 0

    # Add all metrics
    row.update(metrics_results)

    return row


def save_results_to_csv(results: list[dict[str, Any]] | list[dict], output_path: str) -> None:
    """Save experimental results to CSV file.

    Args:
        results: List of result dictionaries
        output_path: Path to output CSV file
    """
    df = pd.DataFrame(results)

    # Define column order for CSV
    base_columns = [
        "algorithm", "column_order", "actual_column_order", "graph_structure", "graph_nodes",
        "graph_edges", "undirected_edges", "train_size", "seed",
        "repetition", "categorical_cols"
    ]

    # Ensure mandatory base columns exist even if older rows were loaded without them
    for col in base_columns:
        if col not in df.columns:
            df[col] = ""

    if "seed_base" in df.columns and "seed_base" not in base_columns:
        seed_index = base_columns.index("seed")
        base_columns.insert(seed_index + 1, "seed_base")
    
    # Add dataset_path if it exists
    if "dataset_path" in df.columns:
        base_columns.append("dataset_path")

    # Get metric columns (everything else)
    metric_columns = [col for col in df.columns if col not in base_columns]

    # Reorder columns
    ordered_columns = base_columns + sorted(metric_columns)
    df = df[ordered_columns]

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Replace None values with empty strings for cleaner CSV output
    df = df.fillna('')
    
    # Save to CSV
    df.to_csv(output_path, index=False)




def print_progress(
    current_config: int,
    total_configs: int,
    current_train_size: int,
    total_train_sizes: int,
    current_seed: int,
    total_seeds: int,
    algorithm: str,
    column_order: str,
    train_size: int
) -> None:
    """Print progress information during experiment execution.

    Args:
        current_config: Current configuration number
        total_configs: Total number of configurations
        current_train_size: Current train size index
        total_train_sizes: Total number of train sizes
        current_seed: Current seed number
        total_seeds: Total number of seeds
        algorithm: Current algorithm
        column_order: Current column order
        train_size: Current train size
    """
    progress_pct = (
        (current_config * total_train_sizes * total_seeds) +
        (current_train_size * total_seeds) +
        current_seed
    ) / (total_configs * total_train_sizes * total_seeds) * 100
    
    print(f"Progress: {algorithm}-{column_order}, train_size={train_size}, seed={current_seed} ({progress_pct:.1f}%)")


def save_global_test_set(
    X_test: np.ndarray,
    output_dir: str | Path,
    column_names: list[str]
) -> str:
    """Save the global test set (used by all experiments) once.
    
    Args:
        X_test: Test data (2000 samples, always same)
        output_dir: Directory to save files
        column_names: List of column names
        
    Returns:
        Path to the saved test set file
    """
    filepath = Path(output_dir) / "datasets" / "global_test_set.npz"
    
    # Skip if already exists
    if filepath.exists():
        return str(filepath)
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save global test set
    np.savez_compressed(
        str(filepath),
        X_test=X_test,
        column_names=column_names,
        description="Global test set used by all experiments (2000 samples)"
    )
    
    return str(filepath)


def save_train_set_per_split(
    X_train: np.ndarray,
    train_size: int,
    seed: int,
    output_dir: str | Path,
    column_names: list[str],
    *,
    overwrite: bool = False,
    dataset_seed: int | None = None,
    metadata: dict | None = None
) -> str:
    """Save training set for specific (train_size, seed) combination.
    
    Args:
        X_train: Training data
        train_size: Training set size
        seed: Base seed used for naming/reproducibility
        output_dir: Directory to save files
        column_names: List of column names
        dataset_seed: Actual random seed used to sample the dataset (optional)
        
    Returns:
        Path to the saved train set file
    """
    filename = f"train_ts{train_size}_s{seed}.npz"
    filepath = Path(output_dir) / "datasets" / filename
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.exists():
        if not overwrite:
            return str(filepath)
        filepath.unlink()

    # Save train set
    stored_seed = dataset_seed if dataset_seed is not None else seed
    data_to_save: dict[str, Any] = {
        "X_train": X_train,
        "train_size": train_size,
        "seed": stored_seed,
        "seed_base": seed,
        "column_names": column_names,
    }

    if metadata is not None:
        data_to_save["metadata"] = np.array(metadata, dtype=object)

    np.savez_compressed(str(filepath), **data_to_save)
    
    return str(filepath)


def load_global_test_set(
    output_dir: str | Path
) -> tuple[np.ndarray, list[str]] | None:
    """Load the global test set if it exists.
    
    Args:
        output_dir: Directory where files are saved
        
    Returns:
        Tuple of (X_test, column_names) if file exists, None otherwise
    """
    filepath = Path(output_dir) / "datasets" / "global_test_set.npz"
    
    if not filepath.exists():
        return None
    
    try:
        data = np.load(str(filepath), allow_pickle=True)
        X_test = data['X_test']
        column_names = data['column_names'].tolist()
        return X_test, column_names
    except Exception as e:
        print(f"Warning: Failed to load global test set from {filepath}: {e}")
        return None


def load_train_set_per_split(
    train_size: int,
    seed: int,
    output_dir: str | Path
) -> tuple[np.ndarray, list[str]] | None:
    """Load training set for specific (train_size, seed) combination if it exists.
    
    Args:
        train_size: Training set size
        seed: Random seed
        output_dir: Directory where files are saved
        
    Returns:
        Tuple of (X_train, column_names) if file exists, None otherwise
    """
    filename = f"train_ts{train_size}_s{seed}.npz"
    filepath = Path(output_dir) / "datasets" / filename
    
    if not filepath.exists():
        return None
    
    try:
        data = np.load(str(filepath), allow_pickle=True)
        X_train = data['X_train']
        column_names = data['column_names'].tolist()
        return X_train, column_names
    except Exception as e:
        print(f"Warning: Failed to load train set from {filepath}: {e}")
        return None


def save_reordered_dataset(
    X_train_reordered: np.ndarray,
    algorithm: str,
    column_order: str,
    train_size: int,
    seed: int,
    output_dir: str | Path,
    reordered_column_names: list[str],
    base_train_path: str
) -> str:
    """Save reordered training dataset (optional, for verification).
    
    Args:
        X_train_reordered: Reordered training data
        algorithm: Algorithm name
        column_order: Column ordering strategy
        train_size: Training set size
        seed: Random seed
        output_dir: Directory to save files
        reordered_column_names: Column names in reordered format
        base_train_path: Path to base training set
        
    Returns:
        Path to the saved reordered dataset
    """
    filename = f"reordered_{algorithm}_{column_order}_ts{train_size}_s{seed}.npz"
    filepath = Path(output_dir) / "datasets" / "reordered" / filename
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save reordered dataset with reference to base
    np.savez_compressed(
        str(filepath),
        X_train_reordered=X_train_reordered,
        algorithm=algorithm,
        column_order=column_order,
        train_size=train_size,
        seed=seed,
        reordered_column_names=reordered_column_names,
        base_train_path=base_train_path,
        description=f"Reordered training data for {algorithm}-{column_order}"
    )
    
    return str(filepath)


def save_synthetic_data(
    synthetic_df: pd.DataFrame,
    algorithm: str,
    column_order: str,
    train_size: int,
    seed: int,
    output_dir: str | Path,
    n_permutations: int,
    temperature: float,
    metrics: dict[str, float] | None = None
) -> str:
    """Save synthetic data generated by TabPFN in efficient NPZ format.
    
    Args:
        synthetic_df: Generated synthetic DataFrame
        algorithm: Algorithm name
        column_order: Column ordering strategy
        train_size: Training set size
        seed: Random seed
        output_dir: Directory to save files
        n_permutations: Number of permutations used
        temperature: Temperature parameter used
        metrics: Calculated metrics (optional)
        
    Returns:
        Path to the saved synthetic data file
    """
    # Special naming for CPDAG 2.0 algorithms to distinguish from original
    if algorithm.startswith("cpdag_v2_"):
        filename = f"synthetic_{algorithm}_v2_{column_order}_ts{train_size}_s{seed}.npz"
    else:
        filename = f"synthetic_{algorithm}_{column_order}_ts{train_size}_s{seed}.npz"
    
    filepath = Path(output_dir) / "datasets" / "synthetic" / filename
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare metadata dictionary
    metadata = {
        "algorithm": algorithm,
        "column_order": column_order,
        "train_size": train_size,
        "seed": seed,
        "n_permutations": n_permutations,
        "temperature": temperature,
        "n_samples": len(synthetic_df),
        "description": "Synthetic data generated by TabPFN"
    }
    
    # Add metrics if provided
    if metrics:
        metadata["metrics"] = metrics
    
    # Save in efficient NPZ format
    np.savez_compressed(
        str(filepath),
        synthetic_data=synthetic_df.values,
        column_names=synthetic_df.columns.tolist(),
        metadata=metadata
    )
    
    return str(filepath)


def load_synthetic_data(filepath: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load synthetic data from NPZ file.
    
    Args:
        filepath: Path to the NPZ file
        
    Returns:
        Tuple of (synthetic_dataframe, metadata_dict)
        
    Example:
        >>> df, meta = load_synthetic_data("synthetic_vanilla_original_ts20_s0.npz")
        >>> print(f"Algorithm: {meta['algorithm']}, Seed: {meta['seed']}")
        >>> print(df.head())
    """
    with np.load(filepath, allow_pickle=True) as data:
        synthetic_data = data['synthetic_data']
        column_names = data['column_names'].tolist()
        metadata = data['metadata'].item()  # .item() converts numpy scalar to dict
        
        # Reconstruct DataFrame
        synthetic_df = pd.DataFrame(synthetic_data, columns=column_names)
        
        return synthetic_df, metadata


def identify_vanilla_duplicates(configs: dict) -> dict[str, str]:
    """Identify duplicate vanilla configurations that produce the same column ordering.
    
    Args:
        configs: Experimental configurations dictionary
        
    Returns:
        Dictionary mapping duplicate column_order to original column_order
    """
    vanilla_duplicates = {}
    
    if "vanilla" not in configs:
        return vanilla_duplicates
    
    vanilla_config = configs["vanilla"]
    dag_for_ordering = vanilla_config.get("dag_for_ordering")
    
    if dag_for_ordering is None:
        return vanilla_duplicates
    
    # Calculate all orderings for vanilla
    from causal_experiments.utils.dag_utils import get_ordering_strategies
    orderings_dict = get_ordering_strategies(dag_for_ordering)
    
    # Create cache of orderings
    vanilla_orderings_cache = {}
    
    for column_order_item in vanilla_config["orderings"]:
        # Standard format: (column_order, order_indices) tuple
        if isinstance(column_order_item, tuple):
            column_order, ordering_indices = column_order_item
        else:
            # Fallback for old string format
            column_order = column_order_item
            if column_order in orderings_dict:
                ordering_indices = orderings_dict[column_order]
            else:
                continue  # Skip if not found in orderings_dict
        
        ordering_tuple = tuple(ordering_indices)  # Make hashable
        
        if ordering_tuple in vanilla_orderings_cache:
            # Found duplicate! Map to original
            original_order = vanilla_orderings_cache[ordering_tuple]
            vanilla_duplicates[column_order] = original_order
            print(f"🔄 VANILLA DUPLICATE: {column_order} → {original_order} (same ordering: {ordering_indices})")
        else:
            # New unique ordering
            vanilla_orderings_cache[ordering_tuple] = column_order
            print(f"✅ VANILLA UNIQUE: {column_order} (ordering: {ordering_indices})")
    
    print(f"Vanilla optimization: {len(vanilla_duplicates)} duplicates found out of {len(vanilla_config['orderings'])} configurations")
    return vanilla_duplicates


def copy_vanilla_duplicate_result(results: list[dict], algorithm: str, column_order: str, 
                                train_size: int, seed: int, vanilla_duplicates: dict) -> dict | None:
    """Copy result from original configuration if this is a vanilla duplicate.
    
    Args:
        results: List of existing results
        algorithm: Current algorithm
        column_order: Current column order
        train_size: Current train size
        seed: Current seed
        vanilla_duplicates: Dictionary of duplicate mappings
        
    Returns:
        Copied result row if duplicate found, None otherwise
    """
    if algorithm != "vanilla" or column_order not in vanilla_duplicates:
        return None
    
    original_order = vanilla_duplicates[column_order]
    
    # Find existing result for the original configuration
    for result in results:
        result_seed_base = result.get('seed_base', result['seed'])
        if (result['algorithm'] == algorithm and 
            result['column_order'] == original_order and 
            result['train_size'] == train_size and 
            result_seed_base == seed):
            
            # Copy the result with updated column_order
            result_row = result.copy()
            result_row['column_order'] = column_order
            result_row['actual_column_order'] = result['actual_column_order']
            print(f"📋 COPIED: {algorithm}-{column_order} → {algorithm}-{original_order} (seed_base={seed})")
            return result_row
    
    print(f"⚠️  WARNING: Duplicate {column_order} but no original {original_order} found for seed_base={seed}")
    return None


def pre_calculate_cpdags_for_splits(
    all_splits: dict,
    column_names: list[str],
    categorical_cols: list[str],
    use_categorical: bool = False
) -> dict:
    """Pre-calculate CPDAGs for all splits to avoid repeated discovery.
    
    Args:
        all_splits: Dictionary of {(seed, train_size): (X_train, X_test)}
        column_names: List of column names
        categorical_cols: List of categorical column names
        use_categorical: Whether data includes categorical features
        
    Returns:
        Dictionary of {(seed, train_size): cpdag_dict}
    """
    all_cpdags = {}
    
    print("Pre-calculating CPDAGs for all splits...")
    total_splits = len(all_splits)
    
    for i, (split_key, (X_train, _)) in enumerate(all_splits.items()):
        try:
            cpdag = discover_cpdag_from_data(
                X_train, column_names, categorical_cols, use_categorical
            )
            all_cpdags[split_key] = cpdag
            
            if (i + 1) % 10 == 0 or i == total_splits - 1:
                print(f"  CPDAG discovery: {i + 1}/{total_splits} splits completed")
                
        except Exception as e:
            print(f"  ⚠️  CPDAG discovery failed for {split_key}: {e}")
            all_cpdags[split_key] = None
    
    print(f"CPDAG pre-calculation completed: {len(all_cpdags)} splits")
    return all_cpdags
