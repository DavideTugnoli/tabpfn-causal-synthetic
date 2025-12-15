"""Utilities for causal experiments."""

from .dag_utils import (
    convert_indices_dag_to_named,
    convert_named_dag_to_indices,
    count_graph_edges,
    dag_to_ideal_cpdag,
    format_graph_structure_string,
    get_graph_nodes_count,
    get_ordering_strategies,
    get_worst_ordering,
    topological_sort,
)
from .experiment_utils import (
    create_result_row,
    create_train_test_splits,
    discover_cpdag_from_data,
    get_experimental_configs,
    identify_vanilla_duplicates,
    copy_vanilla_duplicate_result,
    pre_calculate_cpdags_for_splits,
    prepare_data_and_graph,
    prepare_vanilla_data,
    prepare_dag_data,
    prepare_cpdag_data,
    _get_true_dag_for_ordering,
    print_progress,
    save_base_datasets_npz,
    save_processed_datasets_npz,
    save_global_test_set,
    save_train_set_per_split,
    save_reordered_dataset,
    save_synthetic_data,
    load_synthetic_data,
    load_global_test_set,
    load_train_set_per_split,
    save_results_to_csv,
    validate_splits_consistency,
)
from .metrics import FaithfulDataEvaluator
from .scm_data import (
    MIXED_CATEGORICAL_COLS,
    MIXED_COL_NAMES,
    NUMERIC_CATEGORICAL_COLS,
    NUMERIC_COL_NAMES,
    generate_mixed_scm_data,
    generate_numeric_scm_data,
    get_mixed_cpdag_and_config,
    get_mixed_dag_and_config,
    get_numeric_cpdag_and_config,
    get_numeric_dag_and_config,
)
from .csuite_loader import (
    load_csuite_dataset,
    adjacency_matrix_to_dag_dict,
    get_variable_types_info,
    list_available_datasets,
)
from .determinism import (
    ensure_gpu_determinism,
    set_experiment_seeds,
    setup_determinism,
    setup_full_determinism,
)

__all__ = [
    # SCM data generation
    "generate_numeric_scm_data",
    "generate_mixed_scm_data",
    "get_numeric_dag_and_config",
    "get_mixed_dag_and_config",
    "get_numeric_cpdag_and_config",
    "get_mixed_cpdag_and_config",
    "NUMERIC_COL_NAMES",
    "MIXED_COL_NAMES",
    "NUMERIC_CATEGORICAL_COLS",
    "MIXED_CATEGORICAL_COLS",

    # Metrics
    "FaithfulDataEvaluator",

    # DAG utilities
    "topological_sort", "get_worst_ordering", "get_ordering_strategies",
    "count_graph_edges", "format_graph_structure_string", "dag_to_ideal_cpdag",
    "get_graph_nodes_count", "convert_named_dag_to_indices", "convert_indices_dag_to_named",

    # Experiment utilities
    "get_experimental_configs", "create_train_test_splits", "prepare_data_and_graph",
    "prepare_vanilla_data", "prepare_dag_data", "prepare_cpdag_data", "_get_true_dag_for_ordering",
    "identify_vanilla_duplicates", "copy_vanilla_duplicate_result", "pre_calculate_cpdags_for_splits",
    "create_result_row", "save_results_to_csv", "save_base_datasets_npz", "save_processed_datasets_npz", 
    "save_global_test_set", "save_train_set_per_split", "save_reordered_dataset", "save_synthetic_data", "load_synthetic_data", "load_global_test_set", "load_train_set_per_split",
    "validate_splits_consistency", "print_progress", "discover_cpdag_from_data",
    
    # CSuite dataset utilities
    "load_csuite_dataset", "adjacency_matrix_to_dag_dict", "get_variable_types_info", "list_available_datasets",
    
    # Determinism utilities
    "setup_determinism", "set_experiment_seeds", "ensure_gpu_determinism", "setup_full_determinism",
    
    # Categorical mapping utilities
]
