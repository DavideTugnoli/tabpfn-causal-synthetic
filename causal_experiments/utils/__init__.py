"""Utilities for causal experiments.

This module intentionally avoids eager imports of heavy dependencies (for example,
``torch`` pulled by ``experiment_utils``) so lightweight tooling such as plotting
can run in environments without full training dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple


_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    # SCM data generation
    "generate_numeric_scm_data": ("scm_data", "generate_numeric_scm_data"),
    "generate_mixed_scm_data": ("scm_data", "generate_mixed_scm_data"),
    "get_numeric_dag_and_config": ("scm_data", "get_numeric_dag_and_config"),
    "get_mixed_dag_and_config": ("scm_data", "get_mixed_dag_and_config"),
    "get_numeric_cpdag_and_config": ("scm_data", "get_numeric_cpdag_and_config"),
    "get_mixed_cpdag_and_config": ("scm_data", "get_mixed_cpdag_and_config"),
    "NUMERIC_COL_NAMES": ("scm_data", "NUMERIC_COL_NAMES"),
    "MIXED_COL_NAMES": ("scm_data", "MIXED_COL_NAMES"),
    "NUMERIC_CATEGORICAL_COLS": ("scm_data", "NUMERIC_CATEGORICAL_COLS"),
    "MIXED_CATEGORICAL_COLS": ("scm_data", "MIXED_CATEGORICAL_COLS"),
    # Metrics
    "FaithfulDataEvaluator": ("metrics", "FaithfulDataEvaluator"),
    # DAG utilities
    "topological_sort": ("dag_utils", "topological_sort"),
    "get_worst_ordering": ("dag_utils", "get_worst_ordering"),
    "get_ordering_strategies": ("dag_utils", "get_ordering_strategies"),
    "count_graph_edges": ("dag_utils", "count_graph_edges"),
    "format_graph_structure_string": ("dag_utils", "format_graph_structure_string"),
    "dag_to_ideal_cpdag": ("dag_utils", "dag_to_ideal_cpdag"),
    "get_graph_nodes_count": ("dag_utils", "get_graph_nodes_count"),
    "convert_named_dag_to_indices": ("dag_utils", "convert_named_dag_to_indices"),
    "convert_indices_dag_to_named": ("dag_utils", "convert_indices_dag_to_named"),
    # Experiment utilities
    "get_experimental_configs": ("experiment_utils", "get_experimental_configs"),
    "create_train_test_splits": ("experiment_utils", "create_train_test_splits"),
    "prepare_data_and_graph": ("experiment_utils", "prepare_data_and_graph"),
    "prepare_vanilla_data": ("experiment_utils", "prepare_vanilla_data"),
    "prepare_dag_data": ("experiment_utils", "prepare_dag_data"),
    "prepare_cpdag_data": ("experiment_utils", "prepare_cpdag_data"),
    "_get_true_dag_for_ordering": ("experiment_utils", "_get_true_dag_for_ordering"),
    "identify_vanilla_duplicates": ("experiment_utils", "identify_vanilla_duplicates"),
    "copy_vanilla_duplicate_result": ("experiment_utils", "copy_vanilla_duplicate_result"),
    "pre_calculate_cpdags_for_splits": ("experiment_utils", "pre_calculate_cpdags_for_splits"),
    "create_result_row": ("experiment_utils", "create_result_row"),
    "save_results_to_csv": ("experiment_utils", "save_results_to_csv"),
    "save_base_datasets_npz": ("experiment_utils", "save_base_datasets_npz"),
    "save_processed_datasets_npz": ("experiment_utils", "save_processed_datasets_npz"),
    "save_global_test_set": ("experiment_utils", "save_global_test_set"),
    "save_train_set_per_split": ("experiment_utils", "save_train_set_per_split"),
    "save_reordered_dataset": ("experiment_utils", "save_reordered_dataset"),
    "save_synthetic_data": ("experiment_utils", "save_synthetic_data"),
    "load_synthetic_data": ("experiment_utils", "load_synthetic_data"),
    "load_global_test_set": ("experiment_utils", "load_global_test_set"),
    "load_train_set_per_split": ("experiment_utils", "load_train_set_per_split"),
    "validate_splits_consistency": ("experiment_utils", "validate_splits_consistency"),
    "print_progress": ("experiment_utils", "print_progress"),
    "discover_cpdag_from_data": ("experiment_utils", "discover_cpdag_from_data"),
    # CSuite dataset utilities
    "load_csuite_dataset": ("csuite_loader", "load_csuite_dataset"),
    "adjacency_matrix_to_dag_dict": ("csuite_loader", "adjacency_matrix_to_dag_dict"),
    "get_variable_types_info": ("csuite_loader", "get_variable_types_info"),
    "list_available_datasets": ("csuite_loader", "list_available_datasets"),
    # Determinism utilities
    "setup_determinism": ("determinism", "setup_determinism"),
    "set_experiment_seeds": ("determinism", "set_experiment_seeds"),
    "ensure_gpu_determinism": ("determinism", "ensure_gpu_determinism"),
    "setup_full_determinism": ("determinism", "setup_full_determinism"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
