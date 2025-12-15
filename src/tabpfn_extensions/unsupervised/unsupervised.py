#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""TabPFNUnsupervisedModel: Unsupervised learning capabilities for TabPFN.

This module enables TabPFN to be used for unsupervised learning tasks
including missing value imputation, outlier detection, and synthetic data
generation. It leverages TabPFN's probabilistic nature to model joint data
distributions without training labels.

Key features:
- Missing value imputation with probabilistic sampling
- Outlier detection based on feature-wise probability estimation
- Synthetic data generation with controllable randomness
- Compatibility with both TabPFN and TabPFN-client backends
- Support for mixed data types (categorical and numerical features)
- Flexible permutation-based approach for feature dependencies
- Support for Directed Acyclic Graphs (DAGs) to define feature relationships

Example usage:
    ```python
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

    # Create TabPFN models for classification and regression
    clf = TabPFNClassifier()
    reg = TabPFNRegressor()

    # Create the unsupervised model
    model = TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)

    # Fit the model on data without labels
    model.fit(X_train)

    # Different unsupervised tasks
    X_imputed = model.impute(X_with_missing_values)  # Fill missing values
    outlier_scores = model.outliers(X_test)          # Detect outliers
    X_synthetic = model.generate_synthetic_data(100)  # Generate new samples
    ```
"""

from __future__ import annotations

import copy
import os
import random
from typing import Any, Dict, List

from graphlib import TopologicalSorter
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from tqdm import tqdm

# Import TabPFN models from extensions (which handles backend compatibility)
from tabpfn_extensions.utils import (  # type: ignore
    TabPFNClassifier,
    TabPFNRegressor,
    infer_categorical_features,
)

# Import causal utilities
from .causal_utils import parse_cpdag_adjacency_matrix


class TabPFNUnsupervisedModel(BaseEstimator):
    """TabPFN experiments model for imputation, outlier detection, and synthetic data generation.

    This model combines a TabPFNClassifier for categorical features and a TabPFNRegressor for
    numerical features to perform various experiments learning tasks on tabular data.

    Parameters:
        tabpfn_clf : TabPFNClassifier, optional
            TabPFNClassifier instance for handling categorical features. If not provided, the model
            assumes that there are no categorical features in the data.

        tabpfn_reg : TabPFNRegressor, optional
            TabPFNRegressor instance for handling numerical features. If not provided, the model
            assumes that there are no numerical features in the data.

    Attributes:
        categorical_features : list
            List of indices of categorical features in the input data.

    Examples:
    ```python title="Example"
    >>> tabpfn_clf = TabPFNClassifier()
    >>> tabpfn_reg = TabPFNRegressor()
    >>> model = TabPFNUnsupervisedModel(tabpfn_clf, tabpfn_reg)
    >>>
    >>> X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    >>> model.fit(X)
    >>>
    >>> X_imputed = model.impute(X)
    >>> X_outliers = model.outliers(X)
    >>> X_synthetic = model.generate_synthetic_data(n_samples=100)
    ```
    """

    def _more_tags(self):
        return {"allow_nan": True}

    def __init__(
        self,
        tabpfn_clf: TabPFNClassifier | None = None,
        tabpfn_reg: TabPFNRegressor | None = None,
    ) -> None:
        """Initialize the TabPFNUnsupervisedModel.

        Args:
            tabpfn_clf : TabPFNClassifier, optional
                TabPFNClassifier instance for handling categorical features. If not provided, the model
                assumes that there are no categorical features in the data.

            tabpfn_reg : TabPFNRegressor, optional
                TabPFNRegressor instance for handling numerical features. If not provided, the model
                assumes that there are no numerical features in the data.

        Raises:
            AssertionError
                If both tabpfn_clf and tabpfn_reg are None.
        """
        assert (
            tabpfn_clf is not None or tabpfn_reg is not None
        ), "You cannot set both `tabpfn_clf` and `tabpfn_reg` to None. You can set one to None, if your table exclusively consists of categoricals/numericals."

        self.tabpfn_clf = tabpfn_clf
        self.tabpfn_reg = tabpfn_reg
        self.estimators = [self.tabpfn_clf, self.tabpfn_reg]

        self.categorical_features: list[int] = []
        # Optional: human-readable feature names in the CURRENT column order
        self.feature_names: list[str] | None = None

    def set_categorical_features(self, categorical_features: list[int]) -> None:
        """Set categorical feature indices for the model.

        Args:
            categorical_features: List of indices of categorical features
        """
        self.categorical_features = categorical_features
        for estimator in self.estimators:
            if hasattr(estimator, "set_categorical_features"):
                try:
                    estimator.set_categorical_features(categorical_features)
                except AttributeError:
                    # Estimator has the attribute but it's not callable
                    pass
                except TypeError:
                    # Wrong argument type
                    pass
                except ValueError:
                    # Invalid values in categorical_features
                    pass

    def set_feature_names(self, feature_names: list[str]) -> None:
        """Set display names for features in the CURRENT column order.

        This is used only for debugging/trace logs to avoid confusion when
        columns have been reordered upstream. If not provided, debug logs
        fall back to generic labels like X0, X1, ...
        """
        self.feature_names = feature_names

    # First implementation of fit - will be replaced by the updated version below

    def init_model_and_get_model_config(self) -> None:
        """Initialize TabPFN models for use in unsupervised learning.

        This function provides compatibility with different TabPFN implementations.
        It tries to initialize the model using the appropriate method based on the
        TabPFN implementation in use.

        Raises:
            RuntimeError: If model initialization fails
        """
        for estimator in self.estimators:
            if estimator is None:
                continue

            try:
                # First try the direct method (original TabPFN implementation)
                if hasattr(estimator, "init_model_and_get_model_config"):
                    estimator.init_model_and_get_model_config()

                # For TabPFN models from our unified import system (or v2), we need to ensure
                # they're initialized without requiring specific methods
                # Check if the model has a model attribute (TabPFN package)
                # This is a no-op for most implementations and is just to ensure compatibility
                elif hasattr(estimator, "model") and estimator.model is None:
                    # Call predict once to initialize the model
                    _ = estimator.predict(torch.zeros((1, 2)))

                    # For client implementations, there's no additional initialization needed
                    # The model will be initialized on first prediction call
            except Exception as e:
                raise RuntimeError(f"Failed to initialize model: {e}") from e

    # Add the method to the TabPFNClassifier and TabPFNRegressor if they don't have it
    def _ensure_init_model_method(self):
        """Ensure all estimators have the init_model_and_get_model_config method."""
        for idx, estimator in enumerate(self.estimators):
            if estimator is None:
                continue

            # Skip if the estimator already has the method
            if hasattr(estimator, "init_model_and_get_model_config"):
                continue

            # Add a compatibility wrapper method to the estimator
            def init_wrapper(est=estimator):
                """Compatibility wrapper for init_model_and_get_model_config."""
                # For TabPFN models, ensure they're initialized by calling predict once
                if hasattr(est, "model") and est.model is None:
                    _ = est.predict(torch.zeros((1, 2)))
                # For client implementations, there's nothing to do

            # Add the method to the estimator
            estimator.init_model_and_get_model_config = init_wrapper

            # Update the estimator in the list
            self.estimators[idx] = estimator

    def fit(
        self,
        X: np.ndarray | torch.Tensor | pd.DataFrame,
        y: np.ndarray | torch.Tensor | pd.Series | None = None,
    ) -> TabPFNUnsupervisedModel:
        """Fit the model to the input data.

        Args:
            X: Union[np.ndarray, torch.Tensor, pd.DataFrame]
                Input data to fit the model, shape (n_samples, n_features).

            y: Optional[Union[np.ndarray, torch.Tensor, pd.Series]], default=None
                Target values, shape (n_samples,). Optional since this is an unsupervised model.

        Returns:
            TabPFNUnsupervisedModel
                Fitted model instance (self).
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, pd.DataFrame):
            X = torch.tensor(X.values, dtype=torch.float32)

        self.X_ = copy.deepcopy(X)

        # Ensure y is not None and doesn't contain NaN values
        if y is not None:
            # Create a dummy y if none is provided
            y_clean = copy.deepcopy(y)
            # Replace any NaN values with zeros
            if torch.is_tensor(y_clean):
                if torch.isnan(y_clean).any():
                    y_clean = torch.nan_to_num(y_clean, nan=0.0)
            elif hasattr(y_clean, "numpy"):
                arr = y_clean.numpy()
                if np.isnan(arr).any():
                    arr = np.nan_to_num(arr, nan=0.0)
                    y_clean = torch.tensor(arr)
        else:
            # Create a dummy target with zeros if none is provided
            y_clean = torch.zeros(X.shape[0])

        self.y = y_clean

        # Get a numpy array from X for feature inference
        X_np = X
        if torch.is_tensor(X_np):
            X_np = X_np.cpu().numpy()

        self.categorical_features = infer_categorical_features(
            X_np,
            self.categorical_features,
        )

        # Ensure all estimators have the init_model_and_get_model_config method
        self._ensure_init_model_method()

        return self

    def _get_cpdag_original_ordering(
        self,
        cpdag: Dict[int, Dict[str, List[int]]],
        all_features: List[int],
        causal_structures_last: bool = False,
    ) -> tuple[List[int], Dict[int, List[int]], Dict[int, str]]:
        """Get ordering for original CPDAG approach.
        
        Original CPDAG approach:
        - Causal nodes (with directed parents): Use causal parents for conditioning
        - Correlational nodes (without directed parents): Use previous features (TabPFN default)
        
        Args:
            cpdag: CPDAG dictionary
            all_features: List of all feature indices
            causal_structures_last: If True, generate causal structures (parents + children) 
                                  after correlational nodes. If False, use standard ordering.
            
        Returns:
            tuple containing:
                - Ordered list of features for generation
                - Dictionary mapping each feature to its conditioning features
                - Dictionary mapping each feature to its generation strategy
        """
        from graphlib import TopologicalSorter
        
        print(f"\n=== DEBUG CPDAG ORIGINAL ORDERING ===")
        # Prepare pretty names if available
        use_names = getattr(self, "feature_names", None) is not None
        def n(i: int) -> str:
            return self.feature_names[i] if use_names else f"X{i}"

        print(f"Original CPDAG: {cpdag}")
        print(f"All features: {all_features} = {[n(i) for i in all_features]}")
        print(f"Causal structures last: {causal_structures_last}")
        
        # Fill missing nodes in CPDAG
        for i in all_features:
            if i not in cpdag:
                cpdag[i] = {"parents": [], "undirected": []}
                
        # Classify nodes (original approach)
        causal_nodes = []
        correlational_nodes = []
        hybrid_nodes = []  # NEW: Nodes with both directed and undirected edges
        
        for node in all_features:
            parents = cpdag[node]["parents"]
            undirected = cpdag[node]["undirected"]
            # Check if this node is a causal parent to other nodes
            is_causal_parent = any(node in cpdag[other_node]["parents"] for other_node in all_features if other_node != node)
            
            if parents:
                causal_nodes.append(node)
            elif is_causal_parent and undirected:  # NEW: Is causal parent AND has undirected edges
                hybrid_nodes.append(node)
            else:
                correlational_nodes.append(node)
        
        print(f"Causal nodes: {causal_nodes} = {[n(i) for i in causal_nodes]}")
        print(f"Correlational nodes: {correlational_nodes} = {[n(i) for i in correlational_nodes]}")
        print(f"Hybrid nodes (causal parent + undirected): {hybrid_nodes} = {[n(i) for i in hybrid_nodes]}")
        
        # Extract all parents of causal nodes (these must come first)
        causal_parents = set()
        for node in causal_nodes:
            causal_parents.update(cpdag[node]["parents"])
        
        # Separate correlational nodes into parents and non-parents
        # Hybrid nodes are treated as parents but will use vanilla strategy
        parent_nodes = [node for node in correlational_nodes if node in causal_parents] + hybrid_nodes
        non_parent_nodes = [node for node in correlational_nodes if node not in causal_parents]
        
        print(f"Parent nodes (must come first): {parent_nodes} = {[n(i) for i in parent_nodes]}")
        print(f"Causal nodes: {causal_nodes} = {[n(i) for i in causal_nodes]}")
        print(f"Non-parent correlational nodes: {non_parent_nodes} = {[n(i) for i in non_parent_nodes]}")
        
        # Create DAG only with causal nodes and their relationships
        causal_only_dag = {}
        for node in causal_nodes:
            # Only include parents that are also causal nodes
            causal_parents_only = [p for p in cpdag[node]["parents"] if p in causal_nodes]
            causal_only_dag[node] = causal_parents_only
        
        print(f"Causal-only DAG for topological sort: {causal_only_dag}")
        
        # Sort only the causal nodes among themselves
        ts = TopologicalSorter(causal_only_dag)
        ordered_causal = list(ts.static_order())
        print(f"Ordered causal nodes: {ordered_causal} = {[n(i) for i in ordered_causal]}")
        
        # Final ordering depends on causal_structures_last parameter
        if causal_structures_last:
            # Causal-last approach: correlational -> parents -> causal children
            final_ordering = non_parent_nodes + parent_nodes + ordered_causal
            print(f"Final ordering (causal-last): {final_ordering} = {[n(i) for i in final_ordering]}")
        else:
            # Standard approach: parents -> causal children -> correlational
            final_ordering = parent_nodes + ordered_causal + non_parent_nodes
            print(f"Final ordering (standard): {final_ordering} = {[n(i) for i in final_ordering]}")
        
        # Create parent mapping and strategy mapping
        parent_mapping = {}
        strategy_mapping = {}
        
        for node in all_features:
            if node in causal_nodes:
                parent_mapping[node] = cpdag[node]["parents"]
                strategy_mapping[node] = "causal"
            elif node in hybrid_nodes:
                # NEW: Hybrid nodes strategy determined dynamically during generation
                parent_mapping[node] = []  # Will be filled during generation
                strategy_mapping[node] = "hybrid_dynamic"  # Dynamic strategy assignment
            elif node in parent_nodes:
                # Parent nodes with undirected connections: condition on undirected neighbors
                # Parent nodes without undirected connections: generate independently
                undirected_neighbors = cpdag[node]["undirected"]
                if undirected_neighbors:
                    parent_mapping[node] = []  # Will be filled during generation based on undirected connections
                    strategy_mapping[node] = "parent_undirected"
                else:
                    parent_mapping[node] = []  # Will be filled during generation  
                    strategy_mapping[node] = "parent_independent"
            else:  # non-parent correlational
                parent_mapping[node] = []  # Will be filled during generation
                strategy_mapping[node] = "correlational"
        
        print(f"Parent mapping: {parent_mapping}")
        if use_names:
            parent_mapping_names = {n(k): [n(p) for p in v] for k, v in parent_mapping.items()}
            print(f"Parent mapping (names): {parent_mapping_names}")
        print(f"Strategy mapping: {strategy_mapping}")
        print("=== END DEBUG CPDAG ORIGINAL ORDERING ===\n")
        
        return final_ordering, parent_mapping, strategy_mapping


    def impute_(
        self,
        X: torch.Tensor,
        t: float = 0.000000001,
        n_permutations: int = 10,
        condition_on_all_features: bool = True,
        dag: dict[int, list[int]] | None = None,
        cpdag: dict[int, dict[str, list[int]]] | np.ndarray | None = None,
        causal_structures_last: bool = False,
        fast_mode: bool = False,
    ) -> torch.Tensor:
        """Impute missing values (np.nan) in X by sampling all cells independently from the trained models.

        Parameters:
            X: torch.Tensor
                Input data of shape (n_samples, n_features) with missing values encoded as np.nan
            t: float, default=0.000000001
                Temperature for sampling from the imputation distribution, lower values are more deterministic
            n_permutations: int, default=10
                Number of permutations to use for imputation
            condition_on_all_features: bool, default=True
                Whether to condition on all other features (True) or only previous features (False)
            dag: dict[int, list[int]] | None, default=None
                Dictionary representing a Directed Acyclic Graph (DAG) defining feature dependencies.
            cpdag: dict[int, dict[str, list[int]]] | None, default=None
                Dictionary representing a CPDAG with format:
                {node_idx: {"parents": [parent_indices], "undirected": [undirected_indices]}}
            causal_structures_last: bool, default=False
                If True, generate causal structures (parents -> children) after correlational nodes.
                If False, use standard ordering (causal structures first).
            fast_mode: bool, default=False
                Whether to use faster settings for testing

        Returns:
            torch.Tensor: Imputed data with missing values replaced
        """
        n_features = X.shape[1]
        all_features = list(range(n_features))

        X_fit = self.X_
        impute_X = copy.deepcopy(X)

        # Handle CPDAG input (original approach)
        if cpdag is not None:
            if dag is not None:
                raise ValueError("Cannot use both DAG and CPDAG simultaneously.")
            if condition_on_all_features:
                raise ValueError(
                    "CPDAG cannot be used with condition_on_all_features=True."
                )
            
            # Parse CPDAG and get ordering
            from .causal_utils import parse_cpdag_adjacency_matrix
            cpdag_dict = parse_cpdag_adjacency_matrix(cpdag) if isinstance(cpdag, np.ndarray) else cpdag
            all_features, parent_mapping, strategy_mapping = self._get_cpdag_original_ordering(
                cpdag_dict, all_features, causal_structures_last
            )
            
        # Handle DAG input (existing logic)
        elif dag is not None:
            if condition_on_all_features:
                raise ValueError(
                    "DAG cannot be used with condition_on_all_features=True."
                )
            print(f"\n=== DEBUG DAG ORDERING ===")
            print(f"Original DAG: {dag}")
            # Pretty names if available
            use_names = getattr(self, "feature_names", None) is not None
            def n(i: int) -> str:
                return self.feature_names[i] if use_names else f"X{i}"
            print(f"All features: {all_features} = {[n(i) for i in all_features]}")
            # fill up the DAG with empty lists for features not in the DAG
            for i in all_features:
                if i not in dag:
                    dag[i] = []
            print(f"DAG after filling: {dag}")
            ts = TopologicalSorter(dag)
            # re-order all_features based on the DAG (throws error in case of cycles)
            all_features = list(ts.static_order())
            print(f"Final ordering: {all_features} = {[n(i) for i in all_features]}")
            print("=== END DEBUG DAG ORDERING ===\n")
            
        # Create variable name mapping for debug output. If feature names are provided
        # (in current column order), use them to make logs reflect ORIGINAL variable names
        # after any upstream reordering.
        original_features = list(range(n_features))
        if getattr(self, "feature_names", None):
            # Names in the CURRENT column order
            names_current_order = self.feature_names  # length == n_features
            variable_names = names_current_order
            ordered_variable_names = [names_current_order[i] for i in all_features]
        else:
            variable_names = [f"X{i}" for i in original_features]
            ordered_variable_names = [f"X{i}" for i in all_features]
        
        # Debug print for generation order
        if cpdag is not None:
            print(f"\n=== DEBUG CPDAG ORIGINAL GENERATION ===")
            print(f"Original feature order: {original_features} = {variable_names}")
            print(f"Generation order: {all_features} = {ordered_variable_names}")
            print(f"Parent mapping: {parent_mapping}")
            print(f"Strategy mapping: {strategy_mapping}")
            print("=== GENERATION PROCESS ===")
        elif dag is not None:
            print(f"\n=== DEBUG DAG GENERATION ===")
            print(f"Original feature order: {original_features} = {variable_names}")
            print(f"Generation order: {all_features} = {ordered_variable_names}")
            print(f"DAG: {dag}")
            print("=== GENERATION PROCESS ===")
        else:
            # Vanilla TabPFN generation (no DAG/CPDAG)
            print(f"\n=== DEBUG VANILLA GENERATION ===")
            print(f"Original feature order: {original_features} = {variable_names}")
            print(f"Generation order: {all_features} = {ordered_variable_names}")
            print(f"Mode: {'All features conditioning' if condition_on_all_features else 'Sequential conditioning'}")
            print("=== GENERATION PROCESS ===")
        
        for i in tqdm(range(len(all_features))):
            column_idx = all_features[i]

            if cpdag is not None:
                # CPDAG approach: for undirected edges, both ends use vanilla conditioning
                # (condition on all predecessors in generation order)
                if strategy_mapping[column_idx] == "causal":
                    undirected_neighbors = cpdag_dict[column_idx]["undirected"]
                    # Force vanilla for both ends if any undirected edge exists
                    if undirected_neighbors:
                        conditional_idx = all_features[:i] if i > 0 else []
                        strategy_mapping[column_idx] = "hybrid_vanilla"
                    else:
                        conditional_idx = parent_mapping[column_idx]
                elif strategy_mapping[column_idx] == "hybrid_dynamic":
                    # Hybrid nodes with undirected edges: use vanilla conditioning
                    conditional_idx = all_features[:i] if i > 0 else []
                    strategy_mapping[column_idx] = "hybrid_vanilla"
                elif strategy_mapping[column_idx] == "parent_independent":
                    # Independent generation for non-causal parent without undirected neighbors
                    conditional_idx = []
                elif strategy_mapping[column_idx] == "parent_undirected":
                    # Parent nodes with undirected connections: use vanilla conditioning
                    conditional_idx = all_features[:i] if i > 0 else []
                else:  # correlational
                    # Vanilla sequential conditioning on all predecessors
                    conditional_idx = all_features[:i] if i > 0 else []
                    
                # Create readable conditioning variables list
                if getattr(self, "feature_names", None):
                    conditional_names = [variable_names[idx] for idx in conditional_idx] if conditional_idx else []
                    current_name = variable_names[column_idx]
                else:
                    conditional_names = [f"X{idx}" for idx in conditional_idx] if conditional_idx else []
                    current_name = f"X{column_idx}"
                print(f"Step {i}: Generating variable idx={column_idx} name={current_name} (position {i} in generation order) ({strategy_mapping[column_idx]}) conditioned on {conditional_idx} = {conditional_names}")
            elif dag is not None:
                # If a DAG is provided, use the dependencies from the DAG
                conditional_idx = dag.get(column_idx, [])
                # Create readable conditioning variables list
                if getattr(self, "feature_names", None):
                    conditional_names = [variable_names[idx] for idx in conditional_idx] if conditional_idx else []
                    current_name = variable_names[column_idx]
                else:
                    conditional_names = [f"X{idx}" for idx in conditional_idx] if conditional_idx else []
                    current_name = f"X{column_idx}"
                print(f"Step {i}: Generating variable idx={column_idx} name={current_name} (position {i} in generation order) (DAG) conditioned on {conditional_idx} = {conditional_names}")
            elif not condition_on_all_features:
                conditional_idx = all_features[:i] if i > 0 else []
                if getattr(self, "feature_names", None):
                    conditional_names = [variable_names[idx] for idx in conditional_idx] if conditional_idx else []
                    current_name = variable_names[column_idx]
                else:
                    conditional_names = [f"X{idx}" for idx in conditional_idx] if conditional_idx else []
                    current_name = f"X{column_idx}"
                print(f"Step {i}: Generating variable idx={column_idx} name={current_name} (position {i} in generation order) (vanilla sequential) conditioned on {conditional_idx} = {conditional_names}")
            else:
                conditional_idx = list(set(range(X.shape[1])) - {column_idx})
                if getattr(self, "feature_names", None):
                    conditional_names = [variable_names[idx] for idx in conditional_idx] if conditional_idx else []
                    current_name = variable_names[column_idx]
                else:
                    conditional_names = [f"X{idx}" for idx in conditional_idx] if conditional_idx else []
                    current_name = f"X{column_idx}"
                print(f"Step {i}: Generating variable idx={column_idx} name={current_name} (position {i} in generation order) (vanilla all features) conditioned on {conditional_idx} = {conditional_names}")

            y_predict = impute_X[:, column_idx]

            if torch.isnan(y_predict).sum() == 0:
                continue

            X_where_y_is_nan = impute_X[torch.isnan(y_predict)]
            X_where_y_is_nan = X_where_y_is_nan.reshape(-1, impute_X.shape[1])

            densities: list[Any] = []
            # Use fewer permutations in fast mode
            actual_n_permutations = 1 if fast_mode else n_permutations
            
            for perm in efficient_random_permutation(
                conditional_idx,
                actual_n_permutations,
            ):
                perm = (*perm, column_idx)
                _, pred = self.impute_single_permutation_(
                    X_where_y_is_nan,
                    perm,
                    t,
                    condition_on_all_features,
                )
                densities.append(pred)

            if not self.use_classifier_(column_idx, X_fit[:, column_idx]):
                pred_merged = densities[0][
                    "criterion"
                ].average_bar_distributions_into_this(
                    [d["criterion"] for d in densities],
                    [
                        d["logits"].clone().detach()
                        if torch.is_tensor(d["logits"])
                        else torch.tensor(d["logits"])
                        for d in densities
                    ],
                )
                pred_sampled = densities[0]["criterion"].sample(pred_merged, t=t)
            else:
                # Convert numpy arrays to tensors if necessary before stacking
                tensor_densities = [
                    torch.tensor(d) if isinstance(d, np.ndarray) else d
                    for d in densities
                ]
                pred = torch.stack(tensor_densities).mean(dim=0)
                pred_sampled = (
                    torch.distributions.Categorical(probs=pred).sample().float()
                )

            impute_X[torch.isnan(y_predict), column_idx] = pred_sampled

        return impute_X


    def impute_single_permutation_(
        self,
        X: torch.Tensor,
        feature_permutation: list[int] | tuple[int, ...],
        t: float = 0.000000001,
        condition_on_all_features: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Impute missing values (np.nan) in X by sampling all cells independently from the trained models.

        :param X: Input data of the shape (num_examples, num_features) with missing values encoded as np.nan
        :param t: Temperature for sampling from the imputation distribution, lower values are more deterministic
        :return: Imputed data, with missing values replaced
        """
        X_fit = self.X_
        impute_X = copy.deepcopy(X)

        for i in range(len(feature_permutation)):
            column_idx = feature_permutation[i]

            if not condition_on_all_features:
                conditional_idx = feature_permutation[:i] if i > 0 else []
            else:
                conditional_idx = list(set(range(X.shape[1])) - {column_idx})

            y_predict = impute_X[:, column_idx]

            if torch.isnan(y_predict).sum() == 0:
                continue

            X_where_y_is_nan = impute_X[torch.isnan(y_predict)]
            X_where_y_is_nan = X_where_y_is_nan.reshape(-1, impute_X.shape[1])

            model, X_predict, _ = self.density_(
                X_where_y_is_nan,
                X_fit,
                conditional_idx,
                column_idx,
            )

            pred, pred_sampled = self.sample_from_model_prediction_(
                column_idx,
                X_fit,
                model,
                X_predict,
                t,
            )

            impute_X[torch.isnan(y_predict), column_idx] = pred_sampled

        return impute_X, pred

    def sample_from_model_prediction_(
        self,
        column_idx: int,
        X_fit: torch.Tensor,
        model: Any,
        X_predict: torch.Tensor,
        t: float,
    ) -> tuple[dict[str, Any] | np.ndarray, torch.Tensor]:
        """Sample values from a model's prediction distribution.

        Args:
            column_idx: Index of the column being predicted
            X_fit: Training data used to determine feature type
            model: The trained model (classifier or regressor)
            X_predict: Input data for prediction
            t: Temperature parameter for sampling (lower values = more deterministic)

        Returns:
            tuple containing:
                - The raw prediction output (dictionary for regressors, array for classifiers)
                - The sampled values as a tensor
        """
        if not self.use_classifier_(column_idx, X_fit[:, column_idx]):
            pred = model.predict(X_predict.numpy(), output_type="full")
            # Proper tensor construction to avoid warnings
            logits = pred["logits"]
            logits_tensor = (
                logits.clone().detach()
                if torch.is_tensor(logits)
                else torch.as_tensor(logits)
            )
            pred_sampled = pred["criterion"].sample(logits_tensor, t=t)
        else:
            pred = model.predict_proba(X_predict.numpy())
            # Proper tensor construction to avoid warnings
            probs_tensor = torch.as_tensor(pred)
            pred_sampled = (
                torch.distributions.Categorical(probs=probs_tensor).sample().float()
            )

        return pred, pred_sampled

    def use_classifier_(self, column_idx: int, y: torch.Tensor | np.ndarray) -> bool:
        """Determine whether to use a classifier or regressor for a feature.

        Args:
            column_idx: Index of the column to check
            y: Values of the feature

        Returns:
            bool: True if a classifier should be used, False for a regressor
        """
        # Check if we should use classifier based on feature type and number of unique values
        max_classes = getattr(self.tabpfn_clf, "max_num_classes_", 10)
        return (
            column_idx in self.categorical_features and len(np.unique(y)) < max_classes
        )

    def density_(
        self,
        X_predict: torch.Tensor,
        X_fit: torch.Tensor,
        conditional_idx: list[int],
        column_idx: int,
    ) -> tuple[Any, torch.Tensor, torch.Tensor]:
        """Generate density predictions for a specific feature based on other features.

        This internal method is used by the imputation and outlier detection algorithms
        to model the conditional probability distribution of one feature given others.

        Args:
            X_predict: Input data for which to make predictions
            X_fit: Training data to fit the model
            conditional_idx: Indices of features to condition on
            column_idx: Index of the feature to predict

        Returns:
            tuple containing:
                - The fitted model (classifier or regressor)
                - The filtered features used for prediction
                - The target feature values to predict
        """
        # Initialize model if needed
        self.init_model_and_get_model_config()

        if len(conditional_idx) > 0:
            # If not the first feature, use all previous features
            mask = torch.zeros_like(X_fit).bool()
            mask[:, conditional_idx] = True
            X_fit, y_fit = X_fit[mask], X_fit[:, column_idx]
            X_fit = X_fit.reshape(mask.shape[0], -1)

            mask = torch.zeros_like(X_predict).bool()
            mask[:, conditional_idx] = True
            X_predict, y_predict = X_predict[mask], X_predict[:, column_idx]
            X_predict = X_predict.reshape(mask.shape[0], -1)
        else:
            # If the first feature, use a zero feature as input
            # Because of preprocessing, we can't use a zero feature, so we use a random feature
            X_fit, y_fit = torch.randn_like(X_fit[:, 0:1]), X_fit[:, column_idx]
            X_predict, y_predict = torch.randn_like(X_predict[:, 0:1]), X_predict[:, column_idx]

        model = (
            self.tabpfn_clf
            if self.use_classifier_(column_idx, y_fit)
            else self.tabpfn_reg
        )

        # Handle potential nan values in y_fit
        y_fit_np = y_fit.numpy() if hasattr(y_fit, "numpy") else y_fit
        if np.isnan(y_fit_np).any():
            y_fit_np = np.nan_to_num(y_fit_np, nan=0.0)

        X_fit_np = X_fit.numpy() if hasattr(X_fit, "numpy") else X_fit

        model.fit(X_fit_np, y_fit_np)

        return model, X_predict, y_predict

    def impute(
        self,
        X: torch.Tensor | np.ndarray | pd.DataFrame,
        t: float = 0.000000001,
        n_permutations: int = 10,
        dag: dict[int, list[int]] | None = None,
        cpdag: dict[int, dict[str, list[int]]] | np.ndarray | None = None,
        causal_structures_last: bool = False,
    ) -> torch.Tensor:
        """Impute missing values in the input data using the fitted TabPFN models.

        This method fills missing values (np.nan) in the input data by predicting
        each missing value based on the observed values in the same sample. The
        imputation uses multiple random feature permutations to improve robustness.

        Parameters:
            X: Union[torch.Tensor, np.ndarray, pd.DataFrame]
                Input data of shape (n_samples, n_features) with missing values
                encoded as np.nan.

            t: float, default=0.000000001
                Temperature for sampling from the imputation distribution.
                Lower values result in more deterministic imputations, while
                higher values introduce more randomness.

            n_permutations: int, default=10
                Number of random feature permutations to use for imputation.
                Higher values may improve robustness but increase computation time.

            dag: dict[int, list[int]] | None, default=None
                Dictionary representing a Directed Acyclic Graph (DAG) defining feature dependencies.

            cpdag: dict[int, dict[str, list[int]]] | np.ndarray | None, default=None
                Dictionary representing a CPDAG with format:
                {node_idx: {"parents": [parent_indices], "undirected": [undirected_indices]}}

            causal_structures_last: bool, default=False
                If True, generate causal structures (parents -> children) after correlational nodes.
                If False, use standard ordering (causal structures first).

        Returns:
            torch.Tensor
                Imputed data with missing values replaced, of shape (n_samples, n_features).

        Note:
            The model must be fitted with training data before calling this method.
        """
        # Convert input to torch tensor if needed
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, pd.DataFrame):
            X = torch.tensor(X.values, dtype=torch.float32)

        # Check if running in test mode
        fast_mode = os.environ.get("FAST_TEST_MODE", "0") == "1"

        return self.impute_(
            X,
            t,
            condition_on_all_features=True,
            n_permutations=n_permutations,
            dag=dag,
            cpdag=cpdag,
            causal_structures_last=causal_structures_last,
            fast_mode=fast_mode,
        )

    def outliers_single_permutation_(
        self,
        X: torch.tensor,
        feature_permutation: list[int] | tuple[int],
    ) -> torch.tensor:
        log_p = torch.zeros_like(
            X[:, 0],
        )  # Start with a log probability of 0 (log(1) = 0)

        for i, column_idx in enumerate(feature_permutation):
            model, X_predict, y_predict = self.density_(
                X,
                self.X_,
                feature_permutation[:i],
                column_idx,
            )
            if self.use_classifier_(column_idx, y_predict):
                # Get predictions and convert to torch tensor
                pred_np = model.predict_proba(X_predict.numpy())

                # Convert y_predict to indices for indexing the probabilities
                y_indices = (
                    y_predict.long()
                    if torch.is_tensor(y_predict)
                    else torch.tensor(y_predict, dtype=torch.long)
                )

                # Check indices are in bounds
                valid_indices = (y_indices >= 0) & (y_indices < pred_np.shape[1])
                # Get default probability tensor filled with a reasonable value
                pred = torch.ones_like(log_p) * 0.1  # Default small probability

                # Only index with valid indices
                if valid_indices.any():
                    # Get probabilities for each sample based on its class in y_predict
                    for idx, (prob_row, y_idx) in enumerate(zip(pred_np, y_indices)):
                        if (
                            0 <= y_idx < pred_np.shape[1]
                        ):  # Check bounds again per sample
                            # Proper tensor construction to avoid warning
                            pred[idx] = torch.as_tensor(prob_row[y_idx])
            else:
                pred = model.predict(X_predict, output_type="full")

                # Get logits tensor properly
                logits = pred["logits"]
                logits_tensor = logits.clone().detach()

                y_tensor = y_predict.clone().detach().to(logits.device)

                pred = pred["criterion"].pdf(logits_tensor, y_tensor).to(log_p.device)

            # Handle zero or negative probabilities (avoid log(0))
            pred = torch.clamp(pred, min=1e-10)

            # Convert probabilities to log probabilities
            log_pred = torch.log(pred)

            # Add log probabilities instead of multiplying probabilities
            log_p = log_p + log_pred

        return log_p, torch.exp(log_p)

    def outliers_pdf(self, X: torch.Tensor, n_permutations: int = 10) -> torch.Tensor:
        """Calculate outlier scores based on probability density functions for continuous features.

        This method filters out categorical features and only considers numerical features
        for outlier detection using probability density functions.

        Args:
            X: Input data tensor
            n_permutations: Number of permutations to use for the outlier calculation

        Returns:
            Tensor of outlier scores (lower values indicate more likely outliers)
        """
        X_store = copy.deepcopy(self.X_)
        mask = torch.ones_like(X_store).bool()
        mask[self.categorical_features] = False
        self.X_ = self.X_[mask]
        mask = torch.ones_like(X).bool()
        mask[self.categorical_features] = False
        X = X[mask]

        pdf = self.outliers(X, n_permutations=n_permutations)
        self.X_ = X_store
        return pdf

    def outliers_pmf(self, X: torch.Tensor, n_permutations: int = 10) -> torch.Tensor:
        """Calculate outlier scores based on probability mass functions for categorical features.

        This method filters out numerical features and only considers categorical features
        for outlier detection using probability mass functions.

        Args:
            X: Input data tensor
            n_permutations: Number of permutations to use for the outlier calculation

        Returns:
            Tensor of outlier scores (lower values indicate more likely outliers)
        """
        X_store = copy.deepcopy(self.X_)
        mask = torch.zeros_like(X_store).bool()
        mask[self.categorical_features] = True
        self.X_ = self.X_[mask]
        mask = torch.zeros_like(X).bool()
        mask[self.categorical_features] = True
        X = X[mask]

        pmf = self.outliers(X, n_permutations=n_permutations)
        self.X_ = X_store
        return pmf

    def outliers(
        self,
        X: torch.Tensor | np.ndarray | pd.DataFrame,
        n_permutations: int = 10,
    ) -> torch.Tensor:
        """Calculate outlier scores for each sample in the input data.

        This is the preferred implementation for outlier detection, which calculates
        sample probability for each sample in X by multiplying the probabilities of
        each feature according to chain rule of probability. Lower probabilities
        indicate samples that are more likely to be outliers.

        Parameters:
            X: Union[torch.Tensor, np.ndarray, pd.DataFrame]
                Samples to calculate outlier scores for, shape (n_samples, n_features)
            n_permutations: int, default=10
                Number of permutations to use for more robust probability estimates.
                Higher values may produce more stable results but increase computation time.

        Returns:
            torch.Tensor:
                Tensor of outlier scores (lower values indicate more likely outliers),
                shape (n_samples,)

        Raises:
            RuntimeError: If the model initialization fails
            ValueError: If the input data has incompatible dimensions
        """
        # Convert input to torch tensor if needed
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, pd.DataFrame):
            X = torch.tensor(X.values, dtype=torch.float32)

        # Initialize model if needed
        self.init_model_and_get_model_config()

        n_features = X.shape[1]
        all_features = list(range(n_features))

        # Check if running in test mode
        fast_mode = os.environ.get("FAST_TEST_MODE", "0") == "1"

        # Use fewer permutations in fast mode
        actual_n_permutations = 1 if fast_mode else n_permutations

        densities: list[torch.Tensor | np.ndarray] = []
        for perm in efficient_random_permutation(all_features, actual_n_permutations):
            perm_density_log, perm_density = self.outliers_single_permutation_(
                X,
                feature_permutation=perm,
            )
            densities.append(perm_density)

        # Average the densities across all permutations
        # Handle potential infinite values by replacing them with large finite values
        densities_clean: list[torch.Tensor] = [
            torch.nan_to_num(d, nan=0.0, posinf=1e30, neginf=1e-30)
            if torch.is_tensor(d)
            else torch.nan_to_num(
                torch.tensor(d, dtype=torch.float32),
                nan=0.0,
                posinf=1e30,
                neginf=1e-30,
            )
            for d in densities
        ]

        # Stack the clean tensors and compute mean
        densities_tensor = torch.stack(densities_clean)
        return densities_tensor.mean(dim=0)

    def generate_synthetic_data(
        self,
        n_samples: int = 100,
        t: float = 1.0,
        n_permutations: int = 3,
        dag: dict[int, list[int]] | None = None,
        cpdag: dict[int, dict[str, list[int]]] | np.ndarray | None = None,
        causal_structures_last: bool = False,
    ) -> torch.Tensor:
        """Generate synthetic tabular data samples using the fitted TabPFN models.

        This method uses imputation to create synthetic data, starting with a matrix of NaN
        values and filling in each feature sequentially. Samples are generated feature by
        feature in a single pass, with each feature conditioned on previously generated features.

        Parameters:
            n_samples: int, default=100
                Number of synthetic samples to generate

            t: float, default=1.0
                Temperature parameter for sampling. Controls randomness:
                - Higher values (e.g., 1.0) produce more diverse samples
                - Lower values (e.g., 0.1) produce more deterministic samples

            n_permutations: int, default=3
                Number of feature permutations to use for generation
                More permutations may provide more robust results but increase computation time

            dag: dict[int, list[int]] | None, default=None
                Dictionary representing a Directed Acyclic Graph (DAG) defining feature dependencies.
                If provided, the generation will respect the dependencies defined in the DAG.
                
            cpdag: dict[int, dict[str, list[int]]] | np.ndarray | None, default=None
                Either a dictionary representing a CPDAG with format:
                {node_idx: {"parents": [parent_indices], "undirected": [undirected_indices]}}
                OR a numpy adjacency matrix where:
                0 = no edge, 1 = directed edge (column -> row), -1 = undirected edge
                If provided, uses hybrid approach: causal parents for causal/mixed nodes,
                previous features for correlational nodes.

            causal_structures_last: bool, default=False
                If True, generate causal structures (parents -> children) after correlational nodes.
                If False, use standard ordering (causal structures first).

        Returns:
            torch.Tensor:
                Generated synthetic data of shape (n_samples, n_features)

        Raises:
            AssertionError:
                If the model is not fitted (self.X_ does not exist)
        """
        # TODO: Test generating one feature at a time, with train data only for that feature
        #       and previously generated features, similar to the outliers method
        assert hasattr(
            self,
            "X_",
        ), "You need to fit the model before generating synthetic data"

        # Check if running in test mode
        fast_mode = os.environ.get("FAST_TEST_MODE", "0") == "1"

        # Use smaller number of samples in fast mode
        if fast_mode and n_samples > 10:
            n_samples = 5

        # Use fewer permutations in fast mode
        actual_n_permutations = 1 if fast_mode else n_permutations

        X = torch.zeros(n_samples, self.X_.shape[1]) * np.nan
        return self.impute_(
            X,
            t=t,
            condition_on_all_features=False,
            n_permutations=actual_n_permutations,
            dag=dag,
            cpdag=cpdag,
            causal_structures_last=causal_structures_last,
            fast_mode=fast_mode,
        )

    def get_embeddings(self, X: torch.tensor, per_column: bool = False) -> torch.tensor:
        """Get the transformer embeddings for the test data X.

        Args:
            X:

        Returns:
            torch.Tensor of shape (n_samples, embedding_dim)
        """
        raise NotImplementedError(
            "This method is not implemented currently. During the main TabPFN refactor this functionality was removed, please see: https://github.com/PriorLabs/TabPFN/issues/111",
        )

        if per_column:
            return self.get_embeddings_per_column(X)
        return self.get_embeddings_(X)

    def get_embeddings_(self, X: torch.tensor) -> torch.tensor:
        model = self.tabpfn_reg
        model.fit(
            self.X_,
            self.y
            if self.y is not None
            else (torch.zeros_like(self.X_[:, 0])),  # Must contain more than one class
        )  # Fit the data for random labels
        embs = model.get_embeddings(X, additional_y=None)
        return embs.reshape(X.shape[0], -1)

    def get_embeddings_per_column(self, X: torch.tensor) -> torch.tensor:
        """Alternative implementation for get_embeddings, where we get the embeddings for each column as a label
        separately and concatenate the results. This alternative way needs more passes but might be more accurate.
        """
        embs = []
        for column_idx in range(X.shape[1]):
            mask = torch.zeros_like(self.X_).bool()
            mask[:, column_idx] = True
            X_train, y_train = (
                self.X_[~(mask)].reshape(self.X_.shape[0], -1),
                self.X_[mask],
            )

            X_pred, _y_pred = X[~(mask)].reshape(X.shape[0], -1), X[mask]

            model = (
                self.tabpfn_clf
                if column_idx in self.categorical_features
                else self.tabpfn_reg
            )
            model.fit(X_train, y_train)
            embs += [model.get_embeddings(X_pred, additional_y=None)]

        return torch.cat(embs, 1).reshape(embs[0].shape[0], -1)


def efficient_random_permutation(
    indices: list[int],
    n_permutations: int = 10,
) -> list[tuple[int, ...]]:
    """Generate multiple unique random permutations of the given indices.

    Args:
        indices: List of indices to permute
        n_permutations: Number of unique permutations to generate

    Returns:
        List of unique permutations
    """
    perms: list[tuple[int, ...]] = []
    n_iter = 0
    max_iterations = n_permutations * 10  # Set a limit to avoid infinite loops

    while len(perms) < n_permutations and n_iter < max_iterations:
        perm = efficient_random_permutation_(indices)
        if perm not in perms:
            perms.append(perm)
        n_iter += 1

    return perms


def efficient_random_permutation_(indices: list[int]) -> tuple[int, ...]:
    """Generate a single random permutation from the given indices.

    Args:
        indices: List of indices to permute

    Returns:
        A tuple representing a random permutation of the input indices
    """
    # Create a copy of the list to avoid modifying the original
    permutation = list(indices)

    # Shuffle the list in-place using Fisher-Yates algorithm
    for i in range(len(indices) - 1, 0, -1):
        # Pick a random index from 0 to i
        j = random.randint(0, i)
        # Swap elements at i and j
        permutation[i], permutation[j] = permutation[j], permutation[i]

    return tuple(permutation)
