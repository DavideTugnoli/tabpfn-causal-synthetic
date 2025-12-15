
"""DAG utilities for causal experiments.

These functions provide utilities for working with DAG structures,
ordering strategies, and data reordering for causal experiments.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def topological_sort(dag: dict[int, list[int]]) -> list[int]:
    """Compute topological ordering of DAG nodes.

    This gives the "best" ordering where parents always come before children.

    Args:
        dag: Dictionary {node: [list_of_parents]}

    Returns:
        List of nodes in topological order
    """
    # Convert to adjacency list (node -> children)
    children = defaultdict(list)
    in_degree = defaultdict(int)

    # Get all nodes
    all_nodes = set(dag.keys())
    for parents in dag.values():
        all_nodes.update(parents)

    # Initialize in_degree
    for node in all_nodes:
        in_degree[node] = 0

    # Build children and in_degree
    for child, parents in dag.items():
        for parent in parents:
            children[parent].append(child)
            in_degree[child] += 1

    # Kahn's algorithm
    queue = deque([node for node in all_nodes if in_degree[node] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(result) != len(all_nodes):
        raise ValueError("DAG contains cycles!")

    return result


def get_worst_ordering(dag: dict[int, list[int]]) -> list[int]:
    """Get the worst possible ordering that maximizes causal violations.

    Uses reverse topological order, which is mathematically guaranteed to
    produce the maximum number of causal violations possible.

    Args:
        dag: Dictionary {node: [list_of_parents]}

    Returns:
        List of nodes in worst possible order (optimal)
    """
    # Get topological order (best ordering with 0 violations)
    topo_order = topological_sort(dag)

    # Reverse it to get worst ordering (maximum violations)
    worst_order = list(reversed(topo_order))

    return worst_order


def count_violations(dag: dict[int, list[int]], ordering: list[int]) -> int:
    """Count total number of causal violations in an ordering.

    A violation occurs when a child node appears before its parent in the ordering.

    Args:
        dag: Dictionary {node: [list_of_parents]}
        ordering: List of nodes in some order

    Returns:
        Number of violated causal dependencies
    """
    # Create position mapping
    position = {node: i for i, node in enumerate(ordering)}

    violations = 0
    for child, parents in dag.items():
        child_pos = position[child]
        for parent in parents:
            parent_pos = position[parent]
            # Violation if child comes before parent
            if child_pos < parent_pos:
                violations += 1

    return violations


def get_ordering_strategies(dag: dict[int, list[int]]) -> dict[str, list[int]]:
    """Get all available ordering strategies for a given DAG.

    Args:
        dag: DAG structure

    Returns:
        Dictionary of {strategy_name: ordering}
    """
    all_nodes = set(dag.keys())
    for parents in dag.values():
        all_nodes.update(parents)

    original_order = sorted(all_nodes)  # [0, 1, 2, 3, ...] - deterministic

    strategies = {
        "original": original_order,
        "topological": topological_sort(dag),
        "reverse_topological": get_worst_ordering(dag),
    }

    return strategies



def count_graph_edges(graph_dict: dict) -> tuple:
    """Count the number of edges in a DAG or CPDAG dict.
    For a standard DAG: returns (directed, 0)
    For a CPDAG dict: returns (directed, undirected).
    """
    if not graph_dict:
        return 0, 0
    sample_val = next(iter(graph_dict.values()))
    if isinstance(sample_val, dict) and "parents" in sample_val:
        # CPDAG dict
        directed = sum(len(v.get("parents", [])) for v in graph_dict.values())
        undirected = sum(len(v.get("undirected", [])) for v in graph_dict.values()) // 2
        return directed, undirected
    # Standard DAG dict
    directed = sum(len(parents) for parents in graph_dict.values())
    return directed, 0


def get_graph_edge_counts(graph_dict: dict) -> dict:
    """Returns a dict with 'directed' and 'undirected' edge counts for a DAG or CPDAG dict."""
    directed, undirected = count_graph_edges(graph_dict)
    return {"directed": directed, "undirected": undirected}


def convert_named_dag_to_indices(named_dag: dict, column_names: list) -> dict:
    """Convert a DAG defined with node names to one with indices.

    Args:
        named_dag: Dictionary {node_name: [list_of_parent_names]}
        column_names: List of column names defining the index mapping

    Returns:
        Dictionary {node_index: [list_of_parent_indices]}
    """
    # Create mapping from names to indices
    name_to_idx = {name: idx for idx, name in enumerate(column_names)}

    # Convert DAG
    index_dag = {}
    for node_name, parent_names in named_dag.items():
        if node_name in name_to_idx:  # Skip nodes not in column_names
            node_idx = name_to_idx[node_name]
            parent_indices = [name_to_idx[p] for p in parent_names if p in name_to_idx]
            index_dag[node_idx] = parent_indices

    return index_dag


def convert_indices_dag_to_named(dag: dict, column_names: list) -> dict:
    """Convert a DAG defined with indices back to one with node names.

    Args:
        dag: Dictionary {node_index: [list_of_parent_indices]}
        column_names: List of column names defining the index mapping

    Returns:
        Dictionary {node_name: [list_of_parent_names]}
    """
    idx_to_name = dict(enumerate(column_names))

    named_dag = {}
    for node_idx, parent_indices in dag.items():
        if node_idx < len(idx_to_name):  # Ensure index is valid
            node_name = idx_to_name[node_idx]
            parent_names = [idx_to_name[p] for p in parent_indices if p < len(idx_to_name)]
            named_dag[node_name] = parent_names

    return named_dag


def format_graph_structure_string(graph_dict: dict, column_names: list) -> str:
    """Format graph structure using integer indices for CSV storage.

    Args:
        graph_dict: DAG or CPDAG dictionary with integer indices
        column_names: List of column names (not used, kept for compatibility)
                     The indices in graph_dict correspond to positions in the current column order

    Returns:
        String representation using integer indices:
        - DAG: "{0: [], 1: [0, 2], 2: [3], 3: []}"
        - CPDAG: "{0: {'parents': [], 'undirected': []}, 1: {'parents': [0, 2], 'undirected': []}}"
    """
    if not graph_dict:
        return "no_graph"

    # Convert to string representation using TabPFN internal format with integer indices
    import json
    
    # Clean the dictionary to ensure proper JSON serialization
    clean_dict = {}
    for key, value in graph_dict.items():
        # Ensure key is integer
        int_key = int(key) if not isinstance(key, int) else key
        
        if isinstance(value, dict):
            # CPDAG format: ensure lists are properly formatted
            clean_dict[int_key] = {
                'parents': list(value.get('parents', [])),
                'undirected': list(value.get('undirected', []))
            }
        else:
            # DAG format: ensure value is a list
            clean_dict[int_key] = list(value) if value is not None else []
    
    # Sort by keys for consistent output
    sorted_dict = dict(sorted(clean_dict.items()))
    
    # Convert to string but make it more readable by replacing quotes
    result = str(sorted_dict)
    # Replace single quotes with double quotes for JSON-like format
    result = result.replace("'", '"')
    
    return result


def get_graph_nodes_count(graph_dict: dict) -> int:
    """Get the number of nodes in a graph structure.

    Args:
        graph_dict: DAG or CPDAG dictionary

    Returns:
        Number of nodes in the graph
    """
    if not graph_dict:
        return 0

    all_nodes = set(graph_dict.keys())

    # Check if this is a CPDAG
    sample_val = next(iter(graph_dict.values()))
    is_cpdag = isinstance(sample_val, dict) and "parents" in sample_val

    if is_cpdag:
        # CPDAG format - collect all nodes from parents and undirected
        for connections in graph_dict.values():
            all_nodes.update(connections.get("parents", []))
            all_nodes.update(connections.get("undirected", []))
    else:
        # DAG format - collect all parent nodes
        for parents in graph_dict.values():
            all_nodes.update(parents)

    return len(all_nodes)


def dag_to_ideal_cpdag(dag: dict[int, list[int]]) -> dict[int, dict[str, list[int]]]:
    """
    Convert DAG to ideal CPDAG preserving ONLY V-structures (colliders).
    
    A V-structure (collider) occurs when a node has two or more parents. In an ideal CPDAG, 
    ONLY these V-structures are preserved with directed edges, while ALL other edges 
    become undirected, including edges outgoing from colliders.
    
    Rules for ideal CPDAG conversion:
    1. Identify all colliders (nodes with 2+ parents)
    2. Keep ONLY edges pointing INTO colliders as directed (preserves V-structures)
    3. Convert ALL other edges to undirected (including edges FROM colliders)
    
    Args:
        dag: DAG as {node_id: [parent_ids]}
        
    Returns:
        CPDAG as {node_id: {"parents": [directed_parent_ids], "undirected": [undirected_neighbor_ids]}}
    
    Examples:
        >>> # Simple collider: X4 → X3 → X2 ← X1
        >>> dag = {0: [], 1: [0, 2], 2: [3], 3: []}
        >>> result = dag_to_ideal_cpdag(dag)
        >>> # Node 1 is collider, keeps directed parents [0, 2]
        >>> # Edge 3→2 becomes undirected since it's not part of V-structure
        
        >>> # Mixed case: X4 → X3 → X2 ← X1, X5 ← X2  
        >>> dag = {0: [], 1: [0, 2], 2: [3], 3: [], 4: [1]}
        >>> result = dag_to_ideal_cpdag(dag)
        >>> # Node 1 is collider, keeps directed parents [0, 2]
        >>> # Edge 2↔5 becomes undirected (not a V-structure)
    """
    if not dag:
        return {}
    
    # Get all nodes in the graph
    all_nodes = set(dag.keys())
    for parents in dag.values():
        all_nodes.update(parents)
    
    # Initialize CPDAG with all nodes
    cpdag = {node: {"parents": [], "undirected": []} for node in all_nodes}
    
    # Build children mapping for easier navigation
    children = {}
    for node in all_nodes:
        children[node] = []
    for child, parents in dag.items():
        for parent in parents:
            children[parent].append(child)
    
    def are_connected(node1: int, node2: int) -> bool:
        """Check if two nodes are directly connected in the DAG."""
        return node2 in dag.get(node1, []) or node1 in dag.get(node2, [])
    
    def is_collider(node: int) -> bool:
        """
        Check if a node is a collider (has multiple parents, potentially unconnected).
        
        For ideal CPDAG construction, any node with 2+ parents forms a V-structure
        that should be preserved, regardless of whether the parents are connected.
        """
        parents = dag.get(node, [])
        return len(parents) >= 2
    
    # Identify all colliders
    colliders = {node for node in all_nodes if is_collider(node)}
    
    # Process each edge in the DAG
    processed_edges = set()
    
    for child, parents in dag.items():
        for parent in parents:
            edge = tuple(sorted([parent, child]))
            if edge in processed_edges:
                continue
            processed_edges.add(edge)
            
            # Rule 1: If child is a collider, keep edge directed (parent → collider)
            # This preserves V-structures - the ONLY directed edges in ideal CPDAG
            if child in colliders:
                cpdag[child]["parents"].append(parent)
            
            # Rule 2: For ALL other edges (including those FROM colliders), make undirected
            # In ideal CPDAG, we preserve ONLY V-structures, everything else is undirected
            else:
                cpdag[parent]["undirected"].append(child)
                cpdag[child]["undirected"].append(parent)
    
    # Clean up and sort for consistency
    for node in cpdag:
        cpdag[node]["parents"] = sorted(list(set(cpdag[node]["parents"])))
        cpdag[node]["undirected"] = sorted(list(set(cpdag[node]["undirected"])))
    
    return cpdag