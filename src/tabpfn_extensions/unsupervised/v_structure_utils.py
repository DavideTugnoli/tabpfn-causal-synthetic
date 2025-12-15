#!/usr/bin/env python3
"""
Utilities for finding V-structures (colliders) in directed graphs.

This module provides functions to identify V-structures in DAGs, which are essential
for proper causal structure generation ordering in TabPFN unsupervised models.
"""

import networkx as nx
from typing import Dict, List, Tuple, Set


def dag_dict_to_networkx(dag_dict: Dict[int, List[int]]) -> nx.DiGraph:
    """
    Convert a DAG dictionary representation to a NetworkX DiGraph.
    
    Args:
        dag_dict: Dictionary where keys are child nodes and values are lists of parent nodes
                  (i.e., the parents point TO the key node)
        
    Returns:
        NetworkX DiGraph representation of the DAG
    """
    G = nx.DiGraph()
    
    # Add all nodes first
    all_nodes = set(dag_dict.keys())
    for parents in dag_dict.values():
        all_nodes.update(parents)
    G.add_nodes_from(all_nodes)
    
    # Add edges from parents to child
    for child, parents in dag_dict.items():
        for parent in parents:
            G.add_edge(parent, child)
    
    return G


def cpdag_dict_to_networkx(cpdag_dict: Dict[int, Dict[str, List[int]]]) -> nx.DiGraph:
    """
    Convert a CPDAG dictionary representation to a NetworkX DiGraph for V-structure analysis.
    
    For V-structure identification, we only consider the directed edges (parents).
    Undirected edges are not part of V-structures by definition.
    
    Args:
        cpdag_dict: Dictionary where keys are nodes and values are dicts with 'parents' and 'undirected' lists
        
    Returns:
        NetworkX DiGraph representation of the directed part of the CPDAG
    """
    G = nx.DiGraph()
    
    # Add all nodes first
    all_nodes = set(cpdag_dict.keys())
    for node_data in cpdag_dict.values():
        all_nodes.update(node_data.get('parents', []))
        all_nodes.update(node_data.get('undirected', []))
    G.add_nodes_from(all_nodes)
    
    # Add only directed edges (parents -> child)
    for child, node_data in cpdag_dict.items():
        parents = node_data.get('parents', [])
        for parent in parents:
            G.add_edge(parent, child)
    
    return G


def find_v_structures(G: nx.DiGraph) -> List[Tuple[Set[int], int]]:
    """
    Find all V-structures (colliders) in a directed graph.
    
    A V-structure occurs when a node has two or more parents that are not connected
    to each other. The pattern is: X → Z ← Y (where X and Y are not connected).
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        List of tuples, where each tuple contains:
        - Set of parent nodes forming the V-structure
        - The collider (child) node
    """
    v_structures = []
    
    # Check each node to see if it's a collider
    for node in G.nodes():
        # Get all predecessors (parents) of this node
        parents = list(G.predecessors(node))
        
        # A node can be a collider only if it has 2 or more parents
        if len(parents) >= 2:
            # Check if any pair of parents are not connected
            # If they're not connected, then this node is part of a V-structure
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    parent1, parent2 = parents[i], parents[j]
                    
                    # Check if parent1 and parent2 are NOT connected
                    # (neither parent1 → parent2 nor parent2 → parent1)
                    if not G.has_edge(parent1, parent2) and not G.has_edge(parent2, parent1):
                        # Found a V-structure: parent1 → node ← parent2
                        v_structures.append(({parent1, parent2}, node))
    
    return v_structures


def find_all_v_structures_comprehensive(G: nx.DiGraph) -> Dict[int, Set[int]]:
    """
    Find all V-structures in a comprehensive way, grouping all unconnected parents
    for each collider node.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Dictionary mapping collider nodes to sets of their unconnected parent pairs
    """
    colliders = {}
    
    for node in G.nodes():
        parents = list(G.predecessors(node))
        
        if len(parents) >= 2:
            # Find all pairs of unconnected parents
            unconnected_parents = set()
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    parent1, parent2 = parents[i], parents[j]
                    
                    if not G.has_edge(parent1, parent2) and not G.has_edge(parent2, parent1):
                        unconnected_parents.update([parent1, parent2])
            
            if unconnected_parents:
                colliders[node] = unconnected_parents
    
    return colliders


def identify_v_structure_components(cpdag_dict: Dict[int, Dict[str, List[int]]]) -> Tuple[Set[int], Set[int], Set[int]]:
    """
    Identify the components of V-structures in a CPDAG for generation ordering.
    
    Args:
        cpdag_dict: CPDAG dictionary representation
        
    Returns:
        Tuple of (v_structure_parents, v_structure_children, other_nodes)
        - v_structure_parents: Nodes that are parents in V-structures
        - v_structure_children: Nodes that are children (colliders) in V-structures  
        - other_nodes: Nodes not involved in any V-structure
    """
    # Convert CPDAG to NetworkX for analysis
    G = cpdag_dict_to_networkx(cpdag_dict)
    
    # Find all V-structures
    v_structures = find_v_structures(G)
    
    # Extract all nodes involved in V-structures
    v_structure_parents = set()
    v_structure_children = set()
    
    for parents, child in v_structures:
        v_structure_parents.update(parents)
        v_structure_children.add(child)
    
    # Find all nodes in the graph
    all_nodes = set(cpdag_dict.keys())
    
    # Nodes not involved in any V-structure
    other_nodes = all_nodes - v_structure_parents - v_structure_children
    
    return v_structure_parents, v_structure_children, other_nodes


def get_v_structure_generation_order(
    cpdag_dict: Dict[int, Dict[str, List[int]]], 
    all_features: List[int],
    causal_structures_last: bool = False
) -> List[int]:
    """
    Get the generation order for V-structures based on causal_structures_last parameter.
    
    Args:
        cpdag_dict: CPDAG dictionary representation
        all_features: List of all feature indices
        causal_structures_last: If True, generate V-structures after other nodes
        
    Returns:
        List of node indices in generation order
    """
    # Identify V-structure components
    v_parents, v_children, other_nodes = identify_v_structure_components(cpdag_dict)
    
    # Convert to lists and sort for deterministic ordering
    v_parents_list = sorted(list(v_parents))
    v_children_list = sorted(list(v_children))  
    other_nodes_list = sorted(list(other_nodes))
    
    if causal_structures_last:
        # Generate other nodes first, then V-structure parents, then V-structure children
        generation_order = other_nodes_list + v_parents_list + v_children_list
    else:
        # Generate V-structure parents first, then V-structure children, then other nodes
        generation_order = v_parents_list + v_children_list + other_nodes_list
    
    return generation_order