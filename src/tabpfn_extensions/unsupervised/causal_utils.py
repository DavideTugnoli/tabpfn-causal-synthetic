#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""Causal utilities for DAG and CPDAG parsing and manipulation.

This module provides utilities for working with causal graphs in the context
of TabPFN unsupervised learning. It handles parsing of Directed Acyclic Graphs (DAGs)
and Completed Partially Directed Acyclic Graphs (CPDAGs) to support causal-aware
data generation and imputation.

Key functions:
- parse_cpdag_adjacency_matrix: Convert CPDAG adjacency matrix to dictionary format
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List


def parse_cpdag_adjacency_matrix(
    cpdag_adj: np.ndarray,
) -> Dict[int, Dict[str, List[int]]]:
    """Parse CPDAG adjacency matrix into dictionary format.
    
    Args:
        cpdag_adj: Adjacency matrix where:
                  0 = no edge
                  1 = directed edge (column -> row)
                  -1 = undirected edge (both directions)
                  
    Returns:
        Dictionary in format {node_idx: {"parents": [parent_indices], "undirected": [undirected_indices]}}
    """
    n = cpdag_adj.shape[0]
    cpdag = {}
    
    for i in range(n):
        # Find causal parents (directed edges pointing to this node)
        # Look at [i, j] to find parents j with directed edges j -> i.
        parents = [j for j in range(n) if cpdag_adj[i, j] == 1]
        
        # Find undirected neighbors (undirected edges)
        undirected = []
        for j in range(n):
            if (cpdag_adj[i, j] == -1 and cpdag_adj[j, i] == -1 and i != j):
                undirected.append(j)
        
        cpdag[i] = {"parents": parents, "undirected": undirected}
        
    return cpdag
