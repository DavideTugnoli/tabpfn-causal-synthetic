# TabPFN Unsupervised Module - Feature Order Experiments

## Overview

This README documents the **Causal Structures Last** feature added to the TabPFN Unsupervised module. This enhancement allows experimenting with different generation orders for causal graphs, specifically testing whether generating causal structures (collider patterns like X1 → X2 ← X3) at the end of the process rather than at the beginning produces more realistic synthetic data.

## Background

### Standard Approach (Default)
The original implementation follows this generation order:
```
1. Parent nodes (sources of causal structures)
2. Causal children (targets of directed edges)  
3. Correlational nodes (nodes with only undirected connections)
```

### Causal-Last Approach (New)
The new approach reverses this order:
```
1. Correlational nodes (establish baseline correlational structure)
2. Parent nodes (sources of causal structures)
3. Causal children (targets of directed edges)
```

## Motivation

The **causal-last approach** was introduced to test the hypothesis that:

1. **Avoid early causal bias**: Generating causal parents first might introduce spurious correlations through undirected connections
2. **Natural correlation establishment**: Let the correlational structure form naturally before adding directed causal relationships  
3. **Realistic collider generation**: Collider structures (A → B ← C) might be more realistic when added as a "refinement" to an already-established correlational base
4. **Reduced interference**: Prevent causal structures from influencing the generation of purely correlational relationships

## API Changes

### New Parameter: `causal_structures_last`

All relevant methods now support the `causal_structures_last: bool = False` parameter:

- `TabPFNUnsupervisedModel.impute()`
- `TabPFNUnsupervisedModel.generate_synthetic_data()`
- `TabPFNUnsupervisedModel.impute_()` (internal)

### Usage Examples

#### Standard Generation (Default Behavior)
```python
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

model = TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
model.fit(X_train)

# Standard approach: causal structures generated first
synthetic_data = model.generate_synthetic_data(
    n_samples=1000,
    cpdag=my_cpdag,
    causal_structures_last=False  # Default
)
```

#### Causal-Last Generation (New Approach)
```python
# Causal-last approach: correlational structure first, then causal structures
synthetic_data = model.generate_synthetic_data(
    n_samples=1000,
    cpdag=my_cpdag,
    causal_structures_last=True  # New feature
)
```

#### Works with All Graph Types

The feature works with all supported graph types:

```python
# With CPDAG (original)
synthetic_data = model.generate_synthetic_data(
    cpdag=my_cpdag,
    causal_structures_last=True
)

# With CPDAG (original approach)
synthetic_data = model.generate_synthetic_data(
    cpdag=my_cpdag,
    causal_structures_last=True
)

# With DAG (though less relevant since DAG already enforces topological order)
synthetic_data = model.generate_synthetic_data(
    dag=my_dag,
    causal_structures_last=True
)
```

## Technical Implementation

### Modified Methods

1. **`_get_cpdag_original_ordering()`**: Updated to support causal-last ordering
2. **`impute_()`**: Passes the parameter to ordering methods
4. **`impute()` and `generate_synthetic_data()`**: Expose the parameter in public API

### Generation Order Logic

#### Standard Ordering (`causal_structures_last=False`)
```python
final_ordering = parent_nodes + ordered_causal + non_parent_nodes
```

#### Causal-Last Ordering (`causal_structures_last=True`)
```python
final_ordering = non_parent_nodes + parent_nodes + ordered_causal
```

### Debug Output

The debug prints clearly indicate which approach is being used:
```
=== DEBUG CPDAG ORIGINAL ORDERING ===
Causal structures last: True
Final ordering (causal-last): [4, 5, 6, 0, 1, 2, 3]
```

## Experimental Considerations

### When to Use Causal-Last Approach

Consider using `causal_structures_last=True` when:

1. **Complex correlational structure**: Your dataset has rich undirected relationships
2. **Collider bias concerns**: You suspect early causal generation might introduce bias
3. **Comparative studies**: You want to test both approaches and compare results
4. **Realistic modeling**: You believe correlations establish before causal relationships in your domain

### Evaluation Metrics

To evaluate the effectiveness of the causal-last approach, consider measuring:

1. **Causal structure preservation**: How well are collider patterns preserved?
2. **Correlation matrix similarity**: Does the correlational structure match better?
3. **Statistical tests**: Distribution differences between real and synthetic data
4. **Domain-specific metrics**: Task-specific evaluation criteria

### Expected Use Cases

This feature is particularly relevant for:

- **Social networks**: Where correlations often precede causal influences
- **Economic data**: Where market correlations exist independently of causal relationships
- **Biological systems**: Where correlation networks form before specific causal pathways emerge
- **Research contexts**: Where understanding generation order effects is scientifically valuable

## Compatibility

- **Backward compatible**: Default behavior unchanged (`causal_structures_last=False`)
- **All graph types supported**: Works with DAG and CPDAG
- **Existing code unaffected**: No changes required for current implementations

## Future Work

Potential extensions of this feature:

1. **Adaptive ordering**: Automatically determine optimal ordering based on graph structure
2. **Mixed strategies**: Different ordering for different parts of the graph
3. **Performance analysis**: Systematic comparison of both approaches across domains
4. **Integration with experiments framework**: Support in comparison experiments

## Hybrid Nodes Fix (January 2025)

### Problem Identified
The original CPDAG v1 implementation incorrectly handled **hybrid nodes** - nodes that are both causal parents AND have undirected connections.

**Example**: In structure `x0 → x1 ← x2 - x3`, node `x2` is hybrid:
- It's a causal parent of `x1` (directed edge)
- It has an undirected connection with `x3`

### Original Behavior (Incorrect)
Fixed strategy applied to ALL hybrid nodes regardless of generation order:
- **Always**: `x2` uses vanilla strategy → conditioned on all previous nodes
- **Problem**: This applied even when undirected neighbors weren't processed yet

### Improved Behavior (Dynamic Strategy)
Hybrid nodes now use **dynamic strategy** based on processing order:
- **If processed BEFORE undirected neighbors**: Treat as independent parent (no conditioning)
- **If processed AFTER undirected neighbors**: Use vanilla strategy to mitigate bias

**Example with `x0 → x1 ← x2 - x3`**:
- **Standard ordering** `[0, 2, 1, 3]`: x2 before x3 → `x2` independent (`[]`)
- **Causal-last ordering** `[3, 0, 2, 1]`: x2 after x3 → `x2` vanilla (`[3, 0]`)

### Technical Implementation
```python
# Dynamic strategy assignment during generation
elif strategy_mapping[column_idx] == "hybrid_dynamic":
    undirected_neighbors = cpdag_dict[column_idx]["undirected"]
    already_processed_undirected = [n for n in undirected_neighbors if n in all_features[:i]]
    
    if already_processed_undirected:
        # Undirected neighbors processed → use vanilla to mitigate bias
        conditional_idx = all_features[:i] if i > 0 else []
        actual_strategy = "hybrid_vanilla"
    else:
        # No undirected neighbors processed → treat as independent parent
        conditional_idx = []
        actual_strategy = "hybrid_independent"
```

### Scientific Rationale
The bias from undirected edges only occurs when **both nodes** of the undirected connection have been considered. When processing the first node, there's no bias to mitigate yet, so independent generation is more appropriate.

## Implementation Date

- **Added**: January 2025
- **Hybrid Nodes Fix**: January 2025
- **Context**: Part of causal machine learning research on TabPFN synthetic data generation