"""
Simple intervention remapping for CSuite experiments.
TabPFN maps all categorical variables to {0, 1, 2, ...}, so we just remap interventions back.
"""

from typing import Dict, Any
import numpy as np

def remap_interventions(
    synthetic_data: np.ndarray,
    intervention_mappings: Dict[int, Dict[float, Any]]
) -> np.ndarray:
    """Remap intervention columns back to original values (robust to -0.0/0.0 and float keys)."""
    result = synthetic_data.copy()
    for col_idx, mapping in intervention_mappings.items():
        for i in range(result.shape[0]):
            val = result[i, col_idx]
            # Normalizza -0.0 a 0.0
            if isinstance(val, float) and val == -0.0:
                val = 0.0
            tabpfn_val = float(np.round(val))
            mapped = None
            if tabpfn_val in mapping:
                mapped = mapping[tabpfn_val]
            elif tabpfn_val == 0.0 and (-0.0 in mapping):
                mapped = mapping[-0.0]
            if mapped is not None:
                result[i, col_idx] = mapped
            else:
                import warnings
                warnings.warn(f"[remap_interventions] Value {val} (col {col_idx}) not found in mapping {mapping}. Leaving as is.")
    return result

def create_intervention_mapping_from_csuite(
    interventions_data: Dict[str, Any]
) -> Dict[int, Dict[int, Any]]:
    """Create intervention mappings from CSuite interventions.json."""
    intervention_mappings = {}
    intervention_values = {}
    
    # Extract intervention values from interventions.json
    if 'environments' in interventions_data:
        for env in interventions_data['environments']:
            if 'intervention_idxs' in env:
                for idx in env['intervention_idxs']:
                    if idx not in intervention_values:
                        intervention_values[idx] = set()
                    
                    # Collect all intervention values
                    if 'intervention_reference' in env:
                        intervention_values[idx].update(env['intervention_reference'])
                    if 'intervention_values' in env:
                        intervention_values[idx].update(env['intervention_values'])
    
    # Create mappings: TabPFN {0, 1, ...} → original values
    for col_idx, values in intervention_values.items():
        unique_values = sorted(values)
        intervention_mappings[col_idx] = {i: val for i, val in enumerate(unique_values)}
    
    return intervention_mappings