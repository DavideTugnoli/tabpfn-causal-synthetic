"""Measure conditional correlation in collider SCM: X3 → X2 → X1 ← X0."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Fix imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from causal_experiments.utils.scm_data import generate_numeric_scm_data


def conditional_correlation(df: pd.DataFrame, n_bins: int = 10, verbose: bool = False) -> float:
    """Compute correlation X0-X2 conditioned on X1 using quantile-based stratification.

    This function implements conditional correlation via stratification:
    1. Split conditioning variable X1 into percentile-based bins
    2. Compute correlation X0-X2 within each bin
    3. Return the average correlation across valid bins

    Binning methodology:
    - Use percentiles from 10% to 90% to avoid extremes
    - Create n_bins intervals: first is -∞ to 10th percentile, last is 90th percentile to +∞
    - Require at least 20 samples per bin for statistical stability

    Args:
        df: DataFrame with columns X0, X1, X2 from the structural causal model
        n_bins: Number of bins to stratify X1 (default: 10)
        verbose: If True, print details for each bin (default: False)

    Returns:
        float: Average Pearson correlation X0-X2 across valid bins.
               Returns 0.0 if no bin has enough samples.

    Notes:
        - Stratification approximates conditioning E[X0*X2 | X1 = x1]
        - Bins with <20 samples are skipped to avoid unstable correlations
        - 10-90% percentiles balance robustness and data coverage
    """
    # Compute percentiles to define bin boundaries (exclude extremes 0-10% and 90-100%)
    x1_percentiles = np.atleast_1d(np.percentile(df["X1"], np.linspace(10, 90, n_bins - 1)))

    if verbose:
        print(f"[DEBUG] Stratifying X1 into {n_bins} bins using 10-90% percentiles")
        print(f"[DEBUG] Percentiles: {x1_percentiles}")
        print(f"[DEBUG] Range X1: [{df['X1'].min():.4f}, {df['X1'].max():.4f}]")

    correlations = []
    valid_bins = 0

    # Create bins: [-∞, p10], [p10, p20], ..., [p90, +∞]
    boundaries = [-np.inf, *x1_percentiles.tolist(), np.inf]

    for i, (low, high) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        # Mask for samples in the current bin (left-exclusive, right-inclusive)
        mask = (df["X1"] > low) & (df["X1"] <= high)
        subset = df[mask]

        # Compute correlation only if there are enough samples (threshold: 20)
        if len(subset) > 20:
            corr, p_value = pearsonr(subset["X0"], subset["X2"])
            correlations.append(corr)
            valid_bins += 1

            if verbose:
                range_str = f"(-∞, {high:.4f}]" if low == -np.inf else f"({low:.4f}, +∞)" if high == np.inf else f"({low:.4f}, {high:.4f}]"
                print(f"[DEBUG] Bin {i+1}: {range_str}, n={len(subset):4d}, r={corr:+.4f}, p={p_value:.4f}")
        else:
            if verbose:
                range_str = f"(-∞, {high:.4f}]" if low == -np.inf else f"({low:.4f}, +∞)" if high == np.inf else f"({low:.4f}, {high:.4f}]"
                print(f"[DEBUG] Bin {i+1}: {range_str}, n={len(subset):4d}, SKIPPED (n<20)")

    if verbose:
        print(f"[DEBUG] Valid bins: {valid_bins}/{n_bins}")
        if correlations:
            print(f"[DEBUG] Conditional correlation mean: {np.mean(correlations):.4f}")
            print(f"[DEBUG] Standard deviation: {np.std(correlations):.4f}")

    return float(np.mean(correlations)) if correlations else 0.0


def measure_collider_bias(n_samples: int = 15000) -> dict:
    """Measure collider bias in the optimized SCM.

    Returns:
        Dict with marginal, conditional correlations and bias strength
    """
    # Generate optimized SCM data
    data = generate_numeric_scm_data(n_samples=n_samples, random_state=42)
    df = pd.DataFrame(data, columns=["X0", "X1", "X2", "X3"])

    # Marginal correlation X0-X2
    marginal = abs(np.corrcoef(df["X0"], df["X2"])[0, 1])

    # Conditional correlation X0-X2 | X1
    conditional = conditional_correlation(df, verbose=True)

    return {
        "marginal": marginal,
        "conditional": conditional,
        "bias_strength": conditional - marginal
    }


def main():
    """Test collider bias."""
    measure_collider_bias()



if __name__ == "__main__":
    main()