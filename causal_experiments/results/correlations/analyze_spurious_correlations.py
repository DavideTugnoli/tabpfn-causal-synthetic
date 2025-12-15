#!/usr/bin/env python3
"""Analyze spurious correlations in synthetic data generation.

This script computes correlation coefficients between variables that should be
independent according to the causal structure of the custom SCM.

For the custom SCM structure (X3 → X2 → X1 ← X0):
- X0 and X3 should be independent (no correlation)
- X0 and X2 should be independent (X2 depends only on X3)
- X3 and X1 should be independent (X1 depends only on X0 and X2)

The script loads synthetic datasets and test sets, computes these correlations,
and creates a summary table comparing different algorithms and configurations.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_synthetic_dataset(path: Path) -> pd.DataFrame:
    """Load synthetic dataset from .npz file."""
    if not path.is_file():
        raise FileNotFoundError(f"Synthetic dataset not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        required_keys = {"synthetic_data", "column_names"}
        missing = required_keys.difference(data.files)
        if missing:
            raise KeyError(f"Missing keys {missing} in synthetic file {path}")
        synthetic_data = data["synthetic_data"]
        column_names = data["column_names"].tolist()

    return pd.DataFrame(synthetic_data, columns=column_names)


def load_test_set(path: Path) -> pd.DataFrame:
    """Load test set from .npz file."""
    if not path.is_file():
        raise FileNotFoundError(f"Test set not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        required_keys = {"X_test", "column_names"}
        missing = required_keys.difference(data.files)
        if missing:
            raise KeyError(f"Missing keys {missing} in test set file {path}")
        X_test = data["X_test"]
        column_names = data["column_names"].tolist()

    return pd.DataFrame(X_test, columns=column_names)


def resolve_data_path(
    path_str: str, synthetic_path: Path | None = None, input_csv: Path | None = None
) -> Path:
    """Resolve relative path to absolute path.
    
    Args:
        path_str: Path string (can be relative or absolute)
        synthetic_path: Optional path to synthetic data file (used as reference for relative paths)
        input_csv: Optional path to input CSV (used as additional reference)
    """
    if not path_str or pd.isna(path_str):
        raise ValueError(f"Empty or NaN path: {path_str}")

    path = Path(path_str.strip())
    if path.is_absolute():
        return path.resolve()

    # Try multiple base directories (similar to recompute_csuite_metrics.py)
    candidates = []
    
    if synthetic_path is not None:
        # Try relative to synthetic file location
        candidates.extend([
            synthetic_path.parents[1] / path,
            synthetic_path.parents[2] / path,
            synthetic_path.parents[2] / path.name,
        ])
    
    if input_csv is not None:
        # Try relative to input CSV location
        candidates.append(input_csv.parent / path)
    
    # Try relative to project root
    candidates.extend([
        project_root / path,
        project_root / path_str,
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve path '{path_str}'. "
        f"Tried: {candidates}"
    )


def compute_spurious_correlations(
    df: pd.DataFrame, column_names: List[str]
) -> Dict[str, float]:
    """Compute spurious correlations for custom SCM.

    For custom SCM (X3 → X2 → X1 ← X0), the following pairs should be independent:
    - X0 and X3 (both root nodes, no common ancestor)
    - X0 and X2 (X2 depends only on X3, not X0)

    NOTE: X3 and X1 are NOT independent! There is an open causal path X3 → X2 → X1,
    so the correlation between X3 and X1 (~0.708) is a legitimate causal correlation,
    not a spurious one.

    Args:
        df: DataFrame with columns matching column_names
        column_names: List of column names in order [X0, X1, X2, X3]

    Returns:
        Dictionary with correlation coefficients for each spurious pair
    """
    # Map column names to indices
    col_to_idx = {name: idx for idx, name in enumerate(column_names)}

    # Verify we have the expected columns
    expected_cols = ["X0", "X1", "X2", "X3"]
    for col in expected_cols:
        if col not in col_to_idx:
            raise ValueError(
                f"Expected column '{col}' not found in column_names: {column_names}"
            )

    correlations = {}

    # X0 and X3 should be independent (both root nodes)
    x0_col = df["X0"]
    x3_col = df["X3"]
    if len(x0_col) > 1:
        corr_x0_x3, _ = pearsonr(x0_col, x3_col)
        correlations["corr_X0_X3"] = corr_x0_x3
    else:
        correlations["corr_X0_X3"] = np.nan

    # X0 and X2 should be independent (X2 depends only on X3, not X0)
    x2_col = df["X2"]
    if len(x0_col) > 1:
        corr_x0_x2, _ = pearsonr(x0_col, x2_col)
        correlations["corr_X0_X2"] = corr_x0_x2
    else:
        correlations["corr_X0_X2"] = np.nan

    # NOTE: We do NOT compute X3-X1 correlation as spurious because:
    # X3 and X1 are NOT independent - there is an open causal path X3 → X2 → X1
    # The correlation between X3 and X1 (~0.708) is a legitimate causal correlation

    return correlations


def analyze_csv(
    csv_path: Path,
    output_dir: Path,
    limit: int | None = None,
    train_sizes: List[int] | None = None,
) -> None:
    """Analyze spurious correlations from experiment CSV.

    Args:
        csv_path: Path to experiment results CSV
        output_dir: Directory to save output files
        limit: Optional limit on number of rows to process
        train_sizes: Optional list of train sizes to filter (if None, use all)
    """
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    if limit is not None:
        df = df.head(limit)
        print(f"Limited to first {limit} rows")

    if train_sizes is not None:
        df = df[df["train_size"].isin(train_sizes)]
        print(f"Filtered to train_sizes: {train_sizes}")

    # Filter to only specific configurations (paper configurations)
    # These are the configurations used in the paper:
    # - vanilla: original, topological, reverse_topological
    # - dag: topological
    # - cpdag_minimal: original
    # - cpdag_discovered: original
    paper_configs = {
        ("vanilla", "original"),
        ("vanilla", "topological"),
        ("vanilla", "reverse_topological"),
        ("dag", "topological"),
        ("cpdag_minimal", "original"),
        ("cpdag_discovered", "original"),
    }
    
    # Create mask for paper configurations
    mask = df.apply(
        lambda row: (row.get("algorithm", ""), row.get("column_order", "")) in paper_configs,
        axis=1
    )
    df = df[mask]
    print(f"Filtered to paper configurations: {len(df)} rows remaining")

    print(f"Processing {len(df)} rows...")

    results: List[Dict] = []
    test_set_cache: Dict[str, pd.DataFrame] = {}

    for idx, row in df.iterrows():
        try:
            # Get paths
            synthetic_path_str = row.get("synthetic_data_path", "")
            test_path_str = row.get("test_dataset_path", "")

            if not synthetic_path_str or pd.isna(synthetic_path_str):
                print(f"  Row {idx}: Skipping (no synthetic_data_path)")
                continue

            if not test_path_str or pd.isna(test_path_str):
                print(f"  Row {idx}: Skipping (no test_dataset_path)")
                continue

            # Resolve paths
            synthetic_path = resolve_data_path(synthetic_path_str, input_csv=csv_path)
            test_path = resolve_data_path(test_path_str, synthetic_path, input_csv=csv_path)

            # Load test set (cache it)
            test_key = str(test_path)
            if test_key not in test_set_cache:
                test_set_cache[test_key] = load_test_set(test_path)

            # Load synthetic data
            synthetic_df = load_synthetic_dataset(synthetic_path)
            test_df = test_set_cache[test_key]

            # Ensure same column order
            column_names = test_df.columns.tolist()
            synthetic_df = synthetic_df[column_names]

            # Compute correlations for synthetic data
            synthetic_corrs = compute_spurious_correlations(synthetic_df, column_names)

            # Compute correlations for test data (baseline)
            test_corrs = compute_spurious_correlations(test_df, column_names)

            # Create result row
            result_row = {
                "algorithm": row.get("algorithm", ""),
                "column_order": row.get("column_order", ""),
                "train_size": row.get("train_size", ""),
                "seed": row.get("seed", ""),
                "repetition": row.get("repetition", ""),
                # Synthetic correlations (spurious pairs only)
                "synthetic_corr_X0_X3": synthetic_corrs["corr_X0_X3"],
                "synthetic_corr_X0_X2": synthetic_corrs["corr_X0_X2"],
                # Test correlations (baseline)
                "test_corr_X0_X3": test_corrs["corr_X0_X3"],
                "test_corr_X0_X2": test_corrs["corr_X0_X2"],
                # Differences (synthetic - test)
                "diff_corr_X0_X3": synthetic_corrs["corr_X0_X3"]
                - test_corrs["corr_X0_X3"],
                "diff_corr_X0_X2": synthetic_corrs["corr_X0_X2"]
                - test_corrs["corr_X0_X2"],
                # Absolute differences
                "abs_diff_corr_X0_X3": abs(
                    synthetic_corrs["corr_X0_X3"] - test_corrs["corr_X0_X3"]
                ),
                "abs_diff_corr_X0_X2": abs(
                    synthetic_corrs["corr_X0_X2"] - test_corrs["corr_X0_X2"]
                ),
            }

            results.append(result_row)

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)} rows...")

        except Exception as e:
            print(f"  Row {idx}: Error - {e}")
            continue

    if not results:
        print("No results to save!")
        return

    # Create detailed results DataFrame
    results_df = pd.DataFrame(results)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create summary table aggregated by (algorithm, column_order, train_size)
    summary_rows: List[Dict] = []

    for (algo, order, train_size), group in results_df.groupby(
        ["algorithm", "column_order", "train_size"]
    ):
        summary_row = {
            "algorithm": algo,
            "column_order": order,
            "train_size": train_size,
            "n_samples": len(group),
            # Mean synthetic correlations (spurious pairs only)
            "mean_synthetic_corr_X0_X3": group["synthetic_corr_X0_X3"].mean(),
            "mean_synthetic_corr_X0_X2": group["synthetic_corr_X0_X2"].mean(),
            # Std synthetic correlations
            "std_synthetic_corr_X0_X3": group["synthetic_corr_X0_X3"].std(),
            "std_synthetic_corr_X0_X2": group["synthetic_corr_X0_X2"].std(),
            # Mean test correlations (baseline)
            "mean_test_corr_X0_X3": group["test_corr_X0_X3"].mean(),
            "mean_test_corr_X0_X2": group["test_corr_X0_X2"].mean(),
            # Mean absolute differences
            "mean_abs_diff_corr_X0_X3": group["abs_diff_corr_X0_X3"].mean(),
            "mean_abs_diff_corr_X0_X2": group["abs_diff_corr_X0_X2"].mean(),
            # Max absolute differences
            "max_abs_diff_corr_X0_X3": group["abs_diff_corr_X0_X3"].max(),
            "max_abs_diff_corr_X0_X2": group["abs_diff_corr_X0_X2"].max(),
        }
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["algorithm", "column_order", "train_size"]
    )

    # Save summary
    summary_output = output_dir / "spurious_correlations_summary.csv"
    summary_df.to_csv(summary_output, index=False)
    print(f"Saved summary to: {summary_output}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nTotal configurations: {len(summary_df)}")
    print(f"\nAlgorithms: {summary_df['algorithm'].unique().tolist()}")
    print(f"Column orders: {summary_df['column_order'].unique().tolist()}")
    print(f"Train sizes: {sorted(summary_df['train_size'].unique().tolist())}")

    # Print top configurations with highest spurious correlations
    print("\n" + "-" * 80)
    print("TOP 10 CONFIGURATIONS BY MEAN ABSOLUTE DIFFERENCE (X0-X3)")
    print("-" * 80)
    top_x0_x3 = summary_df.nlargest(10, "mean_abs_diff_corr_X0_X3")[
        [
            "algorithm",
            "column_order",
            "train_size",
            "mean_synthetic_corr_X0_X3",
            "mean_test_corr_X0_X3",
            "mean_abs_diff_corr_X0_X3",
        ]
    ]
    print(top_x0_x3.to_string(index=False))

    print("\n" + "-" * 80)
    print("TOP 10 CONFIGURATIONS BY MEAN ABSOLUTE DIFFERENCE (X0-X2)")
    print("-" * 80)
    top_x0_x2 = summary_df.nlargest(10, "mean_abs_diff_corr_X0_X2")[
        [
            "algorithm",
            "column_order",
            "train_size",
            "mean_synthetic_corr_X0_X2",
            "mean_test_corr_X0_X2",
            "mean_abs_diff_corr_X0_X2",
        ]
    ]
    print(top_x0_x2.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze spurious correlations in synthetic data generation"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "causal_experiments/results/comparison_experiment/data/"
            "result_custom_scm_comparison_experiment_cleaned_reps_100.csv"
        ),
        help="Path to input CSV file with experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("causal_experiments/results/correlations"),
        help="Directory to save output files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of rows to process (for testing)",
    )
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of train sizes to filter (e.g., --train-sizes 20 50 100)",
    )

    args = parser.parse_args()

    analyze_csv(
        csv_path=args.input,
        output_dir=args.output_dir,
        limit=args.limit,
        train_sizes=args.train_sizes,
    )


if __name__ == "__main__":
    main()

