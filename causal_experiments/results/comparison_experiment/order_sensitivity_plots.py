#!/usr/bin/env python3
"""
Publication-ready plots for demonstrating order sensitivity in vanilla TabPFN.

This script generates line plots showing how synthetic data quality (measured by the range
across three column orderings) changes with training size. The plots demonstrate that while
order sensitivity decreases with more training data, it persists at non-negligible levels
even at the largest training sizes tested (500 samples).

Output structure (only combined panels):
    order_sensitivity_plots/
        combined_panels/
            pdf/
                order_sensitivity_combined_<dataset>.pdf
            order_sensitivity_combined_<dataset>.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fastplot  # type: ignore
import matplotlib as mpl  # FastPlot uses matplotlib under the hood

from causal_experiments.utils.visualization_config import (
    METRIC_CONFIG,
    DPI,
)
from causal_experiments.utils.forest_plot_config import (
    TITLE_FONT_SIZE,
    LABEL_FONT_SIZE,
)

# --------------------------------------------------------------------------
# Font alignment with CLeaR 2026 template (\usepackage{times})
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "serif"],
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{times}",
    # Fallback when LaTeX is not available; kept for completeness
    "mathtext.fontset": "stix",
    "mathtext.default": "regular",
})

# ============================================================================
# CONFIGURATION
# ============================================================================

# Output directory (following forest_plots_2 structure)
OUTPUT_ROOT = SCRIPT_DIR / "order_sensitivity_plots"

# Comparison experiment result files (in results subdirectory)
COMPARISON_RESULT_FILES: Tuple[str, ...] = (
    "result_csuite_large_backdoor_comparison_experiment_cleaned_reps_100.csv",
    "result_csuite_mixed_confounding_comparison_experiment_cleaned_reps_100.csv",
    "result_csuite_mixed_simpson_comparison_experiment_cleaned_reps_100.csv",
    "result_csuite_nonlin_simpson_comparison_experiment_cleaned_reps_100.csv",
    "result_csuite_symprod_simpson_comparison_experiment_cleaned_reps_100.csv",
    "result_csuite_weak_arrows_comparison_experiment_cleaned_reps_100.csv",
    "result_custom_scm_comparison_experiment_cleaned_reps_100.csv",
)

# Orderings to consider (vanilla only)
ORDERINGS = ["original", "topological", "reverse_topological"]

# Train sizes
TRAIN_SIZES = [20, 50, 100, 200, 500]

# Metrics to plot (using spearman instead of pearson for frobenius)
DEFAULT_METRICS = ["correlation_matrix_difference", "k_marginal_tvd", "nnaa"]

# Dataset acronyms (aligned with forest_plots.py)
DATASET_ACRONYMS: Dict[str, str] = {
    "csuite_large_backdoor": "CLB",
    "csuite_mixed_confounding": "CMC",
    "csuite_mixed_simpson": "CMS",
    "csuite_nonlin_simpson": "CNS",
    "csuite_symprod_simpson": "CSS",
    "csuite_weak_arrows": "CWA",
    "custom_scm": "CSM",
    "simglucose": "SGL",
}

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def _format_display_label(text: str) -> str:
    """Replace underscores with spaces and collapse whitespace for presentation."""
    return " ".join(str(text).replace("_", " ").split())


def _dataset_acronym(dataset: str, fallback_label: str | None = None) -> str:
    """Return acronym for dataset if available, otherwise a readable label."""
    return DATASET_ACRONYMS.get(dataset, fallback_label or _format_display_label(dataset))


def _get_metric_slug_and_title(metric: str) -> Tuple[str, str]:
    """Get slug and title for metric, using frobenius_corr_norm config for spearman metric."""
    if metric == "correlation_matrix_difference":
        # Use correlation_matrix_difference slug and title to match original file names
        cmd_config = METRIC_CONFIG.get("correlation_matrix_difference", METRIC_CONFIG["frobenius_corr_norm"])
        metric_slug = cmd_config["slug"]
        # Prefer long_title if available
        metric_title = cmd_config.get("long_title", cmd_config["title"])
    else:
        metric_cfg = METRIC_CONFIG[metric]
        metric_slug = metric_cfg["slug"]
        # Prefer long_title if available
        metric_title = metric_cfg.get("long_title", metric_cfg["title"])

    # Normalize titles to always use full names (no acronyms)
    # This is a fallback if long_title is not set or still contains acronyms
    if "TVD" in metric_title and "Total Variation Distance" not in metric_title:
        metric_title = metric_title.replace("TVD", "Total Variation Distance")
    
    # Robust check for NNAA
    if "NNAA" in metric_title and "Nearest-Neighbor Adversarial Accuracy" not in metric_title:
        metric_title = metric_title.replace("NNAA", "Nearest-Neighbor Adversarial Accuracy")

    return metric_slug, metric_title


def _infer_dataset_name(csv_path: Path) -> str:
    """Extract dataset name from CSV filename."""
    name = csv_path.name
    core = name[len("result_"):] if name.startswith("result_") else csv_path.stem
    if "_comparison" in core:
        core = core.split("_comparison", 1)[0]
    return core.replace(".csv", "")


def load_comparison_data() -> pd.DataFrame:
    """Load all comparison experiment CSVs and extract vanilla data."""
    frames = []

    for relative_path in COMPARISON_RESULT_FILES:
        csv_path = SCRIPT_DIR / "data" / relative_path
        if not csv_path.exists():
            print(f"[WARN] Missing file: {relative_path}")
            continue

        df = pd.read_csv(csv_path)

        # Filter for vanilla algorithm only
        if "algorithm" not in df.columns or "column_order" not in df.columns:
            print(f"[SKIP] {csv_path.name}: missing required columns")
            continue

        vanilla_df = df[df["algorithm"] == "vanilla"].copy()
        if vanilla_df.empty:
            continue

        # Add dataset identifier
        vanilla_df["dataset"] = _infer_dataset_name(csv_path)
        frames.append(vanilla_df)

    if not frames:
        raise RuntimeError("No valid data found in comparison experiment CSVs")

    combined = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Loaded {len(combined)} rows from {combined['dataset'].nunique()} datasets")
    return combined


def compute_order_sensitivity_per_seed(
    data: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    Compute order sensitivity (range across orderings) for each seed.

    Returns a DataFrame with columns:
        - dataset
        - train_size
        - seed
        - range (max - min across orderings)
        - median_value (median across orderings)
        - relative_range (range / median, if median > 0)
    """
    if metric not in data.columns:
        raise ValueError(f"Metric '{metric}' not found in data")

    rows: List[Dict[str, Any]] = []

    for (dataset, ts, seed), group in data.groupby(["dataset", "train_size", "seed"]):
        # Get values for each ordering
        values = {}
        for ordering in ORDERINGS:
            subset = group[group["column_order"] == ordering][metric]
            if len(subset) == 1:
                values[ordering] = float(subset.iloc[0])

        # Need all three orderings
        if len(values) != len(ORDERINGS):
            continue

        vals_array = np.array(list(values.values()))
        range_val = float(vals_array.max() - vals_array.min())
        median_val = float(np.median(vals_array))

        relative_range = float(range_val / median_val) if median_val > 1e-12 else np.nan

        rows.append({
            "dataset": dataset,
            "train_size": int(ts),
            "seed": int(seed),
            "range": range_val,
            "median_value": median_val,
            "relative_range": relative_range,
        })

    return pd.DataFrame(rows)


def bootstrap_ci(
    data: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for median."""
    if len(data) < 2:
        med = float(np.median(data))
        return med, np.nan, np.nan

    boot_medians = np.array([
        np.median(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])

    lower = float(np.percentile(boot_medians, 100 * alpha / 2))
    upper = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))
    median = float(np.median(data))

    return median, lower, upper


def aggregate_trend_data(
    sensitivity_df: pd.DataFrame,
    metric: str,
    datasets: List[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate per-seed sensitivity into trend data (median with bootstrap CI).

    Returns DataFrame with:
        - dataset (if datasets provided, otherwise 'aggregated')
        - train_size
        - median
        - ci_lower
        - ci_upper
        - n_seeds
    """
    if datasets is not None:
        sensitivity_df = sensitivity_df[sensitivity_df["dataset"].isin(datasets)]

    rows: List[Dict[str, Any]] = []

    # Group by dataset if multiple, otherwise aggregate all
    group_cols = ["dataset", "train_size"] if datasets is not None else ["train_size"]

    for group_key, group_df in sensitivity_df.groupby(group_cols):
        values = group_df["range"].dropna().to_numpy()

        if len(values) == 0:
            continue

        median, ci_lower, ci_upper = bootstrap_ci(values, n_boot=1000)

        row = {
            "train_size": int(group_key[-1] if isinstance(group_key, tuple) else group_key),
            "median": median,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_seeds": int(len(values)),
        }

        if datasets is not None:
            row["dataset"] = group_key[0]

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def plot_combined_single_dataset(
    dataset: str,
    metrics: List[str],
    output_dir: Path,
) -> None:
    """
    Generate a 3-panel side-by-side plot for one dataset across multiple metrics.

    Optimized for LaTeX compilation:
    - Large figsize to prevent shrinking
    - Large fonts that remain readable when scaled
    - Shared y-axis across panels for easy comparison
    """
    if not metrics or len(metrics) != 3:
        print(f"[WARN] Combined plot requires exactly 3 metrics, got {len(metrics)}")
        return

    dataset_label = _format_display_label(dataset)
    dataset_acr = _dataset_acronym(dataset, dataset_label)
    dataset_slug = dataset.replace("/", "_")

    # Load comparison data
    data = load_comparison_data()

    # Compute sensitivity for each metric
    trend_dfs = []
    valid_metrics = []

    for metric in metrics:
        if metric not in METRIC_CONFIG:
            continue

        sensitivity_df = compute_order_sensitivity_per_seed(data, metric)
        dataset_df = sensitivity_df[sensitivity_df["dataset"] == dataset]

        if dataset_df.empty:
            continue

        trend_df = aggregate_trend_data(dataset_df, metric, datasets=[dataset])
        if not trend_df.empty:
            trend_dfs.append((metric, trend_df))
            valid_metrics.append(metric)

    if len(trend_dfs) != 3:
        print(f"[WARN] Could not get data for all 3 metrics for dataset {dataset}")
        return

    # Prepare output with subdirectories
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = output_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Use frobenius_corr_norm slug for spearman metric to match original file names
    metrics_slug = "_".join([
        METRIC_CONFIG.get("correlation_matrix_difference", METRIC_CONFIG["frobenius_corr_norm"])["slug"] if m == "correlation_matrix_difference" else METRIC_CONFIG[m]["slug"]
        for m in valid_metrics
    ])
    pdf_path = pdf_dir / f"order_sensitivity_combined_{dataset_slug}.pdf"
    csv_path = output_dir / f"order_sensitivity_combined_{dataset_slug}.csv"

    # Save combined CSV
    combined_rows = []
    for metric, trend_df in trend_dfs:
        df_copy = trend_df.copy()
        df_copy["metric"] = metric
        metric_slug, metric_title = _get_metric_slug_and_title(metric)
        df_copy["metric_title"] = metric_title
        df_copy["dataset"] = dataset
        df_copy["dataset_label"] = dataset_label
        combined_rows.append(df_copy)

    combined_csv = pd.concat(combined_rows, ignore_index=True)
    combined_csv.to_csv(csv_path, index=False)

    # Plot with FastPlot - LARGE figsize for LaTeX
    # When compiled at \textwidth, these will scale down but remain readable
    figsize = (24, 7)  # Even wider format for 3 panels - extra space for larger fonts

    def draw_callback(plt_mod: Any) -> None:
        fig = plt_mod.gcf()
        fig.clear()

        axes = fig.subplots(1, 3, sharey=False)  # Don't share y-axis - different scales

        for idx, (metric, trend_df) in enumerate(trend_dfs):
            ax = axes[idx]

            x = trend_df["train_size"].to_numpy()
            y = trend_df["median"].to_numpy()
            y_lower = trend_df["ci_lower"].to_numpy()
            y_upper = trend_df["ci_upper"].to_numpy()

            # Main line with prominent styling - larger for LaTeX
            ax.plot(x, y, marker="o", linewidth=4, color="#1f77b4",
                   markersize=12, markeredgecolor="black", markeredgewidth=1.5)

            # Confidence interval
            ax.fill_between(x, y_lower, y_upper, alpha=0.25, color="#1f77b4")

            # Styling optimized for LaTeX
            metric_slug, panel_title = _get_metric_slug_and_title(metric)

            # Keep full title (no acronyms) for k-marginal
            if metric_slug == "2marginal":
                panel_title = "k-Marginal Total Variation Distance"

            ax.set_title(panel_title, fontsize=32, pad=25)
            ax.set_xlabel("Training Size", fontsize=30)

            # Only leftmost panel gets y-label
            if idx == 0:
                ax.set_ylabel("Order Sensitivity (range)", fontsize=30)

            # Custom tick positions for better spacing
            # Keep original data positions but space labels better
            custom_ticks = [20, 50, 100, 200, 500]
            ax.set_xticks(custom_ticks)

            # Minimal padding - just enough to prevent label cutoff
            # Let tight_layout handle the spacing
            ax.set_xlim(TRAIN_SIZES[0] - 10, TRAIN_SIZES[-1] + 10)

            ax.grid(True, alpha=0.3, linestyle=":", linewidth=1)
            ax.set_ylim(bottom=0)

            # Remove top and right spines for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Overall title
        fig.suptitle(
            f"Order Sensitivity — {dataset_acr}",
            fontsize=32,
            y=0.92,  # slightly higher than previous placement
        )

        # Use tight_layout for minimal whitespace
        # This automatically removes excess whitespace
        fig.tight_layout(pad=2.0, w_pad=3.0, h_pad=2.0)
        
        # Fine-tune margins for LaTeX optimization
        fig.subplots_adjust(
            left=0.08,      # Minimal left margin
            right=0.98,     # Minimal right margin  
            top=0.90,       # Minimal top margin
            bottom=0.15,    # Space for rotated x-labels
            wspace=0.25     # Space between panels
        )

        # CRITICAL: Force font sizes AFTER layout using setp (matplotlib's recommended way)
        # This overrides any automatic font size adjustments
        for ax in axes:
            # X-axis: rotated labels with larger font
            custom_ticks = [20, 50, 100, 200, 500]
            ax.set_xticklabels(custom_ticks)
            plt_mod.setp(ax.get_xticklabels(), fontsize=28, rotation=45, ha='center')
            ax.tick_params(axis="x", pad=8)  # Reduced padding for tighter layout
            
            # Ensure minimal spacing between ticks
            ax.tick_params(axis="x", which='major', length=6, width=1.5)

            # Y-axis: force same size for all panels
            plt_mod.setp(ax.get_yticklabels(), fontsize=28)

    # Save PDF
    # Use fontsize=28 as base to match our tick labels
    fastplot.plot(
        data=None,
        path=str(pdf_path),
        mode="callback",
        callback=draw_callback,
        style="serif",
        figsize=figsize,
        dpi=DPI,
        fontsize=28,
    )

    print(f"[OK] Combined 3-panel plot: {pdf_path.name}")


def main(metrics: List[str] | None = None) -> None:
    """Generate all order sensitivity plots."""
    if metrics is None:
        metrics = DEFAULT_METRICS

    print("=" * 80)
    print("ORDER SENSITIVITY PLOTS - Vanilla TabPFN")
    print("=" * 80)

    # Load data
    print("\n[1/2] Loading comparison experiment data...")
    data = load_comparison_data()

    # Generate combined plots for ALL datasets
    print("\n[2/2] Generating combined 3-panel plots for all datasets...")
    
    # Get all unique datasets from the data
    all_datasets = sorted(data["dataset"].unique())
    combined_dir = OUTPUT_ROOT / "combined_panels"
    
    for dataset in all_datasets:
        print(f"  Processing {dataset}...")
        plot_combined_single_dataset(dataset, metrics, combined_dir)
        
    print(f"\n[OK] All combined plots saved to: {combined_dir}")

    print("\n" + "=" * 80)
    print(f"All plots saved to: {OUTPUT_ROOT}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate order sensitivity plots for vanilla TabPFN"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to plot (default: all)",
    )

    args = parser.parse_args()
    main(metrics=args.metrics)
