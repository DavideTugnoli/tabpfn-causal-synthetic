#!/usr/bin/env python3
"""
Generate monochrome-safe forest plots comparing Vanilla and related baselines with multiple
comparator conditions.

Each plot shows Hodges–Lehmann effect sizes (with 95% confidence intervals) for a
pairwise comparison, grouped by dataset on the y-axis and by train size via color/offset
and marker shape. Positive values always indicate that the comparator condition
outperforms the baseline according to the metric-specific orientation described in
`METRIC_DIRECTIONS`. Non-significant contrasts are rendered with dashed error bars to
remain interpretable even without color.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import sys
import textwrap
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_experiments.results.comparison_experiment.statistical_tests import (
    CI_ALPHA_DEFAULT,
    StatTestConfig,
    run_stat_tests_for_dataset,
)

# FastPlot provides the publication-ready styling layer we rely on for every figure.
import fastplot  # type: ignore

FormatStrFormatter = fastplot.mpl.ticker.FormatStrFormatter  # type: ignore[attr-defined]
MaxNLocator = fastplot.mpl.ticker.MaxNLocator  # type: ignore[attr-defined]
Line2D = fastplot.mpl.lines.Line2D  # type: ignore[attr-defined]

from causal_experiments.utils.visualization_config import (
    METRIC_CONFIG,
    DPI,
    setup_plotting,
)

# Helper to get CMD config (handles both correlation_matrix_difference and legacy frobenius_corr_norm)
def _get_cmd_config():
    """Get Correlation Matrix Difference config, preferring correlation_matrix_difference over frobenius_corr_norm."""
    return METRIC_CONFIG.get("correlation_matrix_difference", METRIC_CONFIG["frobenius_corr_norm"])
from causal_experiments.utils.forest_plot_utils import (
    ForestRow,
    ComparisonSpec,
    NnaaCheckSummary,
    save_with_fastplot,
    render_forest_panel,
    build_forest_dataframe,
    setup_forest_plot_style,
    _slugify_label,
    _comparison_output_slug,
    _build_caption_text,
    _format_display_label,
    _wrap_xlabel,
    _normalize_metric_title,
    apply_xaxis_tick_locator,
    _validate_nnaa_metric,
    _first_valid_value_with_source,
    _first_valid_value,
    _normalize_effect,
    _orient_scalar,
    _maybe_float,
    _maybe_bool,
    _extract_oriented_ci,
    _transform_effect_for_direction,
    _build_offsets,
    _build_marker_map,
    _ensure_dir,
    TITLE_FONT_SIZE,
    TITLE_FONT_SIZE_SINGLE,
    TITLE_FONT_SIZE_SUBPLOT_COMBINED,
    TITLE_FONT_SIZE_SUPTITLE_COMBINED,
    LABEL_FONT_SIZE,
    LABEL_FONT_SIZE_SINGLE,
    LABEL_FONT_SIZE_COMBINED,
    TICK_FONT_SIZE,
    TICK_FONT_SIZE_SINGLE,
    TICK_FONT_SIZE_SINGLE_ENHANCED,
    TICK_FONT_SIZE_COMBINED,
    LEGEND_FONT_SIZE,
    LEGEND_FONT_SIZE_SINGLE,
    LEGEND_FONT_SIZE_COMBINED,
    LEGEND_TITLE_FONT_SIZE,
    LEGEND_TITLE_FONT_SIZE_SINGLE,
    LEGEND_TITLE_FONT_SIZE_COMBINED,
    CAPTION_FONT_SIZE,
    CAPTION_FONT_SIZE_SINGLE,
    CAPTION_FONT_SIZE_ENHANCED,
    AXES_TITLE_PAD,
    SUPTITLE_Y,
    X_LABEL_PAD,
    SINGLE_BOTTOM_MARGIN,
    SINGLE_TOP_MARGIN,
    SINGLE_LEFT_MARGIN,
    SINGLE_RIGHT_MARGIN,
    COMBINED_TITLE_PAD,
    COMBINED_SHARED_XLABEL_Y,
    COMBINED_LEGEND_Y,
    COMBINED_BOTTOM_MARGIN,
    COMBINED_BOTTOM_MARGIN_NO_LEGEND,
    COMBINED_CAPTION_Y,
    COMBINED_SUPTITLE_Y,
    COMBINED_TOP_MARGIN,
    COMBINED_WSPACE,
    COMBINED_WSPACE_DUO,
    DATASET_SPACING,
    TICK_PAD_Y,
    TICK_PAD_X,
    TICK_PAD_X_SINGLE,
    TICK_PAD_X_COMBINED,
    MARKER_SIZE_PLOT,
    MARKER_SIZE_LEGEND_SINGLE,
    MARKER_SIZE_LEGEND_COMBINED,
    ERROR_BAR_LINEWIDTH,
    ERROR_BAR_CAPSIZE,
    MARKER_EDGEWIDTH,
    MARKER_EDGEWIDTH_LEGEND,
    LEGEND_COLUMNSPACING_SINGLE,
    LEGEND_COLUMNSPACING_COMBINED,
    LEGEND_HANDLETEXTPAD,
    Y_TICK_LABEL_OFFSET_SINGLE,
    Y_TICK_LABEL_OFFSET_COMBINED,
    Y_LABEL_EXTRA_PAD,
    Y_LABEL_EXTRA_PAD_COMBINED,
)

COMPARISON_RESULTS_DIR = SCRIPT_DIR / "comparison_results"
# Output folder dedicated to monochrome-safe visuals.
FOREST_ROOT = SCRIPT_DIR / "forest_plots"
PAPER_ROOT = FOREST_ROOT / "paper" / "comparison_experiment"

# Fixed figure sizes to ensure consistent dimensions across plots
SINGLE_FIGSIZE = (11.0, 7.0)
COMBINED_FIGSIZE_DUO = (18.0, 6.5)  # Reduced height for better LaTeX fit

# Orientation config: value indicates how to orient the effect so that positive values
# reflect the comparator condition outperforming the baseline.
# - 'lower': smaller metric is better ⇒ keep baseline - comparator (positive = comparator better)
# - 'higher': larger metric is better ⇒ invert the sign so that positive = comparator better
METRIC_DIRECTIONS: Dict[str, str] = {
    "correlation_matrix_difference": "lower",
    "k_marginal_tvd": "lower",
    "nnaa": "lower",
}

# Default forest metrics to plot if none supplied via CLI.
DEFAULT_METRICS: Tuple[str, ...] = tuple(METRIC_DIRECTIONS.keys())

COMPARISON_RESULT_FILES: Tuple[str, ...] = (
    "data/result_csuite_large_backdoor_comparison_experiment_cleaned_reps_100.csv",
    "data/result_csuite_mixed_confounding_comparison_experiment_cleaned_reps_100.csv",
    "data/result_csuite_mixed_simpson_comparison_experiment_cleaned_reps_100.csv",
    "data/result_csuite_nonlin_simpson_comparison_experiment_cleaned_reps_100.csv",
    "data/result_csuite_symprod_simpson_comparison_experiment_cleaned_reps_100.csv",
    "data/result_csuite_weak_arrows_comparison_experiment_cleaned_reps_100.csv",
    "data/result_custom_scm_comparison_experiment_cleaned_reps_100.csv",
    "data/result_simglucose_comparison_experiment_cleaned_reps_100.csv",
)

CONDITION_COLORS: Dict[str, str] = {
    "vanilla_original": "#1f77b4",
    "vanilla_topological": "#ff7f0e",
    "vanilla_reverse_topological": "#d62728",
    "dag_topological": "#8c564b",
    "cpdag_minimal_original": "#7f7f7f",
    "cpdag_discovered_original": "#9467bd",
}

CONDITION_ORDER: Tuple[str, ...] = tuple(CONDITION_COLORS.keys())




# Combined-panel defaults
COMBINED_METRIC_KEYS: Tuple[str, ...] = (
    "correlation_matrix_difference",
    "k_marginal_tvd",
    "nnaa",
)
COMBINED_METRIC_KEYS_DUO: Tuple[str, ...] = (
    "correlation_matrix_difference",
    "k_marginal_tvd",
)
COMBINED_METRIC_KEYS_FROBENIUS_NNAA: Tuple[str, ...] = (
    "correlation_matrix_difference",
    "nnaa",
)
COMBINED_COMPARISON_KEY = "cross_dag_topological_vs_vanilla_original"
COMBINED_VANILLA_ORDERING_KEY = "vanilla_ordering_combined"
COMBINED_SUBDIR = "combined_plots"

# Mapping to control which comparisons are exported for the paper bundle and how they
# should be named on disk (baseline order preserved for Overleaf integration).
PAPER_FOLDER_MAP: Dict[Tuple[str, str], str] = {
    (
        "vanilla_original",
        "cpdag_discovered_original",
    ): "vanilla_original_vs_cpdag_original_discovered",
    (
        "vanilla_original",
        "cpdag_minimal_original",
    ): "vanilla_original_vs_cpdag_original_minimal",
    (
        "vanilla_original",
        "dag_topological",
    ): "vanilla_original_vs_dag_topological",
    (
        "vanilla_original",
        "vanilla_topological",
    ): "vanilla_original_vs_vanilla_topological",
    (
        "vanilla_original",
        "vanilla_reverse_topological",
    ): "vanilla_original_vs_vanilla_reverse_topological",
    (
        "vanilla_topological",
        "dag_topological",
    ): "vanilla_topological_vs_dag_topological",
}




COMPARISON_SPECS: Dict[str, ComparisonSpec] = {
    "original_cpdag_minimal_vs_vanilla": ComparisonSpec(
        slug="cpdag_minimal_vs_vanilla",
        baseline="vanilla_original",
        comparator="cpdag_minimal_original",
        baseline_label="Vanilla Original",
        comparator_label="CPDAG Minimal",
        title="Vanilla Original vs CPDAG Minimal",
        group_slug="ordering_original",
        group_label="Original ordering",
    ),
    "original_cpdag_discovered_vs_vanilla": ComparisonSpec(
        slug="cpdag_discovered_vs_vanilla",
        baseline="vanilla_original",
        comparator="cpdag_discovered_original",
        baseline_label="Vanilla Original",
        comparator_label="CPDAG Discovered",
        title="Vanilla Original vs CPDAG Discovered",
        group_slug="ordering_original",
        group_label="Original ordering",
    ),
    "topological_dag_vs_vanilla": ComparisonSpec(
        slug="dag_vs_vanilla_topological",
        baseline="vanilla_topological",
        comparator="dag_topological",
        baseline_label="Vanilla Topological",
        comparator_label="DAG",
        title="Vanilla Topological vs DAG",
        group_slug="ordering_topological",
        group_label="Topological ordering",
    ),
    "cross_dag_topological_vs_vanilla_original": ComparisonSpec(
        slug="dag_topological_vs_vanilla_original",
        baseline="vanilla_original",
        comparator="dag_topological",
        baseline_label="Vanilla Original",
        comparator_label="DAG",
        title="Vanilla Original vs DAG",
        group_slug="ordering_cross/dag_vs_vanilla",
        group_label="Cross ordering · DAG vs Vanilla",
    ),
    "cross_vanilla_topological_vs_original": ComparisonSpec(
        slug="vanilla_topological_vs_vanilla_original_cross",
        baseline="vanilla_original",
        comparator="vanilla_topological",
        baseline_label="Vanilla Original",
        comparator_label="Vanilla Topological",
        title="Vanilla Original vs Vanilla Topological",
        group_slug="ordering_cross/vanilla",
        group_label="Cross ordering · Vanilla",
    ),
    "cross_vanilla_reverse_topological_vs_original": ComparisonSpec(
        slug="vanilla_reverse_topological_vs_vanilla_original_cross",
        baseline="vanilla_original",
        comparator="vanilla_reverse_topological",
        baseline_label="Vanilla Original",
        comparator_label="Vanilla Reverse Topological",
        title="Vanilla Original vs Vanilla Reverse Topological",
        group_slug="ordering_cross/vanilla",
        group_label="Cross ordering · Vanilla",
    ),
    "ordering_effects_vanilla_topological": ComparisonSpec(
        slug="vanilla_topological_vs_original",
        baseline="vanilla_original",
        comparator="vanilla_topological",
        baseline_label="Vanilla Original",
        comparator_label="Vanilla Topological",
        title="Vanilla Original vs Vanilla Topological",
        group_slug="ordering_effects/vanilla",
        group_label="Ordering effects · Vanilla",
    ),
    "ordering_effects_vanilla_reverse_topological": ComparisonSpec(
        slug="vanilla_reverse_topological_vs_original",
        baseline="vanilla_original",
        comparator="vanilla_reverse_topological",
        baseline_label="Vanilla Original",
        comparator_label="Vanilla Reverse Topological",
        title="Vanilla Original vs Vanilla Reverse Topological",
        group_slug="ordering_effects/vanilla",
        group_label="Ordering effects · Vanilla",
    ),
    "vanilla_ordering_combined": ComparisonSpec(
        slug="vanilla_original_vs_vanilla_topological",
        baseline="vanilla_original",
        comparator="vanilla_topological",
        baseline_label="Vanilla Original",
        comparator_label="Vanilla Topological",
        title="Vanilla Original vs Vanilla Topological",
        group_slug="ordering_effects/vanilla_combined",
        group_label="Ordering effects · Vanilla Combined",
    ),
}

DEFAULT_COMPARISON_KEYS: Tuple[str, ...] = tuple(COMPARISON_SPECS.keys())


def _dataset_slug_from_filename(path: Path) -> str:
    stem = path.stem
    if stem.startswith("result_"):
        stem = stem[len("result_") :]
    suffixes = (
        "_comparison_experiment_cleaned_reps_100",
        "_comparison_experiment_cleaned",
        "_comparison_experiment",
    )
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


EXPECTED_DATASET_SLUGS: Tuple[str, ...] = tuple(
    _dataset_slug_from_filename(Path(name)) for name in COMPARISON_RESULT_FILES
)

# Mapping from dataset names to acronyms for combined plots
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

# Default dataset acronym naming already loaded



def _draw_left_aligned_yticklabels(
    ax: Any,
    positions: Sequence[float],
    labels: Sequence[str],
    font_size: int,
    x_offset_points: float = -50.0,
) -> None:
    """Render y-axis tick labels with a shared left edge using point-based offset.

    We bypass Matplotlib's built-in tick labels (which right-align by default on
    the left spine) and use annotate with offset in points for reliable positioning
    with LaTeX fonts like Times. Points are absolute units independent of font metrics.
    """
    # Keep tick locations for grid/limits but hide the default labels
    ax.set_yticks(positions)
    ax.set_yticklabels([])

    # Use blended transform: x in axes fraction, y in data coordinates
    # Anchor point is on the y-axis spine (x=0 in axes coords)
    trans = ax.get_yaxis_transform()

    for y, label in zip(positions, labels):
        safe_label = label.replace("_", r"\_")
        ax.annotate(
            safe_label,
            xy=(0, y),  # anchor on y-axis spine
            xycoords=trans,
            xytext=(x_offset_points, 0),  # offset in points (negative = left)
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=font_size,
            annotation_clip=False,
        )


def _abbreviate_dataset_label_for_combined_plots(dataset_name: str, dataset_label: str) -> str:
    """Convert dataset label to acronym for combined plots to save horizontal space."""
    # Check if we have an acronym mapping for this dataset
    if dataset_name in DATASET_ACRONYMS:
        return DATASET_ACRONYMS[dataset_name]
    # Fallback to original label if no mapping exists
    return dataset_label


def _abbreviate_label_for_combined_plots(label: str) -> str:
    """Abbreviate labels for combined plots to save space and maintain consistency."""
    # Leave Vanilla fully spelled out for clarity/consistency across plots.
    # Abbreviate DAG
    label = label.replace("DAG Topological", "DAG Topo.")
    label = label.replace("DAG Original", "DAG Orig.")
    label = label.replace("DAG Reverse Topological", "DAG Rev. Topo.")
    # Abbreviate CPDAG
    label = label.replace("CPDAG Minimal", "CPDAG Min.")
    label = label.replace("CPDAG Minimal Topological", "CPDAG Min.")
    label = label.replace("CPDAG Minimal Reverse Topological", "CPDAG Min.")
    label = label.replace("CPDAG Discovered", "CPDAG Disc.")
    label = label.replace("CPDAG Discovered Topological", "CPDAG Disc.")
    label = label.replace("CPDAG Discovered Reverse Topological", "CPDAG Disc.")
    return label




def _ensure_condition_column(df: pd.DataFrame) -> pd.DataFrame:
    if "condition" in df.columns:
        return df

    df = df.copy()
    if "algorithm" in df.columns:
        algo = df["algorithm"].astype(str).str.strip()
    else:
        algo = pd.Series(["unknown"] * len(df), index=df.index, dtype="object")
    if "column_order" in df.columns:
        order = df["column_order"].astype(str).str.strip()
        order = order.replace({"nan": "unknown", "None": "unknown"})
        df["condition"] = (algo + "_" + order).str.replace("__+", "_", regex=True).str.strip("_")
    else:
        df["condition"] = algo
    return df


def _build_stat_test_config(metrics: Iterable[str]) -> StatTestConfig:
    metric_list: List[str] = []
    for name in list(DEFAULT_METRICS) + list(metrics):
        if name not in METRIC_CONFIG:
            continue
        if name not in metric_list:
            metric_list.append(name)
    return StatTestConfig(
        metrics=metric_list,
        metric_config=METRIC_CONFIG,
        condition_order=list(CONDITION_ORDER),
        condition_colors=CONDITION_COLORS,
        alpha=CI_ALPHA_DEFAULT,
    )


def _ensure_stat_tests_from_csvs(csv_files: Iterable[str], metrics: Iterable[str], force_regeneration: bool = True) -> None:
    metrics = tuple(metrics)
    config = _build_stat_test_config(metrics)
    for relative in csv_files:
        csv_path = SCRIPT_DIR / relative
        if not csv_path.exists():
            print(f"[WARN] Comparison CSV not found for forest plots: {relative}")
            continue

        dataset_slug = _dataset_slug_from_filename(csv_path)
        dataset_dir = COMPARISON_RESULTS_DIR / dataset_slug
        stat_tests_dir = dataset_dir / "stat_tests"
        posthoc_path = stat_tests_dir / "posthoc_wilcoxon_summary.csv"
        
        if force_regeneration:
            # Regenerate stat tests if CSV has been updated (e.g., algorithm names changed)
            # Remove old stat test files to force regeneration
            if posthoc_path.exists():
                posthoc_path.unlink()
            # Also remove friedman summary if it exists
            friedman_path = stat_tests_dir / "friedman_summary.csv"
            if friedman_path.exists():
                friedman_path.unlink()
        else:
            # Skip if statistical tests already exist
            if posthoc_path.exists():
                continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Unable to read '{csv_path.name}': {exc}")
            continue

        if df.empty:
            print(f"[INFO] CSV '{csv_path.name}' is empty; skipping stat tests.")
            continue

        df = _ensure_condition_column(df)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        try:
            run_stat_tests_for_dataset(
                dataset_df=df,
                dataset_slug=dataset_slug,
                output_dir=stat_tests_dir,
                config=config,
            )
            print(f"[INFO] Generated statistical tests for dataset '{dataset_slug}'.")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to compute stat tests for '{dataset_slug}': {exc}")




def _collect_forest_rows(metric: str, comparison: ComparisonSpec) -> List[ForestRow]:
    """
    Iterate through dataset directories, collect Hodges–Lehmann statistics for the
    specified baseline/comparator pair, and return normalized rows ready for plotting.
    """
    if metric not in METRIC_CONFIG:
        raise ValueError(f"Unknown metric '{metric}' (not in METRIC_CONFIG).")

    direction = METRIC_DIRECTIONS.get(metric, "lower")
    # Use frobenius_corr_norm slug for spearman metric to match original file names
    if metric == "correlation_matrix_difference":
        metric_slug = _get_cmd_config()["slug"]
    else:
        metric_slug = METRIC_CONFIG[metric]["slug"]

    rows: List[ForestRow] = []
    comparison_output_slug = comparison.output_slug

    allowed_datasets = set(EXPECTED_DATASET_SLUGS)

    if not COMPARISON_RESULTS_DIR.exists():
        COMPARISON_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    for dataset_dir in sorted(COMPARISON_RESULTS_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if allowed_datasets and dataset_dir.name not in allowed_datasets:
            continue

        dataset_name = dataset_dir.name
        posthoc_path = dataset_dir / "stat_tests" / "posthoc_wilcoxon_summary.csv"
        if not posthoc_path.exists():
            continue

        df = pd.read_csv(posthoc_path)
        if df.empty:
            continue

        mask = (
            (df["metric"] == metric)
            & (
                (
                    (df["condition_a"] == comparison.baseline)
                    & (df["condition_b"] == comparison.comparator)
                )
                | (
                    (df["condition_a"] == comparison.comparator)
                    & (df["condition_b"] == comparison.baseline)
                )
            )
        )

        subset = df.loc[mask].copy()
        if subset.empty:
            continue

        ci_lower_candidates = tuple(
            col for col in ("effect_ci_lower_holm", "effect_ci_lower") if col in subset.columns
        )
        ci_upper_candidates = tuple(
            col for col in ("effect_ci_upper_holm", "effect_ci_upper") if col in subset.columns
        )
        if not ci_lower_candidates or not ci_upper_candidates:
            print(
                f"[WARN] Missing CI columns for metric '{metric}' in dataset '{dataset_name}'. "
                "Skipping forest rows."
            )
            continue

        valid_mask = subset["effect_hl"].notna()
        valid_mask &= subset.loc[:, ci_lower_candidates].notna().any(axis=1)
        valid_mask &= subset.loc[:, ci_upper_candidates].notna().any(axis=1)
        subset = subset.loc[valid_mask]

        for _, rec in subset.iterrows():
            try:
                (
                    base_minus_comp,
                    ci_l,
                    ci_u,
                    ci_lower_source,
                    ci_upper_source,
                ) = _normalize_effect(
                    rec,
                    comparison,
                    ci_lower_candidates,
                    ci_upper_candidates,
                )
            except ValueError:
                continue

            effect_plot, plot_lower, plot_upper = _transform_effect_for_direction(
                base_minus_comp,
                ci_l,
                ci_u,
                direction,
            )

            train_size = rec.get("train_size")
            try:
                train_size = int(train_size)
            except (TypeError, ValueError):
                continue

            n_pairs_value = rec.get("n_pairs", 0)
            try:
                n_pairs_int = int(n_pairs_value)
            except (TypeError, ValueError):
                n_pairs_int = 0

            ci_lower_unc, ci_upper_unc = _extract_oriented_ci(
                rec,
                comparison,
                direction,
                "effect_ci_lower",
                "effect_ci_upper",
            )
            ci_lower_holm, ci_upper_holm = _extract_oriented_ci(
                rec,
                comparison,
                direction,
                "effect_ci_lower_holm",
                "effect_ci_upper_holm",
            )

            cond_a = str(rec["condition_a"])
            cond_b = str(rec["condition_b"])
            median_a = _maybe_float(rec.get("median_a"))
            median_b = _maybe_float(rec.get("median_b"))
            median_diff_raw = _maybe_float(rec.get("median_diff"))
            mean_diff_raw = _maybe_float(rec.get("mean_diff"))

            if cond_a == comparison.baseline and cond_b == comparison.comparator:
                median_baseline = median_a
                median_comparator = median_b
                median_diff_base_minus_comp = median_diff_raw
                mean_diff_base_minus_comp = mean_diff_raw
            elif cond_a == comparison.comparator and cond_b == comparison.baseline:
                median_baseline = median_b
                median_comparator = median_a
                median_diff_base_minus_comp = (
                    -median_diff_raw if median_diff_raw is not None else None
                )
                mean_diff_base_minus_comp = -mean_diff_raw if mean_diff_raw is not None else None
            else:
                median_baseline = None
                median_comparator = None
                median_diff_base_minus_comp = None
                mean_diff_base_minus_comp = None

            median_diff_oriented = (
                _orient_scalar(median_diff_base_minus_comp, direction)
                if median_diff_base_minus_comp is not None
                else None
            )
            mean_diff_oriented = (
                _orient_scalar(mean_diff_base_minus_comp, direction)
                if mean_diff_base_minus_comp is not None
                else None
            )

            rows.append(
                ForestRow(
                    dataset=dataset_name,
                    train_size=train_size,
                    effect=float(effect_plot),
                    ci_lower=float(plot_lower),
                    ci_upper=float(plot_upper),
                    n_pairs=n_pairs_int,
                    ci_source=(
                        "holm"
                        if ci_lower_source and "holm" in ci_lower_source
                        else ("uncorrected" if ci_lower_source else None)
                    ),
                    p_value=_maybe_float(rec.get("p_value")),
                    p_holm=_maybe_float(rec.get("p_value_holm")),
                    statistic=_maybe_float(rec.get("statistic")),
                    holm_alpha=_maybe_float(rec.get("holm_alpha")),
                    holm_stage_threshold=_maybe_float(rec.get("holm_stage_threshold")),
                    holm_significant=_maybe_bool(rec.get("holm_significant")),
                    holm_significant_stepdown=_maybe_bool(rec.get("holm_significant_stepdown")),
                    ci_lower_uncorrected=ci_lower_unc,
                    ci_upper_uncorrected=ci_upper_unc,
                    ci_level_uncorrected=_maybe_float(rec.get("effect_ci_level")),
                    ci_lower_holm=ci_lower_holm,
                    ci_upper_holm=ci_upper_holm,
                    ci_level_holm=_maybe_float(rec.get("effect_ci_level_holm")),
                    effect_baseline_minus_comparator=float(base_minus_comp),
                    median_baseline=median_baseline,
                    median_comparator=median_comparator,
                    median_diff_baseline_minus_comparator=median_diff_base_minus_comp,
                    median_diff_oriented=median_diff_oriented,
                    mean_diff_baseline_minus_comparator=mean_diff_base_minus_comp,
                    mean_diff_oriented=mean_diff_oriented,
                    direction=direction,
                    metric=metric,
                    metric_slug=metric_slug,
                    comparison_id=comparison.slug,
                    comparison_slug=comparison_output_slug,
                    comparison_title=comparison.title,
                    baseline_label=comparison.baseline_label,
                    comparator_label=comparison.comparator_label,
                    baseline_condition=comparison.baseline,
                    comparator_condition=comparison.comparator,
                    group_slug=comparison.group_slug,
                    group_label=comparison.group_label,
                )
            )

    return rows








def _reset_paper_root() -> None:
    if PAPER_ROOT.exists():
        shutil.rmtree(PAPER_ROOT)
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)










def plot_forest(
    metric: str,
    rows: List[ForestRow],
    comparison: ComparisonSpec,
    show_caption: bool = False,
    no_csv: bool = False,
) -> None:
    """Render and save the forest plot for a specific metric."""
    if not rows:
        print(
            f"[INFO] No valid data for forest plot of '{metric}' "
            f"({comparison.title})."
        )
        return

    metric_cfg = METRIC_CONFIG[metric]
    # Use correlation_matrix_difference slug and title for spearman metric to match original
    if metric == "correlation_matrix_difference":
        cmd_config = METRIC_CONFIG.get("correlation_matrix_difference", METRIC_CONFIG["frobenius_corr_norm"])
        metric_slug = cmd_config["slug"]
        metric_title = cmd_config["title"]
    else:
        metric_slug = metric_cfg["slug"]
        metric_title = metric_cfg["title"]
    # Normalize metric title for consistent presentation
    metric_title = _normalize_metric_title(metric, metric_title)
    direction = METRIC_DIRECTIONS.get(metric, "lower")

    df = build_forest_dataframe(rows)
    nnaa_summary: NnaaCheckSummary | None = None
    if metric == "nnaa":
        nnaa_summary = _validate_nnaa_metric(df)

    datasets = sorted(df["dataset"].unique())
    dataset_labels = {
        name: _abbreviate_dataset_label_for_combined_plots(
            name, df.loc[df["dataset"] == name, "dataset_label"].iloc[0]
        )
        for name in datasets
    }
    train_sizes = sorted(df["train_size"].unique())
    offsets = _build_offsets(train_sizes)
    marker_lookup = _build_marker_map(train_sizes)

    # Check if this comparison should be exported to paper folder
    key = (comparison.baseline, comparison.comparator)
    folder_name = PAPER_FOLDER_MAP.get(key)
    if folder_name is None:
        # Skip plots that are not in PAPER_FOLDER_MAP
        return

    # Fixed size for all single plots to keep dimensions consistent
    width, height = SINGLE_FIGSIZE

    comparison_slug = comparison.output_slug
    # Save directly to paper folder
    paper_dir = PAPER_ROOT / folder_name
    _ensure_dir(paper_dir)
    if no_csv:
        pdf_dir = paper_dir
        csv_dir = None
    else:
        pdf_dir = paper_dir / "pdf"
        csv_dir = paper_dir / "csv"
        _ensure_dir(pdf_dir)
        _ensure_dir(csv_dir)
    file_stem = f"forest_{comparison_slug}_{metric_slug}"
    pdf_path = pdf_dir / f"{file_stem}.pdf"
    csv_path = csv_dir / f"{file_stem}.csv" if csv_dir is not None else None

    caption_text = _build_caption_text(metric_title, comparison, direction) if show_caption else None

    def _callback(plt_mod: Any) -> None:
        setup_forest_plot_style()
        # Force font sizes via rcParams to override fastplot defaults
        plt_mod.rcParams['font.size'] = LABEL_FONT_SIZE
        plt_mod.rcParams['axes.labelsize'] = LABEL_FONT_SIZE
        plt_mod.rcParams['axes.titlesize'] = TITLE_FONT_SIZE
        plt_mod.rcParams['xtick.labelsize'] = TICK_FONT_SIZE_SINGLE
        plt_mod.rcParams['ytick.labelsize'] = TICK_FONT_SIZE_SINGLE
        plt_mod.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE
        plt_mod.rcParams['legend.title_fontsize'] = LEGEND_TITLE_FONT_SIZE

        fig = plt_mod.gcf()
        fig.clear()
        # Explicitly set figure size to ensure consistent dimensions across all plots
        fig.set_size_inches(width, height)
        ax = fig.add_subplot(111)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Custom color palette: Dark purple -> Blue -> Green -> Orange (replaces yellow) -> Red
        # Color-blind friendly palette that avoids yellow for better visibility
        # Based on viridis progression but with orange instead of yellow for Train 200
        custom_colors = ['#440154', '#31688e', '#35b779', '#f68f46', '#d73027']  # Dark purple, Blue, Green, Orange, Red
        color_lookup = {
            ts: custom_colors[i % len(custom_colors)]
            for i, ts in enumerate(train_sizes)
        }

        legend_entries = render_forest_panel(
            ax,
            df,
            datasets,
            dataset_labels,
            train_sizes,
            offsets,
            marker_lookup,
            color_lookup,
        )

        # Apply appropriate tick locator and tight limits based on data
        # This will calculate tight limits rounded to multiples of 0.1 for medium ranges
        apply_xaxis_tick_locator(ax, df=df)

        # Set tight y-axis limits based on actual data positions with spacing
        n_datasets = len(datasets)
        ax.set_ylim(0.5 * DATASET_SPACING, (n_datasets + 0.5) * DATASET_SPACING)

        # Explicitly set tick label sizes to ensure consistency with combined plots
        y_positions = [(len(datasets) - i) * DATASET_SPACING for i in range(len(datasets))]
        labels = [dataset_labels[name] for name in datasets]
        _draw_left_aligned_yticklabels(
            ax,
            y_positions,
            labels,
            font_size=TICK_FONT_SIZE_SINGLE,
            x_offset_points=Y_TICK_LABEL_OFFSET_SINGLE,
        )
        # Set tick parameters with explicit direction="out" for consistency
        ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE_SINGLE, direction="out")
        # Use reduced padding for x-axis to match Y label distance
        ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE_SINGLE, pad=TICK_PAD_X_SINGLE, direction="out")
        # Force ticks outward for consistency (standard matplotlib behavior) - final override
        ax.tick_params(axis="both", which="both", direction="out")
        
        ax.set_ylabel(
            "Dataset",
            fontsize=LABEL_FONT_SIZE,
            labelpad=X_LABEL_PAD + Y_LABEL_EXTRA_PAD,
        )

        if direction == "lower":
            xlabel = f"Hodges–Lehmann diff ({comparison.baseline_label} − {comparison.comparator_label})"
        else:
            xlabel = f"Hodges–Lehmann diff ({comparison.comparator_label} − {comparison.baseline_label})"

        # Use fixed labelpad to ensure consistent distance from tick labels across all plots
        ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE, labelpad=X_LABEL_PAD)
        title_metric = metric_title
        # Special handling for k-marginal slug: keep consistent format
        if metric_slug == "2marginal":
            title_metric = "k-Marginal Total Variation Distance"
        ax.set_title(f"{comparison.title} · {title_metric}", fontsize=TITLE_FONT_SIZE, pad=AXES_TITLE_PAD)

        handles = [legend_entries[ts] for ts in train_sizes if ts in legend_entries]
        labels = [handle.get_label() for handle in handles]

        # Legend below x-axis label, using axes coordinates for reliable positioning
        if handles:
            ax.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),  # Below subplot in axes coords
                ncol=len(handles),
                frameon=False,
                fontsize=LEGEND_FONT_SIZE,
                title="Train Size",
                title_fontsize=LEGEND_TITLE_FONT_SIZE,
                columnspacing=LEGEND_COLUMNSPACING_SINGLE,
                handletextpad=LEGEND_HANDLETEXTPAD,
            )

        # FIXED margins for ALL single plots - ensures uniform dimensions
        fig.subplots_adjust(
            left=SINGLE_LEFT_MARGIN,
            right=SINGLE_RIGHT_MARGIN,
            bottom=SINGLE_BOTTOM_MARGIN,
            top=SINGLE_TOP_MARGIN,
        )

        # Disable tight_layout to preserve fixed margins
        original_tight_layout = plt_mod.tight_layout
        def _no_tight_layout(*args: Any, **kwargs: Any) -> None:
            fig.subplots_adjust(
                left=SINGLE_LEFT_MARGIN,
                right=SINGLE_RIGHT_MARGIN,
                bottom=SINGLE_BOTTOM_MARGIN,
                top=SINGLE_TOP_MARGIN,
            )
            plt_mod.tight_layout = original_tight_layout
        plt_mod.tight_layout = _no_tight_layout

    # Use standard bbox with fixed padding to ensure consistent dimensions
    # Increased padding to include legend below the subplot area
    FIXED_PAD_INCHES = 0.5
    save_with_fastplot(_callback, pdf_path, (width, height), use_tight_bbox=False, pad_inches=FIXED_PAD_INCHES)

    if csv_path is not None:
        # Prepare DataFrame for CSV
        df_csv = df.copy()
        
        # Add is_significant column based on p_holm/stepdown AND visual fix (effect >= 1e-9)
        def _compute_significance(row):
            p_holm = row.get("p_holm")
            stepdown = row.get("holm_significant_stepdown")
            effect = row.get("effect")
            
            is_stat_sig = (
                (p_holm is not None and not pd.isna(p_holm) and p_holm < 0.05)
                or bool(stepdown)
            )
            # Apply visual fix: zero effect is never significant
            if abs(effect) < 1e-9:
                return False
            return is_stat_sig

        df_csv["is_significant"] = df_csv.apply(_compute_significance, axis=1)

        # Rename metric if needed
        if metric == "correlation_matrix_difference":
             df_csv["metric"] = "correlation_matrix_difference"
        elif "metric" not in df_csv.columns:
             df_csv["metric"] = metric

        essential_columns = [
            "dataset",
            "train_size",
            "effect",
            "ci_lower",
            "ci_upper",
            "p_value",
            "p_holm",
            "is_significant",
            "metric",
        ]
        df_csv = df_csv[[col for col in essential_columns if col in df_csv.columns]]
        df_csv.to_csv(csv_path, index=False)

    print(
        f"[SUCCESS] Saved forest plot for '{metric}' ({comparison.title}) in {pdf_path}"
    )

    return nnaa_summary


def plot_combined_forest(
    metrics: Sequence[str],
    rows_lookup: Dict[str, List[ForestRow]],
    comparison: ComparisonSpec,
    show_caption: bool = False,
    no_csv: bool = False,
) -> None:
    """Generate a multi-panel forest plot across multiple metrics."""
    ordered_metrics = [metric for metric in metrics if metric in METRIC_CONFIG]
    if not ordered_metrics:
        print("[WARN] No known metrics provided for combined forest plot.")
        return

    df_by_metric: Dict[str, pd.DataFrame] = {}
    for metric in ordered_metrics:
        rows = rows_lookup.get(metric, [])
        if not rows:
            print(
                f"[INFO] Skipping combined plot: no rows for metric '{metric}' in {comparison.title}."
            )
            return
        df = build_forest_dataframe(rows)
        if df.empty:
            print(
                f"[INFO] Skipping combined plot: empty dataframe for metric '{metric}' in {comparison.title}."
            )
            return
        df_by_metric[metric] = df

    datasets = sorted({name for df in df_by_metric.values() for name in df["dataset"].unique()})
    if not datasets:
        print(
            f"[INFO] Skipping combined plot: no datasets available for {comparison.title}."
        )
        return

    dataset_labels: Dict[str, str] = {}
    for df in df_by_metric.values():
        dataset_labels.update(
            {
                name: df.loc[df["dataset"] == name, "dataset_label"].iloc[0]
                for name in df["dataset"].unique()
            }
        )
    dataset_labels = {
        name: _abbreviate_dataset_label_for_combined_plots(
            name, dataset_labels.get(name, _format_display_label(name))
        )
        for name in datasets
    }

    train_sizes = sorted(
        {
            int(ts)
            for df in df_by_metric.values()
            for ts in df["train_size"].unique()
            if not pd.isna(ts)
        }
    )
    if not train_sizes:
        print(
            f"[INFO] Skipping combined plot: no train sizes detected for {comparison.title}."
        )
        return

    offsets = _build_offsets(train_sizes)
    marker_lookup = _build_marker_map(train_sizes)

    # Check if this comparison should be exported to paper folder
    key = (comparison.baseline, comparison.comparator)
    folder_name = PAPER_FOLDER_MAP.get(key)
    if folder_name is None:
        # Skip plots that are not in PAPER_FOLDER_MAP
        return

    # Use fixed duo layout for all combined plots
    height = COMBINED_FIGSIZE_DUO[1]
    total_width = COMBINED_FIGSIZE_DUO[0]
    panel_width = total_width / max(1, len(ordered_metrics))

    comparison_slug = comparison.output_slug
    # Use correlation_matrix_difference slug for spearman metric to match original file names
    cmd_config = _get_cmd_config()
    metrics_slug = "_".join(
        cmd_config["slug"] if m == "correlation_matrix_difference" else METRIC_CONFIG[m]["slug"]
        for m in ordered_metrics
    )
    # Save directly to paper folder
    paper_dir = PAPER_ROOT / folder_name
    _ensure_dir(paper_dir)
    if no_csv:
        pdf_dir = paper_dir
        csv_dir = None
    else:
        pdf_dir = paper_dir / "pdf"
        csv_dir = paper_dir / "csv"
        _ensure_dir(pdf_dir)
        _ensure_dir(csv_dir)
    file_stem = f"forest_combined_{comparison_slug}_{metrics_slug}"
    pdf_path = pdf_dir / f"{file_stem}.pdf"
    csv_path = csv_dir / f"{file_stem}.csv" if csv_dir is not None else None

    caption_text_combined: str | None = None
    if show_caption:
        # Normalize metric titles for caption
        normalized_titles = [
            _normalize_metric_title(m, METRIC_CONFIG[m]["title"]) for m in ordered_metrics
        ]
        caption_text_combined = _build_caption_text(
            ", ".join(normalized_titles),
            comparison,
            METRIC_DIRECTIONS.get(ordered_metrics[0], "lower"),
        )

    def _callback(plt_mod: Any) -> None:
        # Use constants from shared config - no hardcoded values!
        setup_forest_plot_style()
        plt_mod.rcParams['font.size'] = LABEL_FONT_SIZE_COMBINED
        plt_mod.rcParams['axes.labelsize'] = LABEL_FONT_SIZE_COMBINED
        plt_mod.rcParams['axes.titlesize'] = TITLE_FONT_SIZE_SUBPLOT_COMBINED
        plt_mod.rcParams['xtick.labelsize'] = TICK_FONT_SIZE_COMBINED
        plt_mod.rcParams['ytick.labelsize'] = TICK_FONT_SIZE_COMBINED
        plt_mod.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE_COMBINED
        plt_mod.rcParams['legend.title_fontsize'] = LEGEND_TITLE_FONT_SIZE_COMBINED
        plt_mod.rcParams['figure.titlesize'] = TITLE_FONT_SIZE_SUPTITLE_COMBINED

        fig = plt_mod.gcf()
        fig.clear()
        fig.set_size_inches(total_width, height)
        axes = fig.subplots(1, len(ordered_metrics), sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        # Custom color palette: Dark purple -> Blue -> Green -> Orange (replaces yellow) -> Red
        # Color-blind friendly palette that avoids yellow for better visibility
        # Based on viridis progression but with orange instead of yellow for Train 200
        custom_colors = ['#440154', '#31688e', '#35b779', '#f68f46', '#d73027']  # Dark purple, Blue, Green, Orange, Red
        color_lookup = {
            ts: custom_colors[i % len(custom_colors)]
            for i, ts in enumerate(train_sizes)
        }

        combined_legend_entries: Dict[int, Any] = {}

        # Tick font overrides for combined plots
        tick_override = TICK_FONT_SIZE_COMBINED

        for idx, metric in enumerate(ordered_metrics):
            ax = axes[idx]

            # Remove top and right spines for each panel
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            df = df_by_metric[metric]
            legend_entries = render_forest_panel(
                ax,
                df,
                datasets,
                dataset_labels,
                train_sizes,
                offsets,
                marker_lookup,
                color_lookup=color_lookup,
                is_combined=True,
                tick_font_size_override=tick_override,
            )
            combined_legend_entries.update(legend_entries)

            # Apply appropriate tick locator and tight limits based on data
            # This will calculate tight limits rounded to multiples of 0.1 for medium ranges
            apply_xaxis_tick_locator(ax, df=df)

            # Set tight y-axis limits based on actual data positions with spacing
            # Datasets are positioned from DATASET_SPACING to n_datasets*DATASET_SPACING
            n_datasets = len(datasets)
            # Set limits to exactly fit the data without extra white space
            ax.set_ylim(0.5 * DATASET_SPACING, (n_datasets + 0.5) * DATASET_SPACING)

            # Explicitly set tick label sizes to ensure consistency across all plots
            # This ensures uniform font size regardless of number of datasets
            y_positions = [(len(datasets) - i) * DATASET_SPACING for i in range(len(datasets))]
            if idx == 0:
                labels = [dataset_labels[name] for name in datasets]
                _draw_left_aligned_yticklabels(
                    ax,
                    y_positions,
                    labels,
                    font_size=tick_override,
                    x_offset_points=Y_TICK_LABEL_OFFSET_COMBINED,
                )
                ax.tick_params(axis="y", labelsize=tick_override)
            else:
                # Hide default y-axis tick labels for other subplots
                ax.set_yticks(y_positions)
                ax.set_yticklabels([])
                ax.tick_params(axis="y", labelleft=False)
                # Ensure labels are hidden even if render_forest_panel set them
                for label in ax.get_yticklabels():
                    label.set_visible(False)
            # Use reduced padding for x-axis to match Y label distance (proportional to single plots)
            ax.tick_params(axis="x", labelsize=tick_override, pad=TICK_PAD_X_COMBINED)
            
            # Force ticks outward for consistency - applied to both axes, all directions
            ax.tick_params(axis="both", which="both", direction="out", left=True, right=False, bottom=True, top=False)

            metric_cfg = METRIC_CONFIG[metric]
            # Use correlation_matrix_difference title for spearman metric to match original
            if metric == "correlation_matrix_difference":
                cmd_config = _get_cmd_config()
                panel_subtitle = cmd_config["title"]
                # Normalize metric title
                panel_subtitle = _normalize_metric_title("correlation_matrix_difference", panel_subtitle)
                # Check if the original correlation_matrix_difference slug is 2marginal
                if cmd_config.get("slug") == "2marginal":
                    panel_subtitle = "k-Marginal Total Variation Distance"
            else:
                panel_subtitle = metric_cfg["title"]
                # Normalize metric title
                panel_subtitle = _normalize_metric_title(metric, panel_subtitle)
                if metric_cfg.get("slug") == "2marginal":
                    panel_subtitle = "k-Marginal Total Variation Distance"

            ax.set_title(
                panel_subtitle,
                fontsize=TITLE_FONT_SIZE_SUBPLOT_COMBINED,
                pad=COMBINED_TITLE_PAD,
                fontweight='normal',
            )

            if idx == 0:
                ax.set_ylabel(
                    "Dataset",
                    fontsize=LABEL_FONT_SIZE_COMBINED,
                    labelpad=X_LABEL_PAD + Y_LABEL_EXTRA_PAD_COMBINED,
                )
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
                for label in ax.get_yticklabels():
                    label.set_visible(False)

        # Shared x-axis label (will be placed after subplots_adjust)
        direction = METRIC_DIRECTIONS.get(ordered_metrics[0], "lower")
        if direction == "lower":
            shared_xlabel = f"Hodges–Lehmann diff ({comparison.baseline_label} − {comparison.comparator_label})"
        else:
            shared_xlabel = f"Hodges–Lehmann diff ({comparison.comparator_label} − {comparison.baseline_label})"

        handles = [combined_legend_entries[ts] for ts in train_sizes if ts in combined_legend_entries]
        labels = [handle.get_label() for handle in handles]

        # Calculate bottom margin based on whether we have legend
        legend_y = COMBINED_LEGEND_Y  # Default position even without legend
        if handles:
            bottom_margin = COMBINED_BOTTOM_MARGIN  # Accommodate shared x-label and legend
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, legend_y),
                bbox_transform=fig.transFigure,
                ncol=len(handles),  # All train sizes in one row
                frameon=False,
                fontsize=LEGEND_FONT_SIZE_COMBINED,
                title="Train Size",
                title_fontsize=LEGEND_TITLE_FONT_SIZE_COMBINED,
                columnspacing=LEGEND_COLUMNSPACING_SINGLE,
                handletextpad=LEGEND_HANDLETEXTPAD,
            )
        else:
            bottom_margin = COMBINED_BOTTOM_MARGIN_NO_LEGEND
        caption_margin = 0.0
        caption_y = COMBINED_CAPTION_Y  # Higher position for caption to avoid legend/xlabel
        if caption_text_combined:
            fig.text(
                0.5,
                caption_y,
                caption_text_combined,
                ha="center",
                va="bottom",
                fontsize=CAPTION_FONT_SIZE_ENHANCED,
            )
            bottom_margin = max(bottom_margin, caption_y + 0.03)
            caption_margin = 0.02

        # Maximize horizontal space; use consistent top margin from config
        actual_bottom = bottom_margin + caption_margin
        fig.subplots_adjust(
            left=0.20,
            right=0.98,
            top=COMBINED_TOP_MARGIN,
            bottom=actual_bottom,
            wspace=COMBINED_WSPACE_DUO,
        )

        # Align x-axis tick labels vertically across all panels
        fig.align_xlabels(axes)
        fig.align_ylabels(axes)

        # Place shared x-axis label AFTER subplots_adjust
        # Position it between the tick labels and the legend using config value
        fig.text(
            0.5,
            COMBINED_SHARED_XLABEL_Y,
            shared_xlabel,
            ha='center',
            va='center',
            fontsize=LABEL_FONT_SIZE_COMBINED,
            fontweight='normal',
            transform=fig.transFigure,
        )

        # Main title at top (comparison VS description) - ADD AFTER subplots_adjust
        fig.text(
            0.5, COMBINED_SUPTITLE_Y,
            comparison.title,
            ha='center',
            va='top',
            fontsize=TITLE_FONT_SIZE_SUPTITLE_COMBINED,
            fontweight='normal',
            transform=fig.transFigure,
        )

        original_tight_layout = plt_mod.tight_layout

        def _no_tight_layout(*args: Any, **kwargs: Any) -> None:
            fig.subplots_adjust(
                left=0.20,
                right=0.98,
                top=COMBINED_TOP_MARGIN,
                bottom=bottom_margin + caption_margin,
                wspace=COMBINED_WSPACE_DUO,
            )
            plt_mod.tight_layout = original_tight_layout

        plt_mod.tight_layout = _no_tight_layout

    figsize = (total_width, height)
    # Use fixed padding to keep legend/caption inside the bbox uniformly
    pad_inches = 0.20
    save_with_fastplot(_callback, pdf_path, figsize, pad_inches=pad_inches)

    # Save combined CSV with only essential columns
    if csv_path is not None:
        essential_columns = ["dataset", "train_size", "effect", "ci_lower", "ci_upper", "p_value", "p_holm", "metric", "holm_significant_stepdown"]
        combined_rows: List[Dict[str, Any]] = []
        for metric in ordered_metrics:
            rows = rows_lookup.get(metric, [])
            for row in rows:
                row_dict = dict(row.__dict__)
                row_dict["metric"] = metric
                # Keep only essential columns
                filtered_dict = {k: v for k, v in row_dict.items() if k in essential_columns}
                combined_rows.append(filtered_dict)

        if combined_rows:
            combined_df = pd.DataFrame(combined_rows)
            
            # Add is_significant column
            def _compute_significance_combined(row):
                p_holm = row.get("p_holm")
                effect = row.get("effect", 0.0)
                # We prioritize p_holm for combined plots logic if stepdown not present
                stepdown = row.get("holm_significant_stepdown")
                
                is_stat_sig = (
                     (p_holm is not None and not pd.isna(p_holm) and p_holm < 0.05)
                     or bool(stepdown)
                )

                if abs(effect) < 1e-9:
                    return False
                return is_stat_sig

            combined_df["is_significant"] = combined_df.apply(_compute_significance_combined, axis=1)

            # Rename metric
            combined_df["metric"] = combined_df["metric"].replace({
                "correlation_matrix_difference": "correlation_matrix_difference"
            })

            if "holm_significant_stepdown" in combined_df.columns:
                combined_df = combined_df.drop(columns=["holm_significant_stepdown"])

            combined_df.to_csv(csv_path, index=False)

    print(
        "[SUCCESS] Saved combined forest plot for metrics "
        f"{', '.join(ordered_metrics)} ({comparison.title}) in {pdf_path}"
    )


def plot_vanilla_ordering_combined_forest(
    metric: str,
    show_caption: bool = False,
    no_csv: bool = False,
) -> None:
    """Generate a combined forest plot showing Vanilla ordering effects (Topological and Worst vs Original)."""
    if metric not in METRIC_CONFIG:
        print(f"[WARN] Unknown metric '{metric}' for vanilla ordering combined forest plot.")
        return

    # Get data for the two comparisons directly
    original_vs_topological = _collect_forest_rows(metric, COMPARISON_SPECS["ordering_effects_vanilla_topological"])
    original_vs_worst = _collect_forest_rows(metric, COMPARISON_SPECS["ordering_effects_vanilla_reverse_topological"])

    if not original_vs_topological and not original_vs_worst:
        print("[INFO] No data available for vanilla ordering combined plot.")
        return

    # Build dataframes for each comparison
    df_orig_top = build_forest_dataframe(original_vs_topological) if original_vs_topological else pd.DataFrame()
    df_orig_worst = build_forest_dataframe(original_vs_worst) if original_vs_worst else pd.DataFrame()

    # Get all datasets from all comparisons
    all_datasets = set()
    for df in [df_orig_top, df_orig_worst]:
        if not df.empty:
            all_datasets.update(df["dataset"].unique())

    datasets = sorted(all_datasets)
    if not datasets:
        print("[INFO] No datasets available for vanilla ordering combined plot.")
        return

    # Build dataset labels from dataframes
    dataset_labels: Dict[str, str] = {}
    for df in [df_orig_top, df_orig_worst]:
        if not df.empty:
            dataset_labels.update(
                {
                    name: df.loc[df["dataset"] == name, "dataset_label"].iloc[0]
                    for name in df["dataset"].unique()
                }
            )
    dataset_labels = {
        name: _abbreviate_dataset_label_for_combined_plots(
            name, dataset_labels.get(name, _format_display_label(name))
        )
        for name in datasets
    }

    # Get train sizes
    train_sizes = sorted({
        int(ts)
        for df in [df_orig_top, df_orig_worst]
        for ts in df["train_size"].unique()
        if not pd.isna(ts)
    })
    if not train_sizes:
        print("[INFO] No train sizes detected for vanilla ordering combined plot.")
        return

    offsets = _build_offsets(train_sizes)
    marker_lookup = _build_marker_map(train_sizes)

    # Fixed dimensions for 2-panel combined plots
    height = COMBINED_FIGSIZE_DUO[1]
    panel_width = COMBINED_FIGSIZE_DUO[0] / 2

    # Prepare output paths - save directly to paper folder
    comparison = COMPARISON_SPECS[COMBINED_VANILLA_ORDERING_KEY]
    # Use correlation_matrix_difference slug for spearman metric to match original file names
    if metric == "correlation_matrix_difference":
        metric_slug = _get_cmd_config()["slug"]
    else:
        metric_slug = METRIC_CONFIG[metric]["slug"]
    paper_dir = PAPER_ROOT / "vanilla_combined_topological_reverse_topological"
    _ensure_dir(paper_dir)
    if no_csv:
        pdf_dir = paper_dir
        csv_dir = None
    else:
        pdf_dir = paper_dir / "pdf"
        csv_dir = paper_dir / "csv"
        _ensure_dir(pdf_dir)
        _ensure_dir(csv_dir)

    file_stem = f"forest_combined_vanilla_ordering_topological_and_reverse_{metric_slug}"
    pdf_path = pdf_dir / f"{file_stem}.pdf"
    csv_path = csv_dir / f"{file_stem}.csv" if csv_dir is not None else None

    def _callback(plt_mod: Any) -> None:
        setup_forest_plot_style()
        # EXACT COPY from plot_combined_forest() - font sizes
        plt_mod.rcParams['font.size'] = LABEL_FONT_SIZE_COMBINED
        plt_mod.rcParams['axes.labelsize'] = LABEL_FONT_SIZE_COMBINED
        plt_mod.rcParams['axes.titlesize'] = TITLE_FONT_SIZE_SUBPLOT_COMBINED
        plt_mod.rcParams['xtick.labelsize'] = TICK_FONT_SIZE_COMBINED
        plt_mod.rcParams['ytick.labelsize'] = TICK_FONT_SIZE_COMBINED
        plt_mod.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE_COMBINED
        plt_mod.rcParams['legend.title_fontsize'] = LEGEND_TITLE_FONT_SIZE_COMBINED
        plt_mod.rcParams['figure.titlesize'] = TITLE_FONT_SIZE_SUPTITLE_COMBINED

        fig = plt_mod.gcf()
        fig.clear()

        # 2 panels instead of N metrics
        axes = fig.subplots(1, 2, sharey=True)

        # Custom color palette: Dark purple -> Blue -> Green -> Orange (replaces yellow) -> Red
        # Color-blind friendly palette that avoids yellow for better visibility
        # Based on viridis progression but with orange instead of yellow for Train 200
        custom_colors = ['#440154', '#31688e', '#35b779', '#f68f46', '#d73027']  # Dark purple, Blue, Green, Orange, Red
        color_lookup = {
            ts: custom_colors[i % len(custom_colors)]
            for i, ts in enumerate(train_sizes)
        }

        combined_legend_entries: Dict[int, Any] = {}

        # Subplot 1: Original vs Topological
        ax1 = axes[0]
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        tick_override = TICK_FONT_SIZE_COMBINED  # closer to interventional sizing

        if original_vs_topological:
            legend_entries_top = render_forest_panel(
                ax1,
                df_orig_top,
                datasets,
                dataset_labels,
                train_sizes,
                offsets,
                marker_lookup,
                color_lookup,
                is_combined=True,
                tick_font_size_override=tick_override,
            )
            combined_legend_entries.update(legend_entries_top)

            # Apply appropriate tick locator and tight limits based on data
            apply_xaxis_tick_locator(ax1, df=df_orig_top)

            n_datasets = len(datasets)
            ax1.set_ylim(0.5 * DATASET_SPACING, (n_datasets + 0.5) * DATASET_SPACING)

            # Explicitly set tick label sizes to ensure consistency (matching interventional_experiment)
            y_positions = [(len(datasets) - idx) * DATASET_SPACING for idx in range(len(datasets))]
            labels = [dataset_labels[name] for name in datasets]
            _draw_left_aligned_yticklabels(
                ax1,
                y_positions,
                labels,
                font_size=TICK_FONT_SIZE_COMBINED,
                x_offset_points=Y_TICK_LABEL_OFFSET_COMBINED,
            )
            ax1.tick_params(axis="y", labelsize=TICK_FONT_SIZE_COMBINED)
            ax1.tick_params(axis="x", labelsize=TICK_FONT_SIZE_COMBINED)
            # Force ticks outward for consistency - applied to both axes, all directions
            ax1.tick_params(axis="both", which="both", direction="out", left=True, right=False, bottom=True, top=False)

            comparison_topological = COMPARISON_SPECS["ordering_effects_vanilla_topological"]
            baseline_label = comparison_topological.baseline_label
            comparator_label = comparison_topological.comparator_label

            ax1.set_title(
                f"{baseline_label} vs {comparator_label}",
                fontsize=TITLE_FONT_SIZE_SUBPLOT_COMBINED,
                pad=COMBINED_TITLE_PAD,
                fontweight='normal',
            )
            ax1.set_ylabel(
                "Dataset",
                fontsize=LABEL_FONT_SIZE_COMBINED,
                labelpad=X_LABEL_PAD + Y_LABEL_EXTRA_PAD_COMBINED,
            )

        # Subplot 2: Original vs Worst
        ax2 = axes[1]
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        if original_vs_worst:
            legend_entries_worst = render_forest_panel(
                ax2,
                df_orig_worst,
                datasets,
                dataset_labels,
                train_sizes,
                offsets,
                marker_lookup,
                color_lookup,
                is_combined=True,
                tick_font_size_override=tick_override,
            )
            combined_legend_entries.update(legend_entries_worst)

            # Apply appropriate tick locator and tight limits based on data
            apply_xaxis_tick_locator(ax2, df=df_orig_worst)

            n_datasets = len(datasets)
            ax2.set_ylim(0.5 * DATASET_SPACING, (n_datasets + 0.5) * DATASET_SPACING)

            # Hide default y-axis tick labels for right subplot
            ax2.set_yticks([(len(datasets) - i) * DATASET_SPACING for i in range(len(datasets))])
            ax2.set_yticklabels([])
            ax2.tick_params(axis="x", labelsize=TICK_FONT_SIZE_COMBINED, pad=TICK_PAD_X_COMBINED)
            ax2.tick_params(axis="y", labelleft=False)
            # Force ticks outward for consistency - applied to both axes, all directions
            ax2.tick_params(axis="both", which="both", direction="out", left=True, right=False, bottom=True, top=False)
            # Ensure labels are hidden even if render_forest_panel set them
            for label in ax2.get_yticklabels():
                label.set_visible(False)

            # Set title for right subplot (was missing!)
            comparison_worst = COMPARISON_SPECS["ordering_effects_vanilla_reverse_topological"]
            baseline_label_worst = comparison_worst.baseline_label
            comparator_label_worst = comparison_worst.comparator_label

            ax2.set_title(
                f"{baseline_label_worst} vs {comparator_label_worst}",
                fontsize=TITLE_FONT_SIZE_SUBPLOT_COMBINED,
                pad=COMBINED_TITLE_PAD,
                fontweight='normal',
            )
            ax2.set_ylabel("")  # No y-label on right subplot

        # Main title at top (metric name becomes main title for vanilla ordering)
        # Use correlation_matrix_difference title for spearman metric to match original
        if metric == "correlation_matrix_difference":
            cmd_config = _get_cmd_config()
            metric_title = cmd_config["title"]
            metric_title = _normalize_metric_title("correlation_matrix_difference", metric_title)
            if cmd_config.get("slug") == "2marginal":
                metric_title = "k-Marginal Total Variation Distance"
        else:
            metric_title = METRIC_CONFIG[metric]['title']
            metric_title = _normalize_metric_title(metric, metric_title)
            if METRIC_CONFIG[metric].get("slug") == "2marginal":
                metric_title = "k-Marginal Total Variation Distance"

        fig.text(
            0.5, COMBINED_SUPTITLE_Y,
            metric_title,
            ha='center',
            va='top',
            fontsize=TITLE_FONT_SIZE_SUPTITLE_COMBINED,
            fontweight='normal',
        )

        # Shared x-axis label (same orientation for both subplots)
        shared_direction = METRIC_DIRECTIONS.get(metric, "lower")
        shared_baseline = COMPARISON_SPECS["ordering_effects_vanilla_topological"].baseline_label
        if shared_direction == "lower":
            shared_xlabel = f"Hodges–Lehmann diff ({shared_baseline} − comparator)"
        else:
            shared_xlabel = f"Hodges–Lehmann diff (comparator − {shared_baseline})"

        fig.text(
            0.5,
            COMBINED_SHARED_XLABEL_Y,
            shared_xlabel,
            ha="center",
            va="center",
            fontsize=LABEL_FONT_SIZE_COMBINED,
            fontweight='normal',
            transform=fig.transFigure,
        )

        handles = [combined_legend_entries[ts] for ts in train_sizes if ts in combined_legend_entries]
        labels = [handle.get_label() for handle in handles]

        # Legend positioning - closer to axes to reduce wasted space
        if handles:
            bottom_margin = COMBINED_BOTTOM_MARGIN
            legend_y = COMBINED_LEGEND_Y
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, legend_y),
                bbox_transform=fig.transFigure,
                ncol=len(handles),
                frameon=False,
                fontsize=LEGEND_FONT_SIZE_COMBINED,
                title="Train Size",
                title_fontsize=LEGEND_TITLE_FONT_SIZE_COMBINED,
                columnspacing=LEGEND_COLUMNSPACING_COMBINED,
                handletextpad=LEGEND_HANDLETEXTPAD,
            )
        else:
            bottom_margin = COMBINED_BOTTOM_MARGIN_NO_LEGEND

        # Compact layout with reduced bottom whitespace (shared xlabel handles spacing)
        fig.subplots_adjust(
            left=0.20,
            right=0.96,
            top=COMBINED_TOP_MARGIN,
            bottom=bottom_margin,
            wspace=COMBINED_WSPACE_DUO,  # Increased spacing to prevent x-axis label overlap
        )

        # Align x-axis tick labels vertically across all panels
        fig.align_xlabels(axes)
        fig.align_ylabels(axes)

        original_tight_layout = plt_mod.tight_layout

        def _no_tight_layout(*args: Any, **kwargs: Any) -> None:
            fig.subplots_adjust(
                left=0.20,
                right=0.96,
                top=COMBINED_TOP_MARGIN,
                bottom=bottom_margin,
                wspace=COMBINED_WSPACE_DUO,
            )
            plt_mod.tight_layout = original_tight_layout

        plt_mod.tight_layout = _no_tight_layout

    figsize = (panel_width * 2, height)
    save_with_fastplot(_callback, pdf_path, figsize, pad_inches=0.20)

    # Save combined CSV with only essential columns
    if csv_path is not None:
        essential_columns = ["dataset", "train_size", "effect", "ci_lower", "ci_upper", "p_value", "p_holm", "comparison", "metric"]
        combined_rows = []
        for data, comparison_name in [(original_vs_topological, "original_vs_topological"),
                                     (original_vs_worst, "original_vs_worst")]:
            if data:
                for row in data:
                    row_dict = dict(row.__dict__)
                    row_dict["comparison"] = comparison_name
                    row_dict["metric"] = metric
                    # Keep only essential columns
                    filtered_dict = {k: v for k, v in row_dict.items() if k in essential_columns}
                    combined_rows.append(filtered_dict)

        if combined_rows:
            combined_df = pd.DataFrame(combined_rows)

            # Add is_significant column
            def _compute_significance_combined(row):
                p_holm = row.get("p_holm")
                effect = row.get("effect", 0.0)
                stepdown = row.get("holm_significant_stepdown")
                
                is_stat_sig = (
                     (p_holm is not None and not pd.isna(p_holm) and p_holm < 0.05)
                     or bool(stepdown)
                )

                if abs(effect) < 1e-9:
                    return False
                return is_stat_sig

            combined_df["is_significant"] = combined_df.apply(_compute_significance_combined, axis=1)

            # Rename metric
            combined_df["metric"] = combined_df["metric"].replace({
                "correlation_matrix_difference": "correlation_matrix_difference"
            })

            if "holm_significant_stepdown" in combined_df.columns:
                combined_df = combined_df.drop(columns=["holm_significant_stepdown"])

            combined_df.to_csv(csv_path, index=False)

    print(
        f"[SUCCESS] Saved vanilla ordering combined forest plot for metric "
        f"{metric} in {pdf_path}"
    )


def plot_cpdag_combined_forest(
    metric: str,
    show_caption: bool = False,
    no_csv: bool = False,
) -> None:
    """Generate a combined forest plot showing CPDAG effects (Minimal vs Discovered) for a single metric."""
    if metric not in METRIC_CONFIG:
        print(f"[WARN] Unknown metric '{metric}' for CPDAG combined forest plot.")
        return

    # Get data for the two comparisons directly
    vanilla_vs_minimal = _collect_forest_rows(metric, COMPARISON_SPECS["original_cpdag_minimal_vs_vanilla"])
    vanilla_vs_discovered = _collect_forest_rows(metric, COMPARISON_SPECS["original_cpdag_discovered_vs_vanilla"])

    if not vanilla_vs_minimal and not vanilla_vs_discovered:
        print("[INFO] No data available for CPDAG combined plot.")
        return

    # Build dataframes for each comparison
    df_minimal = build_forest_dataframe(vanilla_vs_minimal) if vanilla_vs_minimal else pd.DataFrame()
    df_discovered = build_forest_dataframe(vanilla_vs_discovered) if vanilla_vs_discovered else pd.DataFrame()

    # Get all datasets from all comparisons
    all_datasets = set()
    for df in [df_minimal, df_discovered]:
        if not df.empty:
            all_datasets.update(df["dataset"].unique())

    datasets = sorted(all_datasets)
    if not datasets:
        print("[INFO] No datasets available for CPDAG combined plot.")
        return

    # Build dataset labels from dataframes
    dataset_labels: Dict[str, str] = {}
    for df in [df_minimal, df_discovered]:
        if not df.empty:
            dataset_labels.update(
                {
                    name: df.loc[df["dataset"] == name, "dataset_label"].iloc[0]
                    for name in df["dataset"].unique()
                }
            )
    dataset_labels = {
        name: _abbreviate_dataset_label_for_combined_plots(
            name, dataset_labels.get(name, _format_display_label(name))
        )
        for name in datasets
    }

    # Get train sizes
    train_sizes = sorted({
        int(ts)
        for df in [df_minimal, df_discovered]
        for ts in df["train_size"].unique()
        if not pd.isna(ts)
    })
    if not train_sizes:
        print("[INFO] No train sizes detected for CPDAG combined plot.")
        return

    offsets = _build_offsets(train_sizes)
    marker_lookup = _build_marker_map(train_sizes)

    # Fixed dimensions for 2-panel combined plots
    height = COMBINED_FIGSIZE_DUO[1]
    panel_width = COMBINED_FIGSIZE_DUO[0] / 2

    # Prepare output paths - save directly to paper folder
    # Use correlation_matrix_difference slug for spearman metric to match original file names
    if metric == "correlation_matrix_difference":
        metric_slug = _get_cmd_config()["slug"]
    else:
        metric_slug = METRIC_CONFIG[metric]["slug"]
    paper_dir = PAPER_ROOT / "cpdag_minimal_discovered_combined"
    _ensure_dir(paper_dir)
    if no_csv:
        pdf_dir = paper_dir
        csv_dir = None
    else:
        pdf_dir = paper_dir / "pdf"
        csv_dir = paper_dir / "csv"
        _ensure_dir(pdf_dir)
        _ensure_dir(csv_dir)

    file_stem = f"forest_combined_cpdag_minimal_discovered_{metric_slug}"
    pdf_path = pdf_dir / f"{file_stem}.pdf"
    csv_path = csv_dir / f"{file_stem}.csv" if csv_dir is not None else None

    def _callback(plt_mod: Any) -> None:
        setup_forest_plot_style()
        # Same font sizes as other combined plots
        plt_mod.rcParams['font.size'] = LABEL_FONT_SIZE_COMBINED
        plt_mod.rcParams['axes.labelsize'] = LABEL_FONT_SIZE_COMBINED
        plt_mod.rcParams['axes.titlesize'] = TITLE_FONT_SIZE_SUBPLOT_COMBINED
        plt_mod.rcParams['xtick.labelsize'] = TICK_FONT_SIZE_COMBINED
        plt_mod.rcParams['ytick.labelsize'] = TICK_FONT_SIZE_COMBINED
        plt_mod.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE_COMBINED
        plt_mod.rcParams['legend.title_fontsize'] = LEGEND_TITLE_FONT_SIZE_COMBINED
        plt_mod.rcParams['figure.titlesize'] = TITLE_FONT_SIZE_SUPTITLE_COMBINED

        fig = plt_mod.gcf()
        fig.clear()

        # 2 panels (Minimal on left, Discovered on right)
        axes = fig.subplots(1, 2, sharey=True)

        # Custom color palette: Dark purple -> Blue -> Green -> Orange (replaces yellow) -> Red
        # Color-blind friendly palette that avoids yellow for better visibility
        # Based on viridis progression but with orange instead of yellow for Train 200
        custom_colors = ['#440154', '#31688e', '#35b779', '#f68f46', '#d73027']  # Dark purple, Blue, Green, Orange, Red
        color_lookup = {
            ts: custom_colors[i % len(custom_colors)]
            for i, ts in enumerate(train_sizes)
        }

        combined_legend_entries: Dict[int, Any] = {}

        # Subplot 1: Vanilla vs CPDAG Minimal
        ax1 = axes[0]
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        tick_override = TICK_FONT_SIZE_COMBINED  # closer to interventional sizing

        if vanilla_vs_minimal:
            legend_entries_minimal = render_forest_panel(
                ax1,
                df_minimal,
                datasets,
                dataset_labels,
                train_sizes,
                offsets,
                marker_lookup,
                color_lookup,
                is_combined=True,
                tick_font_size_override=tick_override,
            )
            combined_legend_entries.update(legend_entries_minimal)

            # Apply appropriate tick locator and tight limits based on data
            apply_xaxis_tick_locator(ax1, df=df_minimal)

            n_datasets = len(datasets)
            ax1.set_ylim(0.5 * DATASET_SPACING, (n_datasets + 0.5) * DATASET_SPACING)

            # Explicitly set tick label sizes to ensure consistency (matching interventional_experiment)
            y_positions = [(len(datasets) - idx) * DATASET_SPACING for idx in range(len(datasets))]
            labels = [dataset_labels[name] for name in datasets]
            _draw_left_aligned_yticklabels(
                ax1,
                y_positions,
                labels,
                font_size=TICK_FONT_SIZE_COMBINED,
                x_offset_points=Y_TICK_LABEL_OFFSET_COMBINED,
            )
            ax1.tick_params(axis="y", labelsize=TICK_FONT_SIZE_COMBINED)
            ax1.tick_params(axis="x", labelsize=TICK_FONT_SIZE_COMBINED, pad=TICK_PAD_X_COMBINED)
            # Force ticks outward for consistency - applied to both axes, all directions
            ax1.tick_params(axis="both", which="both", direction="out", left=True, right=False, bottom=True, top=False)

            comparison_minimal = COMPARISON_SPECS["original_cpdag_minimal_vs_vanilla"]
            baseline_label = comparison_minimal.baseline_label
            comparator_label = comparison_minimal.comparator_label

            ax1.set_title(
                f"{baseline_label} vs {comparator_label}",
                fontsize=TITLE_FONT_SIZE_SUBPLOT_COMBINED,
                pad=COMBINED_TITLE_PAD,
                fontweight='normal',
            )
            ax1.set_ylabel(
                "Dataset",
                fontsize=LABEL_FONT_SIZE_COMBINED,
                labelpad=X_LABEL_PAD + Y_LABEL_EXTRA_PAD_COMBINED,
            )

        # Subplot 2: Vanilla vs CPDAG Discovered
        ax2 = axes[1]
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        if vanilla_vs_discovered:
            legend_entries_discovered = render_forest_panel(
                ax2,
                df_discovered,
                datasets,
                dataset_labels,
                train_sizes,
                offsets,
                marker_lookup,
                color_lookup,
                is_combined=True,
                tick_font_size_override=tick_override,
            )
            combined_legend_entries.update(legend_entries_discovered)

            # Apply appropriate tick locator and tight limits based on data
            apply_xaxis_tick_locator(ax2, df=df_discovered)

            n_datasets = len(datasets)
            ax2.set_ylim(0.5 * DATASET_SPACING, (n_datasets + 0.5) * DATASET_SPACING)

            # Hide default y-axis tick labels for right subplot
            ax2.set_yticks([(len(datasets) - i) * DATASET_SPACING for i in range(len(datasets))])
            ax2.set_yticklabels([])
            ax2.tick_params(axis="x", labelsize=TICK_FONT_SIZE_COMBINED, pad=TICK_PAD_X_COMBINED)
            ax2.tick_params(axis="y", labelleft=False)
            # Force ticks outward for consistency - applied to both axes, all directions
            ax2.tick_params(axis="both", which="both", direction="out", left=True, right=False, bottom=True, top=False)
            # Ensure labels are hidden even if render_forest_panel set them
            for label in ax2.get_yticklabels():
                label.set_visible(False)

            # Set title for right subplot (was missing!)
            comparison_discovered = COMPARISON_SPECS["original_cpdag_discovered_vs_vanilla"]
            baseline_label_discovered = comparison_discovered.baseline_label
            comparator_label_discovered = comparison_discovered.comparator_label

            ax2.set_title(
                f"{baseline_label_discovered} vs {comparator_label_discovered}",
                fontsize=TITLE_FONT_SIZE_SUBPLOT_COMBINED,
                pad=COMBINED_TITLE_PAD,
                fontweight='normal',
            )
            ax2.set_ylabel("")  # No y-label on right subplot

        # Main title at top (metric name for CPDAG combined)
        # Use correlation_matrix_difference title for spearman metric to match original
        if metric == "correlation_matrix_difference":
            cmd_config = _get_cmd_config()
            metric_title = cmd_config["title"]
            metric_title = _normalize_metric_title("correlation_matrix_difference", metric_title)
            if cmd_config.get("slug") == "2marginal":
                metric_title = "k-Marginal Total Variation Distance"
        else:
            metric_title = METRIC_CONFIG[metric]['title']
            metric_title = _normalize_metric_title(metric, metric_title)
            if METRIC_CONFIG[metric].get("slug") == "2marginal":
                metric_title = "k-Marginal Total Variation Distance"

        fig.text(
            0.5, COMBINED_SUPTITLE_Y,
            metric_title,
            ha='center',
            va='top',
            fontsize=TITLE_FONT_SIZE_SUPTITLE_COMBINED,
            fontweight='normal',
        )

        # Shared x-axis label (baseline shared across both comparisons)
        shared_direction = METRIC_DIRECTIONS.get(metric, "lower")
        shared_baseline = COMPARISON_SPECS["original_cpdag_minimal_vs_vanilla"].baseline_label
        if shared_direction == "lower":
            shared_xlabel = f"Hodges–Lehmann diff ({shared_baseline} − comparator)"
        else:
            shared_xlabel = f"Hodges–Lehmann diff (comparator − {shared_baseline})"

        fig.text(
            0.5,
            COMBINED_SHARED_XLABEL_Y,
            shared_xlabel,
            ha="center",
            va="center",
            fontsize=LABEL_FONT_SIZE_COMBINED,
            fontweight='normal',
            transform=fig.transFigure,
        )

        handles = [combined_legend_entries[ts] for ts in train_sizes if ts in combined_legend_entries]
        labels = [handle.get_label() for handle in handles]

        # Legend positioning (aligned with other combined plots)
        if handles:
            bottom_margin = COMBINED_BOTTOM_MARGIN
            legend_y = COMBINED_LEGEND_Y
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, legend_y),
                bbox_transform=fig.transFigure,
                ncol=len(handles),
                frameon=False,
                fontsize=LEGEND_FONT_SIZE_COMBINED,
                title="Train Size",
                title_fontsize=LEGEND_TITLE_FONT_SIZE_COMBINED,
                columnspacing=LEGEND_COLUMNSPACING_COMBINED,
                handletextpad=LEGEND_HANDLETEXTPAD,
            )
        else:
            bottom_margin = COMBINED_BOTTOM_MARGIN_NO_LEGEND

        # Layout: use same settings as plot_combined_forest for consistency
        fig.subplots_adjust(
            left=0.20,
            right=0.98,  # Same as plot_combined_forest for 2-panel layout
            top=COMBINED_TOP_MARGIN,
            bottom=bottom_margin,
            wspace=COMBINED_WSPACE_DUO,  # Same as plot_combined_forest for 2-panel layout
        )

        # Align x-axis tick labels vertically across all panels
        fig.align_xlabels(axes)
        fig.align_ylabels(axes)

        original_tight_layout = plt_mod.tight_layout

        def _no_tight_layout(*args: Any, **kwargs: Any) -> None:
            fig.subplots_adjust(
                left=0.20,
                right=0.98,  # Same as plot_combined_forest for 2-panel layout
                top=COMBINED_TOP_MARGIN,
                bottom=bottom_margin,
                wspace=COMBINED_WSPACE_DUO,  # Same as plot_combined_forest for 2-panel layout
            )
            plt_mod.tight_layout = original_tight_layout

        plt_mod.tight_layout = _no_tight_layout

    figsize = (panel_width * 2, height)
    save_with_fastplot(_callback, pdf_path, figsize, pad_inches=0.20)

    # Save combined CSV with only essential columns
    if csv_path is not None:
        essential_columns = ["dataset", "train_size", "effect", "ci_lower", "ci_upper", "p_value", "p_holm", "comparison", "metric", "holm_significant_stepdown"]
        combined_rows = []
        for data, comparison_name in [(vanilla_vs_minimal, "vanilla_vs_cpdag_minimal"),
                                     (vanilla_vs_discovered, "vanilla_vs_cpdag_discovered")]:
            if data:
                for row in data:
                    row_dict = dict(row.__dict__)
                    row_dict["comparison"] = comparison_name
                    row_dict["metric"] = metric
                    # Keep only essential columns
                    filtered_dict = {k: v for k, v in row_dict.items() if k in essential_columns}
                    combined_rows.append(filtered_dict)

        if combined_rows:
            combined_df = pd.DataFrame(combined_rows)
            
            # Add is_significant column
            def _compute_significance_combined(row):
                p_holm = row.get("p_holm")
                effect = row.get("effect", 0.0)
                stepdown = row.get("holm_significant_stepdown")
                
                is_stat_sig = (
                     (p_holm is not None and not pd.isna(p_holm) and p_holm < 0.05)
                     or bool(stepdown)
                )

                if abs(effect) < 1e-9:
                    return False
                return is_stat_sig

            combined_df["is_significant"] = combined_df.apply(_compute_significance_combined, axis=1)

            # Rename metric
            combined_df["metric"] = combined_df["metric"].replace({
                "correlation_matrix_difference": "correlation_matrix_difference"
            })

            if "holm_significant_stepdown" in combined_df.columns:
                combined_df = combined_df.drop(columns=["holm_significant_stepdown"])

            combined_df.to_csv(csv_path, index=False)

    print(
        f"[SUCCESS] Saved CPDAG combined forest plot for metric "
        f"{metric} in {pdf_path}"
    )


def main(
    metric_names: Iterable[str],
    comparison_keys: Iterable[str],
    show_caption: bool = False,
    recompute_stats: bool = True,
    no_csv: bool = False,
) -> None:
    metric_names = tuple(metric_names)
    comparison_keys = tuple(comparison_keys)
    _ensure_dir(COMPARISON_RESULTS_DIR)
    _reset_paper_root()
    if recompute_stats:
        _ensure_stat_tests_from_csvs(COMPARISON_RESULT_FILES, metric_names, force_regeneration=True)

    nnaa_summaries: List[NnaaCheckSummary] = []
    rows_cache: Dict[Tuple[str, str], List[ForestRow]] = {}

    for metric in metric_names:
        for key in comparison_keys:
            try:
                comparison = COMPARISON_SPECS[key]
            except KeyError as exc:
                raise SystemExit(f"Unknown comparison '{key}'. Available: {sorted(COMPARISON_SPECS)}") from exc

            rows = _collect_forest_rows(metric, comparison)
            rows_cache[(metric, key)] = rows
            summary = plot_forest(metric, rows, comparison, show_caption=show_caption, no_csv=no_csv)
            if summary is not None:
                nnaa_summaries.append(summary)

    requested_metrics = set(metric_names)
    if COMBINED_COMPARISON_KEY in comparison_keys:
        comparison = COMPARISON_SPECS[COMBINED_COMPARISON_KEY]

        if set(COMBINED_METRIC_KEYS_DUO).issubset(requested_metrics):
            rows_lookup_duo = {
                metric: rows_cache.get((metric, COMBINED_COMPARISON_KEY), [])
                for metric in COMBINED_METRIC_KEYS_DUO
            }
            plot_combined_forest(
                COMBINED_METRIC_KEYS_DUO,
                rows_lookup_duo,
                comparison,
                show_caption=show_caption,
                no_csv=no_csv,
            )

    # Generate vanilla ordering combined plots if both vanilla ordering comparisons are requested
    if ("ordering_effects_vanilla_topological" in comparison_keys or
        "ordering_effects_vanilla_reverse_topological" in comparison_keys):
        for metric in requested_metrics:
            if metric in METRIC_CONFIG:
                plot_vanilla_ordering_combined_forest(metric, show_caption=show_caption, no_csv=no_csv)

    # Generate CPDAG combined plots if CPDAG comparisons are requested
    if ("original_cpdag_minimal_vs_vanilla" in comparison_keys or
        "original_cpdag_discovered_vs_vanilla" in comparison_keys):
        for metric in requested_metrics:
            if metric in METRIC_CONFIG:
                plot_cpdag_combined_forest(metric, show_caption=show_caption, no_csv=no_csv)

    # Generate Frobenius + k-Marginal combined plots for individual vanilla orderings
    if "ordering_effects_vanilla_topological" in comparison_keys:
        if set(COMBINED_METRIC_KEYS_DUO).issubset(requested_metrics):
            comparison = COMPARISON_SPECS["ordering_effects_vanilla_topological"]
            rows_lookup_topo = {
                metric: rows_cache.get((metric, "ordering_effects_vanilla_topological"), [])
                for metric in COMBINED_METRIC_KEYS_DUO
            }
            plot_combined_forest(
                COMBINED_METRIC_KEYS_DUO,
                rows_lookup_topo,
                comparison,
                show_caption=show_caption,
                no_csv=no_csv,
            )

    if "ordering_effects_vanilla_reverse_topological" in comparison_keys:
        if set(COMBINED_METRIC_KEYS_FROBENIUS_NNAA).issubset(requested_metrics):
            comparison = COMPARISON_SPECS["ordering_effects_vanilla_reverse_topological"]
            rows_lookup_worst = {
                metric: rows_cache.get((metric, "ordering_effects_vanilla_reverse_topological"), [])
                for metric in COMBINED_METRIC_KEYS_FROBENIUS_NNAA
            }
            plot_combined_forest(
                COMBINED_METRIC_KEYS_FROBENIUS_NNAA,
                rows_lookup_worst,
                comparison,
                show_caption=show_caption,
                no_csv=no_csv,
            )

    if "topological_dag_vs_vanilla" in comparison_keys:
        if set(COMBINED_METRIC_KEYS_DUO).issubset(requested_metrics):
            comparison = COMPARISON_SPECS["topological_dag_vs_vanilla"]
            rows_lookup_topo_dag = {
                metric: rows_cache.get((metric, "topological_dag_vs_vanilla"), [])
                for metric in COMBINED_METRIC_KEYS_DUO
            }
            plot_combined_forest(
                COMBINED_METRIC_KEYS_DUO,
                rows_lookup_topo_dag,
                comparison,
                show_caption=show_caption,
                no_csv=no_csv,
            )

    if nnaa_summaries:
        overall_min = min(summary.min_value for summary in nnaa_summaries)
        overall_max = max(summary.max_value for summary in nnaa_summaries)
        offenders: List[str] = []
        for summary in nnaa_summaries:
            offenders.extend(summary.high_points)
        offenders = sorted(set(offenders))
        if offenders:
            offenders_text = "; ".join(offenders)
            print(
                "[WARN] NNAA median summary: "
                f"min={overall_min:.3f}, max={overall_max:.3f}. "
                f"Values above 0.7 detected in {offenders_text}."
            )
        else:
            print(
                "[INFO] NNAA median summary: "
                f"min={overall_min:.3f}, max={overall_max:.3f}. "
                "All medians fall within (0.5, 0.7]."
            )
    elif any(metric == "nnaa" for metric in metric_names):
        print("[WARN] NNAA median summary: no valid data available to evaluate.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate monochrome-safe forest plots for Vanilla Original vs comparator conditions."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help=f"Metrics to include (default: {', '.join(DEFAULT_METRICS)})",
    )
    parser.add_argument(
        "--comparisons",
        nargs="+",
        choices=sorted(COMPARISON_SPECS.keys()),
        default=list(DEFAULT_COMPARISON_KEYS),
        help="Comparisons to include (default: all supported combinations)",
    )
    parser.add_argument(
        "--caption",
        dest="caption",
        action="store_true",
        help="Show an auto-generated caption in the plots.",
    )
    parser.add_argument(
        "--no-caption",
        dest="caption",
        action="store_false",
        help="Disable the caption in plots (default).",
    )
    parser.add_argument(
        "--skip-stats",
        dest="skip_stats",
        action="store_true",
        help="Skip recomputing statistical tests (reuse existing results)",
    )
    parser.add_argument(
        "--no-csv",
        dest="no_csv",
        action="store_true",
        help="Generate only PDF files without CSV and without pdf/ subdirectory",
    )
    parser.set_defaults(caption=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = list(args.metrics)
    comparisons = list(args.comparisons)

    main(
        metrics,
        comparisons,
        show_caption=args.caption,
        recompute_stats=not args.skip_stats,
        no_csv=args.no_csv,
    )
