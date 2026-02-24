#!/usr/bin/env python3
"""
Generate monochrome-safe forest plots for the interventional experiment, comparing Vanilla
and related baselines across multiple ordering conditions.

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
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_experiments.results.comparison_experiment.statistical_tests import (
    StatTestConfig,
)
from causal_experiments.results.interventional_experiment import (
    statistical_tests as interventional_stats,
)

# FastPlot provides the publication-ready styling layer we rely on for every figure.
import fastplot  # type: ignore

FormatStrFormatter = fastplot.mpl.ticker.FormatStrFormatter  # type: ignore[attr-defined]
Line2D = fastplot.mpl.lines.Line2D  # type: ignore[attr-defined]

import matplotlib.pyplot as plt  # type: ignore  # fastplot still uses matplotlib under the hood
from matplotlib.ticker import MultipleLocator  # type: ignore

from causal_experiments.utils.visualization_config import (
    METRIC_CONFIG,
    DPI,
    setup_plotting,
)
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
    _first_valid_value_with_source,
    apply_xaxis_tick_locator,
    _calculate_shared_xlim_from_dataframes,
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

STAT_TESTS_ROOT = SCRIPT_DIR / "stat_tests"
FOREST_ROOT = SCRIPT_DIR / "forest_plots"
PAPER_ROOT = FOREST_ROOT / "paper" / "interventional_experiment"

# Orientation config: value indicates how to orient the effect so that positive values
# reflect the comparator condition outperforming the baseline.
# - 'lower': smaller metric is better ⇒ keep baseline - comparator (positive = comparator better)
# - 'higher': larger metric is better ⇒ invert the sign so that positive = comparator better
METRIC_DIRECTIONS: Dict[str, str] = {
    "ate_difference": "lower",
}

# Default forest metrics to plot if none supplied via CLI.
DEFAULT_METRICS: Tuple[str, ...] = tuple(METRIC_DIRECTIONS.keys())

INTERVENTIONAL_RESULT_FILES: Tuple[str, ...] = tuple(
    name
    for name in interventional_stats.INTERVENTIONAL_RESULT_FILES
    if "lingauss" not in name
)

# Resolve result file paths relative to SCRIPT_DIR for consistency
INTERVENTIONAL_RESULT_PATHS: Tuple[Path, ...] = tuple(
    (SCRIPT_DIR / name)
    if not Path(name).is_absolute()
    else Path(name)
    for name in INTERVENTIONAL_RESULT_FILES
)

CONDITION_COLORS: Dict[str, str] = dict(interventional_stats.CONDITION_COLORS)

CONDITION_ORDER: Tuple[str, ...] = tuple(CONDITION_COLORS.keys())



FOREST_PAPER_METRICS: Tuple[str, ...] = ("ate_difference",)
PAPER_METRIC_SLUGS: Set[str] = {
    METRIC_CONFIG[m]["slug"]
    for m in FOREST_PAPER_METRICS
    if m in METRIC_CONFIG
}

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
    "original_dag_vs_vanilla": ComparisonSpec(
        slug="dag_vs_vanilla",
        baseline="vanilla_original",
        comparator="dag_original",
        baseline_label="Vanilla Original",
        comparator_label="DAG Original",
        title="Vanilla Original vs DAG Original",
        group_slug="ordering_original",
        group_label="Original ordering",
    ),
    "original_cpdag_minimal_vs_vanilla": ComparisonSpec(
        slug="cpdag_minimal_vs_vanilla",
        baseline="vanilla_original",
        comparator="cpdag_minimal_original",
        baseline_label="Vanilla Original",
        comparator_label="Minimal CPDAG",
        title="Vanilla Original vs Minimal CPDAG",
        group_slug="ordering_original",
        group_label="Original ordering",
    ),
    "original_cpdag_discovered_vs_vanilla": ComparisonSpec(
        slug="cpdag_discovered_vs_vanilla",
        baseline="vanilla_original",
        comparator="cpdag_discovered_original",
        baseline_label="Vanilla Original",
        comparator_label="Discovered CPDAG",
        title="Vanilla Original vs Discovered CPDAG",
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
}

DEFAULT_COMPARISON_KEYS: Tuple[str, ...] = tuple(COMPARISON_SPECS.keys())


def _dataset_slug_from_filename(path: Path) -> str:
    stem = path.stem
    if stem.startswith("result_"):
        stem = stem[len("result_") :]
    suffixes = (
        "_interventional_experiment_cleaned_reps_100",
        "_intervention_experiment_cleaned_reps_100",
        "_interventional_experiment_cleaned",
        "_intervention_experiment_cleaned",
        "_interventional_experiment",
        "_intervention_experiment",
    )
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


EXPECTED_DATASET_SLUGS: Tuple[str, ...] = tuple(
    _dataset_slug_from_filename(Path(name)) for name in INTERVENTIONAL_RESULT_FILES
)

DATASET_FILTER: Set[str] | None = None


def _set_expected_dataset_slugs(result_files: Iterable[str]) -> None:
    global EXPECTED_DATASET_SLUGS
    EXPECTED_DATASET_SLUGS = tuple(
        _dataset_slug_from_filename(Path(name)) for name in result_files
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
    "custom_scm_noise1e-2": "CSMn2",
    "simglucose": "SGL",
}


def _abbreviate_dataset_label_for_combined_plots(dataset_name: str, dataset_label: str) -> str:
    """Convert dataset label to acronym for combined plots to save horizontal space."""
    # Check if we have an acronym mapping for this dataset
    if dataset_name in DATASET_ACRONYMS:
        return DATASET_ACRONYMS[dataset_name]
    # Fallback to original label if no mapping exists
    return dataset_label




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
    return interventional_stats.build_stat_test_config(metrics)


def _ensure_stat_tests_from_csvs(csv_files: Iterable[str], metrics: Iterable[str]) -> None:
    interventional_stats.ensure_stat_tests_from_csvs(csv_files, metrics)




def _collect_forest_rows(metric: str, comparison: ComparisonSpec) -> List[ForestRow]:
    """
    Iterate through dataset directories, collect Hodges–Lehmann statistics for the
    specified baseline/comparator pair, and return normalized rows ready for plotting.
    """
    if metric not in METRIC_CONFIG:
        raise ValueError(f"Unknown metric '{metric}' (not in METRIC_CONFIG).")

    direction = METRIC_DIRECTIONS.get(metric, "lower")
    metric_slug = METRIC_CONFIG[metric]["slug"]

    rows: List[ForestRow] = []
    comparison_output_slug = comparison.output_slug

    if not STAT_TESTS_ROOT.exists():
        return rows

    if DATASET_FILTER is not None:
        allowed_datasets = set(DATASET_FILTER)
    else:
        allowed_datasets = set(EXPECTED_DATASET_SLUGS)
        # Include simglucose for comparisons involving vanilla_original or vanilla_topological
        if (comparison.baseline in ("vanilla_original", "vanilla_topological") or
            comparison.comparator in ("vanilla_original", "vanilla_topological")):
            # Add simglucose and any variants that might exist
            allowed_datasets.add("simglucose")
            allowed_datasets.add("simglucose_static_scm")
            # Also check for any directory starting with "simglucose" in stat_tests
            if STAT_TESTS_ROOT.exists():
                for dataset_dir in STAT_TESTS_ROOT.iterdir():
                    if dataset_dir.is_dir() and dataset_dir.name.startswith("simglucose"):
                        allowed_datasets.add(dataset_dir.name)

    for dataset_dir in sorted(STAT_TESTS_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        if dataset_name == "by_dataset_groups":
            continue
        if allowed_datasets and dataset_name not in allowed_datasets:
            continue

        posthoc_path = dataset_dir / "posthoc_wilcoxon_summary.csv"
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


def _normalize_noise_layout_fix_root(path: Path) -> Path:
    """Map legacy paper_noise1e-2/layout_fix paths to the promoted root."""
    normalized = str(path)
    normalized = normalized.replace("/paper_noise1e-2/layout_fix/", "/paper_noise1e-2/")
    normalized = normalized.replace("/paper_noise1e-2/layout_fix", "/paper_noise1e-2")
    return Path(normalized)




# Fixed figure sizes to ensure consistent dimensions across plots
# Fixed figure sizes to ensure consistent dimensions across plots
SINGLE_FIGSIZE = (11.0, 7.0)
SIMGLUCOSE_FIGSIZE = (11.0, 4.25)  # Reduced height for single-dataset simglucose plots (adjusted to 4.25" as requested)
COMBINED_FIGSIZE_DUO = (18.0, 6.5)  # Reduced height for better LaTeX fit

# Simglucose specific margins to maintain text spacing with reduced height
# Calculated to preserve exactly 1.40" bottom and 0.70" top space
# Bottom: 4.25 * 0.33 = 1.40" (matches 7.0 * 0.20)
# Top: 4.25 * (1 - 0.835) = 0.70" (matches 7.0 * (1 - 0.90))
SIMGLUCOSE_BOTTOM_MARGIN = 0.33
SIMGLUCOSE_TOP_MARGIN = 0.835


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


def _needs_extra_left(labels: Sequence[str]) -> bool:
    return any(("CSM-1e-2" in label) or ("CSMn2" in label) for label in labels)


def _compute_left_adjustments(
    labels: Sequence[str],
    base_tick_offset: float,
    base_labelpad: float,
    combined: bool = False,
) -> Tuple[float, float]:
    if not _needs_extra_left(labels):
        return base_tick_offset, base_labelpad

    ratio = TICK_FONT_SIZE_COMBINED / TICK_FONT_SIZE_SINGLE if combined else 1.0
    # Separate tuning for single vs combined plots.
    if combined:
        # Combined plots need a visible label->graph gap while keeping y-label readable.
        extra_tick = 18.0 * ratio
        extra_labelpad = 14.0 * ratio
    else:
        # Singles: preserve the graph gap and push "Dataset" slightly further left.
        extra_tick = 22.0 * ratio
        extra_labelpad = 16.0 * ratio

    tick_offset = base_tick_offset - extra_tick
    labelpad = base_labelpad + extra_labelpad
    return tick_offset, labelpad




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
    metric_slug = metric_cfg["slug"]
    metric_title = metric_cfg["title"]
    # Normalize metric title for consistent presentation
    metric_title = _normalize_metric_title(metric, metric_title)
    direction = METRIC_DIRECTIONS.get(metric, "lower")

    df = build_forest_dataframe(rows)

    datasets = sorted(df["dataset"].unique())
    dataset_labels = {
        name: _abbreviate_dataset_label_for_combined_plots(
            name, df.loc[df["dataset"] == name, "dataset_label"].iloc[0]
        )
        for name in datasets
    }
    
    # Separate simglucose datasets from others
    simglucose_datasets = [d for d in datasets if d.startswith("simglucose")]
    other_datasets = [d for d in datasets if not d.startswith("simglucose")]
    
    # For "Vanilla Original vs Vanilla Topological", exclude simglucose from main plot
    # (it will only appear in the separate simglucose plot)
    if comparison.slug == "vanilla_topological_vs_original":
        # Only create separate plots if simglucose datasets exist
        if simglucose_datasets:
            datasets_to_plot = [other_datasets, simglucose_datasets] if other_datasets else [simglucose_datasets]
        else:
            datasets_to_plot = [other_datasets] if other_datasets else []
    else:
        # For other comparisons, use the standard logic
        datasets_to_plot = [other_datasets, simglucose_datasets] if simglucose_datasets and other_datasets else [datasets]
    
    # Check if this comparison and metric should be exported to paper folder
    key = (comparison.baseline, comparison.comparator)
    folder_name = PAPER_FOLDER_MAP.get(key)
    if folder_name is None:
        # Skip plots that are not in PAPER_FOLDER_MAP
        return
    
    if metric not in FOREST_PAPER_METRICS:
        # Skip metrics that are not in FOREST_PAPER_METRICS
        return

    train_sizes = sorted(df["train_size"].unique())
    offsets = _build_offsets(train_sizes)
    marker_lookup = _build_marker_map(train_sizes)

    # Create plots for each group
    for plot_idx, plot_datasets in enumerate(datasets_to_plot):
        if not plot_datasets:
            continue
            
        is_simglucose_plot = bool(plot_datasets) and all(d.startswith("simglucose") for d in plot_datasets)
        
        # Fixed size for all single plots to keep dimensions consistent
        # Use reduced height for valid simglucose plots to avoid excessive whitespace
        if is_simglucose_plot:
            width, height = SIMGLUCOSE_FIGSIZE
        else:
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
        file_suffix = "_simglucose" if is_simglucose_plot else ""
        file_stem = f"forest_{comparison_slug}_{metric_slug}{file_suffix}"
        pdf_path = pdf_dir / f"{file_stem}.pdf"
        csv_path = csv_dir / f"{file_stem}.csv" if csv_dir is not None else None

        caption_text = _build_caption_text(metric_title, comparison, direction) if show_caption else None
        plot_labels = [dataset_labels[name] for name in plot_datasets]
        tick_offset, ylabel_pad = _compute_left_adjustments(
            plot_labels,
            Y_TICK_LABEL_OFFSET_SINGLE,
            X_LABEL_PAD + Y_LABEL_EXTRA_PAD,
            combined=False,
        )

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

            # Custom color palette: Dark purple -> Blue -> Green -> Orange (replaces yellow) -> Red -> Brown
            # Color-blind friendly palette that avoids yellow for better visibility
            # Based on viridis progression but with orange instead of yellow for Train 200
            # Added Brown (#8c564b) for Train 1000
            custom_colors = ['#440154', '#31688e', '#35b779', '#f68f46', '#d73027', '#8c564b']  # Dark purple, Blue, Green, Orange, Red, Brown
            color_lookup = {
                ts: custom_colors[i % len(custom_colors)]
                for i, ts in enumerate(train_sizes)
            }

            legend_entries = render_forest_panel(
                ax,
                df,
                plot_datasets,
                dataset_labels,
                train_sizes,
                offsets,
                marker_lookup,
                color_lookup,
            )

            # Apply appropriate tick locator and tight limits based on data
            apply_xaxis_tick_locator(ax, df=df)

            # Custom x-axis settings for simglucose plots
            # User requested: start from -10 and use ticks every 10 instead of every 20
            if is_simglucose_plot:
                current_xlim = ax.get_xlim()
                # Set left limit to -10
                ax.set_xlim(left=-10, right=current_xlim[1])
                # Set locator to step 10 instead of 20
                ax.xaxis.set_major_locator(MultipleLocator(10))

            # Set tight y-axis limits based on actual data positions with spacing
            n_datasets_plot = len(plot_datasets)
            ax.set_ylim(0.5 * DATASET_SPACING, (n_datasets_plot + 0.5) * DATASET_SPACING)

            # Explicitly set tick label sizes to ensure consistency with combined plots
            y_positions = [(len(plot_datasets) - i) * DATASET_SPACING for i in range(len(plot_datasets))]
            labels = plot_labels
            _draw_left_aligned_yticklabels(
                ax,
                y_positions,
                labels,
                font_size=TICK_FONT_SIZE_SINGLE,
                x_offset_points=tick_offset,
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
                labelpad=ylabel_pad,
            )

            if direction == "lower":
                xlabel = f"Hodges–Lehmann diff ({comparison.baseline_label} − {comparison.comparator_label})"
            else:
                xlabel = f"Hodges–Lehmann diff ({comparison.comparator_label} − {comparison.baseline_label})"

            ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE, labelpad=X_LABEL_PAD)
            # Escape LaTeX special characters (especially % which is a comment character)
            escaped_metric_title = metric_title.replace('%', r'\%')
            ax.set_title(f"{comparison.title} · {escaped_metric_title}", fontsize=TITLE_FONT_SIZE, pad=AXES_TITLE_PAD)

            handles = [legend_entries[ts] for ts in train_sizes if ts in legend_entries]
            labels = [handle.get_label() for handle in handles]
            is_custom_noise_case = _needs_extra_left(plot_labels)
            legend_fontsize = LEGEND_FONT_SIZE
            legend_title_fontsize = LEGEND_TITLE_FONT_SIZE
            legend_columnspacing = LEGEND_COLUMNSPACING_SINGLE
            legend_handletextpad = LEGEND_HANDLETEXTPAD

            # Legend below x-axis label, using axes coordinates for reliable positioning
            # Adjust anchor for simglucose plots to maintain physical distance from axis
            # Standard offset -0.15 corresponds to ~0.74" below a 4.9" high axis
            # For simglucose (4.25" height): Axis height ~2.15". 
            # To maintain ~0.75" physical distance: -0.75 / 2.15 = -0.35
            # User requested slightly more distance to match other plots: -0.42
            if is_simglucose_plot:
                legend_y_anchor = -0.42
            else:
                legend_y_anchor = -0.15

            if handles:
                if is_custom_noise_case and len(handles) > 5:
                    # Keep one-row legend like other plots, but compact spacing to avoid clipping.
                    labels = [label.replace("Train ", "Train") for label in labels]
                    if is_simglucose_plot:
                        legend_y_anchor = -0.46
                    else:
                        legend_y_anchor = -0.18
                    legend_fontsize = max(LEGEND_FONT_SIZE - 2, 12)
                    legend_title_fontsize = max(LEGEND_TITLE_FONT_SIZE - 2, 12)
                    legend_columnspacing = 0.8
                    legend_handletextpad = 0.5
                ax.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, legend_y_anchor),  # Below subplot in axes coords
                    ncol=len(handles),
                    frameon=False,
                    fontsize=legend_fontsize,
                    title="Train Size",
                    title_fontsize=legend_title_fontsize,
                    columnspacing=legend_columnspacing,
                    handletextpad=legend_handletextpad,
                )

            single_bottom_margin = SINGLE_BOTTOM_MARGIN
            simglucose_bottom_margin = SIMGLUCOSE_BOTTOM_MARGIN
            if is_custom_noise_case and len(handles) > 5:
                single_bottom_margin = max(SINGLE_BOTTOM_MARGIN, 0.24)
                simglucose_bottom_margin = max(SIMGLUCOSE_BOTTOM_MARGIN, 0.37)

            # FIXED margins for ALL single plots - ensures uniform dimensions
            # Increase left margin slightly for noise=1e-2 plots to prevent label overlap
            left_margin = SINGLE_LEFT_MARGIN
            if is_custom_noise_case:
                left_margin = max(SINGLE_LEFT_MARGIN, 0.22)  # Increase from default (~0.15) to 0.22 for noise=1e-2
            
            # Adjust margins for simglucose plots to preserve physical spacing with reduced height
            if is_simglucose_plot:
                fig.subplots_adjust(
                    left=left_margin,
                    right=SINGLE_RIGHT_MARGIN,
                    bottom=simglucose_bottom_margin,
                    top=SIMGLUCOSE_TOP_MARGIN,
                )
            else:
                fig.subplots_adjust(
                    left=left_margin,
                    right=SINGLE_RIGHT_MARGIN,
                    bottom=single_bottom_margin,
                    top=SINGLE_TOP_MARGIN,
                )

            # Disable tight_layout to preserve fixed margins
            original_tight_layout = plt_mod.tight_layout
            def _no_tight_layout(*args: Any, **kwargs: Any) -> None:
                if is_simglucose_plot:
                    fig.subplots_adjust(
                        left=left_margin,
                        right=SINGLE_RIGHT_MARGIN,
                        bottom=simglucose_bottom_margin,
                        top=SIMGLUCOSE_TOP_MARGIN,
                    )
                else:
                    fig.subplots_adjust(
                        left=left_margin,
                        right=SINGLE_RIGHT_MARGIN,
                        bottom=single_bottom_margin,
                        top=SINGLE_TOP_MARGIN,
                    )
                plt_mod.tight_layout = original_tight_layout
            plt_mod.tight_layout = _no_tight_layout

        # Use standard bbox with fixed padding to ensure consistent dimensions
        # Increased padding to include legend below the subplot area
        FIXED_PAD_INCHES = 0.5
        
        # User requested to trim the bottom white space completely for simglucose plots
        # We enable tight_bbox with minimal padding to crop to the content (legend)
        # This removes the bottom white space while preserving text dimensions and proportions
        if is_simglucose_plot:
            # Use tight bbox with minimal padding (0.12 is the minimum enforced by save_with_fastplot)
            # This will crop the PDF right to the legend, removing bottom white space
            save_with_fastplot(_callback, pdf_path, (width, height), use_tight_bbox=True, pad_inches=0.12)
        else:
            save_with_fastplot(_callback, pdf_path, (width, height), use_tight_bbox=False, pad_inches=FIXED_PAD_INCHES)
        

        if csv_path is not None:
            # Filter CSV data for this plot's datasets
            df_plot = df[df["dataset"].isin(plot_datasets)].copy()
            
            # Add is_significant column
            def _compute_significance(row):
                p_holm = row.get("p_holm")
                stepdown = row.get("holm_significant_stepdown")
                effect = row.get("effect")
                
                is_stat_sig = (
                    (p_holm is not None and not pd.isna(p_holm) and p_holm < 0.05)
                    or bool(stepdown)
                )
                if abs(effect) < 1e-9:
                    return False
                return is_stat_sig

            df_plot["is_significant"] = df_plot.apply(_compute_significance, axis=1)
            
            if metric == "correlation_matrix_difference":
                 df_plot["metric"] = "correlation_matrix_difference"
            
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
            df_csv = df_plot[[col for col in essential_columns if col in df_plot.columns]].copy()
            df_csv.to_csv(csv_path, index=False)

        plot_type = "SimGlucose" if is_simglucose_plot else "main"
        print(
            f"[SUCCESS] Saved {plot_type} forest plot for '{metric}' ({comparison.title}) in {pdf_path}"
        )


def plot_cpdag_combined_forest(
    metric: str,
    show_caption: bool = False,
    no_csv: bool = False,
) -> None:
    """Generate a combined forest plot showing Vanilla vs Minimal CPDAG and Vanilla vs Discovered CPDAG."""
    if metric not in METRIC_CONFIG:
        print(f"[WARN] Unknown metric '{metric}' for CPDAG combined forest plot.")
        return

    # Get data for the two comparisons
    vanilla_vs_minimal = _collect_forest_rows(metric, COMPARISON_SPECS["original_cpdag_minimal_vs_vanilla"])
    vanilla_vs_discovered = _collect_forest_rows(metric, COMPARISON_SPECS["original_cpdag_discovered_vs_vanilla"])

    if not vanilla_vs_minimal and not vanilla_vs_discovered:
        print("[INFO] No data available for CPDAG combined plot.")
        return

    # Build dataframes
    df_minimal = build_forest_dataframe(vanilla_vs_minimal)
    df_discovered = build_forest_dataframe(vanilla_vs_discovered)

    # Filter out simglucose datasets from dataframes
    if not df_minimal.empty:
        df_minimal = df_minimal[~df_minimal["dataset"].str.startswith("simglucose", na=False)].copy()
    if not df_discovered.empty:
        df_discovered = df_discovered[~df_discovered["dataset"].str.startswith("simglucose", na=False)].copy()

    # Get all datasets (simglucose already filtered out)
    all_datasets = set()
    for df in [df_minimal, df_discovered]:
        if not df.empty:
            all_datasets.update(df["dataset"].unique())

    datasets = sorted(all_datasets)
    if not datasets:
        print("[INFO] No datasets available for CPDAG combined plot.")
        return

    # Build dataset labels
    dataset_labels: Dict[str, str] = {}
    for df in [df_minimal, df_discovered]:
        if not df.empty:
            # Re-apply formatting just in case
            df["dataset_label"] = df["dataset"].apply(_format_display_label)
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

    # Use fixed duo layout for all combined plots
    height = COMBINED_FIGSIZE_DUO[1]
    total_width = COMBINED_FIGSIZE_DUO[0]
    panel_width = total_width / 2

    # Prepare output paths
    # Check if metric should be exported to paper folder
    if metric not in FOREST_PAPER_METRICS:
        return

    metric_cfg = METRIC_CONFIG[metric]
    metric_slug = metric_cfg["slug"]
    metric_title = metric_cfg["title"]
    # Normalize metric title for consistent presentation
    metric_title = _normalize_metric_title(metric, metric_title)
    # Save directly to paper folder
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

    figsize = (panel_width * 2, height)
    direction = METRIC_DIRECTIONS.get(metric, "lower")

    def _callback(plt_mod: Any) -> None:
        setup_forest_plot_style()
        # Force font sizes via rcParams to override fastplot defaults
        plt_mod.rcParams['font.size'] = LABEL_FONT_SIZE_COMBINED
        plt_mod.rcParams['axes.labelsize'] = LABEL_FONT_SIZE_COMBINED
        plt_mod.rcParams['axes.titlesize'] = TITLE_FONT_SIZE_SUBPLOT_COMBINED
        plt_mod.rcParams['xtick.labelsize'] = TICK_FONT_SIZE_COMBINED
        plt_mod.rcParams['ytick.labelsize'] = TICK_FONT_SIZE_COMBINED
        plt_mod.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE_COMBINED
        plt_mod.rcParams['legend.title_fontsize'] = LEGEND_TITLE_FONT_SIZE_COMBINED

        fig = plt_mod.gcf()
        fig.clear()
        axes = fig.subplots(1, 2, sharey=True)

        # Custom color palette: Dark purple -> Blue -> Green -> Orange (replaces yellow) -> Red -> Brown
        # Color-blind friendly palette that avoids yellow for better visibility
        # Based on viridis progression but with orange instead of yellow for Train 200
        # Added Brown (#8c564b) for Train 1000
        custom_colors = ['#440154', '#31688e', '#35b779', '#f68f46', '#d73027', '#8c564b']  # Dark purple, Blue, Green, Orange, Red, Brown
        color_lookup = {
            ts: custom_colors[i % len(custom_colors)]
            for i, ts in enumerate(train_sizes)
        }

        combined_legend_entries: Dict[int, Line2D] = {}

        # Subplot 1: Vanilla vs Minimal CPDAG
        ax1 = axes[0]
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        if not df_minimal.empty:
            entries = render_forest_panel(
                ax1,
                df_minimal,
                datasets,
                dataset_labels,
                train_sizes,
                offsets,
                marker_lookup,
                color_lookup,
                is_combined=True,
            )
            combined_legend_entries.update(entries)

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
                labelpad=_compute_left_adjustments(
                    [dataset_labels[name] for name in datasets],
                    Y_TICK_LABEL_OFFSET_COMBINED,
                    X_LABEL_PAD + Y_LABEL_EXTRA_PAD_COMBINED,
                    combined=True,
                )[1],
            )

        # Subplot 2: Vanilla vs Discovered CPDAG
        ax2 = axes[1]
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        if not df_discovered.empty:
            entries = render_forest_panel(
                ax2,
                df_discovered,
                datasets,
                dataset_labels,
                train_sizes,
                offsets,
                marker_lookup,
                color_lookup,
                is_combined=True,
            )
            combined_legend_entries.update(entries)

            # Set tight y-axis limits based on actual data positions with spacing
            n_datasets = len(datasets)
            ax2.set_ylim(0.5 * DATASET_SPACING, (n_datasets + 0.5) * DATASET_SPACING)

            # Hide default y-axis tick labels for right subplot
            y_positions = [(len(datasets) - i) * DATASET_SPACING for i in range(len(datasets))]
            ax2.set_yticks(y_positions)
            ax2.set_yticklabels([])
            ax2.tick_params(axis="x", labelsize=TICK_FONT_SIZE_COMBINED, pad=TICK_PAD_X_COMBINED)
            ax2.tick_params(axis="y", labelleft=False)
            # Force ticks outward for consistency - applied to both axes, all directions
            ax2.tick_params(axis="both", which="both", direction="out", left=True, right=False, bottom=True, top=False)
            # Ensure labels are hidden even if render_forest_panel set them
            for label in ax2.get_yticklabels():
                label.set_visible(False)

            comparison_discovered = COMPARISON_SPECS["original_cpdag_discovered_vs_vanilla"]
            baseline_label = comparison_discovered.baseline_label
            comparator_label = comparison_discovered.comparator_label

            ax2.set_title(
                f"{baseline_label} vs {comparator_label}",
                fontsize=TITLE_FONT_SIZE_SUBPLOT_COMBINED,
                pad=COMBINED_TITLE_PAD,
                fontweight='normal',
            )
            ax2.set_ylabel("")

        # Apply individual tick locators and limits for each panel
        # Each panel gets its own optimal scale (no shared limits for CPDAG plots)
        if not df_minimal.empty:
            apply_xaxis_tick_locator(ax1, df=df_minimal)
        if not df_discovered.empty:
            apply_xaxis_tick_locator(ax2, df=df_discovered)

        # Set tight y-axis limits for both panels
        n_datasets = len(datasets)
        if not df_minimal.empty:
            ax1.set_ylim(0.5 * DATASET_SPACING, (n_datasets + 0.5) * DATASET_SPACING)
            # Explicitly set tick label sizes to ensure consistency across all plots
            y_positions = [(len(datasets) - i) * DATASET_SPACING for i in range(len(datasets))]
            labels = [dataset_labels[name] for name in datasets]
            tick_offset_combined, _ = _compute_left_adjustments(
                labels,
                Y_TICK_LABEL_OFFSET_COMBINED,
                X_LABEL_PAD + Y_LABEL_EXTRA_PAD_COMBINED,
                combined=True,
            )
            _draw_left_aligned_yticklabels(
                ax1,
                y_positions,
                labels,
                font_size=TICK_FONT_SIZE_COMBINED,
                x_offset_points=tick_offset_combined,
            )
            ax1.tick_params(axis="y", labelsize=TICK_FONT_SIZE_COMBINED)
            ax1.tick_params(axis="x", labelsize=TICK_FONT_SIZE_COMBINED, pad=TICK_PAD_X_COMBINED)
            ax1.tick_params(axis="both", which="both", direction="out", left=True, right=False, bottom=True, top=False)

        if not df_discovered.empty:
            ax2.set_ylim(0.5 * DATASET_SPACING, (n_datasets + 0.5) * DATASET_SPACING)
            # Hide default y-axis tick labels for right subplot
            y_positions = [(len(datasets) - i) * DATASET_SPACING for i in range(len(datasets))]
            ax2.set_yticks(y_positions)
            ax2.set_yticklabels([])
            ax2.tick_params(axis="x", labelsize=TICK_FONT_SIZE_COMBINED, pad=TICK_PAD_X_COMBINED)
            ax2.tick_params(axis="y", labelleft=False)
            ax2.tick_params(axis="both", which="both", direction="out", left=True, right=False, bottom=True, top=False)
            # Ensure labels are hidden even if render_forest_panel set them
            for label in ax2.get_yticklabels():
                label.set_visible(False)
            ax2.set_ylabel("")

        # Shared x-axis label (baseline shared across both comparisons)
        shared_baseline = COMPARISON_SPECS["original_cpdag_minimal_vs_vanilla"].baseline_label
        if direction == "lower":
            shared_xlabel = f"Hodges–Lehmann diff ({shared_baseline} − comparator)"
        else:
            shared_xlabel = f"Hodges–Lehmann diff (comparator − {shared_baseline})"

        # Legend at bottom
        handles = [combined_legend_entries[ts] for ts in train_sizes if ts in combined_legend_entries]
        labels = [handle.get_label() for handle in handles]

        if handles:
            bottom_margin = COMBINED_BOTTOM_MARGIN
            legend_y = COMBINED_LEGEND_Y
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, legend_y),
                bbox_transform=fig.transFigure,
                ncol=len(handles),  # Keep one-row legend layout consistent with existing plots
                frameon=False,
                fontsize=LEGEND_FONT_SIZE_COMBINED,
                title="Train Size",
                title_fontsize=LEGEND_TITLE_FONT_SIZE_COMBINED,
                columnspacing=LEGEND_COLUMNSPACING_COMBINED,
                handletextpad=LEGEND_HANDLETEXTPAD,
            )
        else:
            bottom_margin = COMBINED_BOTTOM_MARGIN_NO_LEGEND

        # Keep the same margins used by standard plots.
        left_margin = 0.20
        right_margin = 0.98
        wspace = COMBINED_WSPACE_DUO

        # Maximize horizontal space; use consistent top margin from config
        fig.subplots_adjust(
            left=left_margin,
            right=right_margin,
            top=COMBINED_TOP_MARGIN,
            bottom=bottom_margin,
            wspace=wspace,
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

        # Main title at top (metric name) - ADD AFTER subplots_adjust
        # Escape LaTeX special characters (especially % which is a comment character)
        escaped_metric_title = metric_title.replace('%', r'\%')
        fig.text(
            0.5, COMBINED_SUPTITLE_Y,
            escaped_metric_title,
            ha='center',
            va='top',
            fontsize=TITLE_FONT_SIZE_SUPTITLE_COMBINED,
            fontweight='normal',
            transform=fig.transFigure,
        )

        original_tight_layout = plt_mod.tight_layout
        def _no_tight_layout(*args: Any, **kwargs: Any) -> None:
            fig.subplots_adjust(
                left=left_margin,
                right=0.98,
                top=COMBINED_TOP_MARGIN,
                bottom=bottom_margin,
                wspace=COMBINED_WSPACE_DUO,
            )
            plt_mod.tight_layout = original_tight_layout
        plt_mod.tight_layout = _no_tight_layout

    # Save using fastplot
    save_with_fastplot(_callback, pdf_path, figsize, pad_inches=0.20)

    # Save combined CSV with only essential columns
    # Added holm_significant_stepdown to essential columns
    essential_columns = ["dataset", "train_size", "effect", "ci_lower", "ci_upper", "p_value", "p_holm", "comparison", "holm_significant_stepdown"]
    combined_rows = []
    for data, comparison_name in [(vanilla_vs_minimal, "vanilla_vs_cpdag_minimal"),
                                 (vanilla_vs_discovered, "vanilla_vs_cpdag_discovered")]:
        if data:
            for row in data:
                row_dict = dict(row.__dict__)
                row_dict["comparison"] = comparison_name
                # Keep only essential columns - ensure stepdown is kept if present
                filtered_dict = {k: v for k, v in row_dict.items() if k in essential_columns}
                combined_rows.append(filtered_dict)

    if csv_path is not None:
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

            if metric == "correlation_matrix_difference":
                combined_df["metric"] = "correlation_matrix_difference"

            if "holm_significant_stepdown" in combined_df.columns:
                combined_df = combined_df.drop(columns=["holm_significant_stepdown"])

            combined_df.to_csv(csv_path, index=False)

    print(
        f"[SUCCESS] Saved CPDAG combined forest plot for metric "
        f"{metric} in {pdf_path}"
    )


def plot_dag_cpdag_minimal_combined_forest(
    metric: str,
    show_caption: bool = False,
    no_csv: bool = False,
) -> None:
    """Generate a combined forest plot showing Vanilla vs DAG (left) and Vanilla vs Minimal CPDAG (right)."""
    if metric not in METRIC_CONFIG:
        print(f"[WARN] Unknown metric '{metric}' for DAG+Minimal CPDAG combined forest plot.")
        return

    # Get data for the two comparisons
    vanilla_vs_dag = _collect_forest_rows(metric, COMPARISON_SPECS["cross_dag_topological_vs_vanilla_original"])
    vanilla_vs_minimal = _collect_forest_rows(metric, COMPARISON_SPECS["original_cpdag_minimal_vs_vanilla"])

    if not vanilla_vs_dag and not vanilla_vs_minimal:
        print("[INFO] No data available for DAG+Minimal CPDAG combined plot.")
        return

    # Build dataframes
    df_dag = build_forest_dataframe(vanilla_vs_dag)
    df_minimal = build_forest_dataframe(vanilla_vs_minimal)

    # Filter out simglucose datasets from dataframes
    if not df_dag.empty:
        df_dag = df_dag[~df_dag["dataset"].str.startswith("simglucose", na=False)].copy()
    if not df_minimal.empty:
        df_minimal = df_minimal[~df_minimal["dataset"].str.startswith("simglucose", na=False)].copy()

    # Keep canonical paper plots free from noise-specific datasets unless explicitly
    # exporting under a noise paper root.
    if "paper_noise1e-2" not in str(PAPER_ROOT):
        if not df_dag.empty:
            df_dag = df_dag[~df_dag["dataset"].str.contains("noise1e-2", na=False)].copy()
        if not df_minimal.empty:
            df_minimal = df_minimal[~df_minimal["dataset"].str.contains("noise1e-2", na=False)].copy()

    # Get all datasets (simglucose already filtered out)
    all_datasets = set()
    for df in [df_dag, df_minimal]:
        if not df.empty:
            all_datasets.update(df["dataset"].unique())

    datasets = sorted(all_datasets)
    if not datasets:
        print("[INFO] No datasets available for DAG+Minimal CPDAG combined plot.")
        return

    # Build dataset labels
    dataset_labels: Dict[str, str] = {}
    for df in [df_dag, df_minimal]:
        if not df.empty:
            # Re-apply formatting just in case
            df["dataset_label"] = df["dataset"].apply(_format_display_label)
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
        for df in [df_dag, df_minimal]
        for ts in df["train_size"].unique()
        if not pd.isna(ts)
    })
    if not train_sizes:
        print("[INFO] No train sizes detected for DAG+Minimal CPDAG combined plot.")
        return

    offsets = _build_offsets(train_sizes)
    marker_lookup = _build_marker_map(train_sizes)

    # Use fixed duo layout for all combined plots
    height = COMBINED_FIGSIZE_DUO[1]
    total_width = COMBINED_FIGSIZE_DUO[0]
    panel_width = total_width / 2

    # Check if metric should be exported to paper folder
    if metric not in FOREST_PAPER_METRICS:
        return

    # Prepare output paths - save directly to paper folder
    metric_cfg = METRIC_CONFIG[metric]
    metric_slug = metric_cfg["slug"]
    metric_title = metric_cfg["title"]
    # Normalize metric title for consistent presentation
    metric_title = _normalize_metric_title(metric, metric_title)
    paper_dir = PAPER_ROOT / "dag_cpdag_minimal_combined"
    _ensure_dir(paper_dir)
    if no_csv:
        pdf_dir = paper_dir
        csv_dir = None
    else:
        pdf_dir = paper_dir / "pdf"
        csv_dir = paper_dir / "csv"
        _ensure_dir(pdf_dir)
        _ensure_dir(csv_dir)

    file_stem = f"forest_combined_dag_and_cpdag_minimal_{metric_slug}"
    pdf_path = pdf_dir / f"{file_stem}.pdf"
    csv_path = csv_dir / f"{file_stem}.csv" if csv_dir is not None else None

    figsize = (panel_width * 2, height)
    direction = METRIC_DIRECTIONS.get(metric, "lower")

    def _callback(plt_mod: Any) -> None:
        setup_forest_plot_style()
        # Force font sizes via rcParams to override fastplot defaults
        plt_mod.rcParams['font.size'] = LABEL_FONT_SIZE_COMBINED
        plt_mod.rcParams['axes.labelsize'] = LABEL_FONT_SIZE_COMBINED
        plt_mod.rcParams['axes.titlesize'] = TITLE_FONT_SIZE_SUBPLOT_COMBINED
        plt_mod.rcParams['xtick.labelsize'] = TICK_FONT_SIZE_COMBINED
        plt_mod.rcParams['ytick.labelsize'] = TICK_FONT_SIZE_COMBINED
        plt_mod.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE_COMBINED
        plt_mod.rcParams['legend.title_fontsize'] = LEGEND_TITLE_FONT_SIZE_COMBINED

        fig = plt_mod.gcf()
        fig.clear()
        axes = fig.subplots(1, 2, sharey=True)

        # Custom color palette: Dark purple -> Blue -> Green -> Orange (replaces yellow) -> Red
        # Color-blind friendly palette that avoids yellow for better visibility
        # Based on viridis progression but with orange instead of yellow for Train 200
        # Custom color palette: Dark purple -> Blue -> Green -> Orange (replaces yellow) -> Red -> Brown
        # Color-blind friendly palette that avoids yellow for better visibility
        # Based on viridis progression but with orange instead of yellow for Train 200
        # Added Brown (#8c564b) for Train 1000
        custom_colors = ['#440154', '#31688e', '#35b779', '#f68f46', '#d73027', '#8c564b']  # Dark purple, Blue, Green, Orange, Red, Brown
        color_lookup = {
            ts: custom_colors[i % len(custom_colors)]
            for i, ts in enumerate(train_sizes)
        }

        combined_legend_entries: Dict[int, Line2D] = {}

        # Subplot 1: Vanilla vs DAG
        ax1 = axes[0]
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        if not df_dag.empty:
            entries = render_forest_panel(
                ax1,
                df_dag,
                datasets,
                dataset_labels,
                train_sizes,
                offsets,
                marker_lookup,
                color_lookup,
                is_combined=True,
            )
            combined_legend_entries.update(entries)

            comparison_dag = COMPARISON_SPECS["cross_dag_topological_vs_vanilla_original"]
            baseline_label = comparison_dag.baseline_label
            # Use "DAG" instead of "DAG Original" for the combined plot title
            comparator_label = "DAG"

            ax1.set_title(
                f"{baseline_label} vs {comparator_label}",
                fontsize=TITLE_FONT_SIZE_SUBPLOT_COMBINED,
                pad=COMBINED_TITLE_PAD,
                fontweight='normal',
            )
            ax1.set_ylabel(
                "Dataset",
                fontsize=LABEL_FONT_SIZE_COMBINED,
                labelpad=_compute_left_adjustments(
                    [dataset_labels[name] for name in datasets],
                    Y_TICK_LABEL_OFFSET_COMBINED,
                    X_LABEL_PAD + Y_LABEL_EXTRA_PAD_COMBINED,
                    combined=True,
                )[1],
            )

        # Subplot 2: Vanilla vs Minimal CPDAG
        ax2 = axes[1]
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        if not df_minimal.empty:
            entries = render_forest_panel(
                ax2,
                df_minimal,
                datasets,
                dataset_labels,
                train_sizes,
                offsets,
                marker_lookup,
                color_lookup,
                is_combined=True,
            )
            combined_legend_entries.update(entries)

            # Set tight y-axis limits based on actual data positions with spacing
            n_datasets = len(datasets)
            ax2.set_ylim(0.5 * DATASET_SPACING, (n_datasets + 0.5) * DATASET_SPACING)

            # Hide default y-axis tick labels for right subplot
            y_positions = [(len(datasets) - i) * DATASET_SPACING for i in range(len(datasets))]
            ax2.set_yticks(y_positions)
            ax2.set_yticklabels([])
            ax2.tick_params(axis="x", labelsize=TICK_FONT_SIZE_COMBINED, pad=TICK_PAD_X_COMBINED)
            ax2.tick_params(axis="y", labelleft=False)
            # Force ticks outward for consistency - applied to both axes, all directions
            ax2.tick_params(axis="both", which="both", direction="out", left=True, right=False, bottom=True, top=False)
            # Ensure labels are hidden even if render_forest_panel set them
            for label in ax2.get_yticklabels():
                label.set_visible(False)

            comparison_minimal = COMPARISON_SPECS["original_cpdag_minimal_vs_vanilla"]
            baseline_label = comparison_minimal.baseline_label
            comparator_label = comparison_minimal.comparator_label

            ax2.set_title(
                f"{baseline_label} vs {comparator_label}",
                fontsize=TITLE_FONT_SIZE_SUBPLOT_COMBINED,
                pad=COMBINED_TITLE_PAD,
                fontweight='normal',
            )
            ax2.set_ylabel("")

        # Use a shared x-axis scale across both panels for direct comparability.
        shared_xlim = _calculate_shared_xlim_from_dataframes(
            [df for df in [df_dag, df_minimal] if not df.empty],
            step=0.2,
        )
        if metric == "ate_difference":
            # Keep the established paper scale for this combined ATE plot.
            shared_xlim = (min(shared_xlim[0], -0.2), max(shared_xlim[1], 1.6))

        if not df_dag.empty:
            apply_xaxis_tick_locator(ax1, shared_xlim=shared_xlim)
            ax1.xaxis.set_major_locator(MultipleLocator(0.2))
            ax1.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        if not df_minimal.empty:
            apply_xaxis_tick_locator(ax2, shared_xlim=shared_xlim)
            ax2.xaxis.set_major_locator(MultipleLocator(0.2))
            ax2.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        # Set tight y-axis limits for both panels
        n_datasets = len(datasets)
        if not df_dag.empty:
            ax1.set_ylim(0.5 * DATASET_SPACING, (n_datasets + 0.5) * DATASET_SPACING)
            # Explicitly set tick label sizes to ensure consistency across all plots
            y_positions = [(len(datasets) - i) * DATASET_SPACING for i in range(len(datasets))]
            labels = [dataset_labels[name] for name in datasets]
            tick_offset_combined, _ = _compute_left_adjustments(
                labels,
                Y_TICK_LABEL_OFFSET_COMBINED,
                X_LABEL_PAD + Y_LABEL_EXTRA_PAD_COMBINED,
                combined=True,
            )
            _draw_left_aligned_yticklabels(
                ax1,
                y_positions,
                labels,
                font_size=TICK_FONT_SIZE_COMBINED,
                x_offset_points=tick_offset_combined,
            )
            ax1.tick_params(axis="y", labelsize=TICK_FONT_SIZE_COMBINED)
            ax1.tick_params(axis="x", labelsize=TICK_FONT_SIZE_COMBINED, pad=TICK_PAD_X_COMBINED)
            ax1.tick_params(axis="both", which="both", direction="out", left=True, right=False, bottom=True, top=False)

        if not df_minimal.empty:
            ax2.set_ylim(0.5 * DATASET_SPACING, (n_datasets + 0.5) * DATASET_SPACING)
            # Hide default y-axis tick labels for right subplot
            y_positions = [(len(datasets) - i) * DATASET_SPACING for i in range(len(datasets))]
            ax2.set_yticks(y_positions)
            ax2.set_yticklabels([])
            ax2.tick_params(axis="x", labelsize=TICK_FONT_SIZE_COMBINED, pad=TICK_PAD_X_COMBINED)
            ax2.tick_params(axis="y", labelleft=False)
            ax2.tick_params(axis="both", which="both", direction="out", left=True, right=False, bottom=True, top=False)
            # Ensure labels are hidden even if render_forest_panel set them
            for label in ax2.get_yticklabels():
                label.set_visible(False)
            ax2.set_ylabel("")

        # Shared x-axis label (baseline shared across both comparisons)
        shared_baseline = COMPARISON_SPECS["cross_dag_topological_vs_vanilla_original"].baseline_label
        if direction == "lower":
            shared_xlabel = f"Hodges–Lehmann diff ({shared_baseline} − comparator)"
        else:
            shared_xlabel = f"Hodges–Lehmann diff (comparator − {shared_baseline})"

        # Legend at bottom
        handles = [combined_legend_entries[ts] for ts in train_sizes if ts in combined_legend_entries]
        labels = [handle.get_label() for handle in handles]

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

        # Keep the same margins used by standard plots.
        left_margin = 0.20
        right_margin = 0.98
        wspace = COMBINED_WSPACE_DUO

        # Maximize horizontal space; use consistent top margin from config
        fig.subplots_adjust(
            left=left_margin,
            right=right_margin,
            top=COMBINED_TOP_MARGIN,
            bottom=bottom_margin,
            wspace=wspace,
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

        # Main title at top (metric name) - ADD AFTER subplots_adjust
        # Escape LaTeX special characters (especially % which is a comment character)
        escaped_metric_title = metric_title.replace('%', r'\%')
        fig.text(
            0.5, COMBINED_SUPTITLE_Y,
            escaped_metric_title,
            ha='center',
            va='top',
            fontsize=TITLE_FONT_SIZE_SUPTITLE_COMBINED,
            fontweight='normal',
            transform=fig.transFigure,
        )

        original_tight_layout = plt_mod.tight_layout
        def _no_tight_layout(*args: Any, **kwargs: Any) -> None:
            fig.subplots_adjust(
                left=left_margin,
                right=0.98,
                top=COMBINED_TOP_MARGIN,
                bottom=bottom_margin,
                wspace=COMBINED_WSPACE_DUO,
            )
            plt_mod.tight_layout = original_tight_layout
        plt_mod.tight_layout = _no_tight_layout

    # Save using fastplot
    save_with_fastplot(_callback, pdf_path, figsize, pad_inches=0.20)

    # Save combined CSV with only essential columns
    essential_columns = ["dataset", "train_size", "effect", "ci_lower", "ci_upper", "p_value", "p_holm", "comparison", "holm_significant_stepdown"]
    exclude_noise = "paper_noise1e-2" not in str(PAPER_ROOT)
    combined_rows = []
    for data, comparison_name in [(vanilla_vs_dag, "vanilla_vs_dag"),
                                 (vanilla_vs_minimal, "vanilla_vs_minimal")]:
        if data:
            for row in data:
                row_dict = dict(row.__dict__)
                dataset_name = str(row_dict.get("dataset", ""))
                if dataset_name.startswith("simglucose"):
                    continue
                if exclude_noise and "noise1e-2" in dataset_name:
                    continue
                row_dict["comparison"] = comparison_name
                # Keep only essential columns
                filtered_dict = {k: v for k, v in row_dict.items() if k in essential_columns}
                combined_rows.append(filtered_dict)

    if csv_path is not None:
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

            if metric == "correlation_matrix_difference":
                combined_df["metric"] = "correlation_matrix_difference"

            if "holm_significant_stepdown" in combined_df.columns:
                combined_df = combined_df.drop(columns=["holm_significant_stepdown"])

            combined_df.to_csv(csv_path, index=False)

    print(
        f"[SUCCESS] Saved DAG+Minimal CPDAG combined forest plot for metric "
        f"{metric} in {pdf_path}"
    )


def plot_vanilla_ordering_combined_forest(
    metric: str,
    show_caption: bool = False,
    no_csv: bool = False,
) -> None:
    """Generate a combined forest plot showing Vanilla ordering effects (Topological and Worst vs Original)."""
    if metric not in METRIC_CONFIG:
        print(f"[WARN] Unknown metric '{metric}' for Vanilla ordering combined forest plot.")
        return

    # Get data for the two comparisons
    vanilla_orig_vs_topo = _collect_forest_rows(metric, COMPARISON_SPECS["ordering_effects_vanilla_topological"])
    vanilla_orig_vs_worst = _collect_forest_rows(metric, COMPARISON_SPECS["ordering_effects_vanilla_reverse_topological"])

    if not vanilla_orig_vs_topo and not vanilla_orig_vs_worst:
        print("[INFO] No data available for Vanilla ordering combined plot.")
        return

    # Build dataframes
    df_topo = build_forest_dataframe(vanilla_orig_vs_topo)
    df_worst = build_forest_dataframe(vanilla_orig_vs_worst)

    # Filter out simglucose datasets from dataframes
    if not df_topo.empty:
        df_topo = df_topo[~df_topo["dataset"].str.startswith("simglucose", na=False)].copy()
    if not df_worst.empty:
        df_worst = df_worst[~df_worst["dataset"].str.startswith("simglucose", na=False)].copy()

    # Get all datasets (simglucose already filtered out)
    all_datasets = set()
    for df in [df_topo, df_worst]:
        if not df.empty:
            all_datasets.update(df["dataset"].unique())

    datasets = sorted(all_datasets)
    if not datasets:
        print("[INFO] No datasets available for Vanilla ordering combined plot.")
        return

    # Build dataset labels
    dataset_labels: Dict[str, str] = {}
    for df in [df_topo, df_worst]:
        if not df.empty:
            df["dataset_label"] = df["dataset"].apply(_format_display_label)
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

    # Get train sizes (filter out empty dataframes that have no columns)
    train_sizes = sorted({
        int(ts)
        for df in [df_topo, df_worst]
        if not df.empty and "train_size" in df.columns
        for ts in df["train_size"].unique()
        if not pd.isna(ts)
    })
    if not train_sizes:
        print("[INFO] No train sizes detected for Vanilla ordering combined plot.")
        return

    offsets = _build_offsets(train_sizes)
    marker_lookup = _build_marker_map(train_sizes)

    # Use fixed duo layout for all combined plots
    height = COMBINED_FIGSIZE_DUO[1]
    total_width = COMBINED_FIGSIZE_DUO[0]
    panel_width = total_width / 2

    # Check if metric should be exported to paper folder
    if metric not in FOREST_PAPER_METRICS:
        return

    # Prepare output paths - save directly to paper folder
    metric_cfg = METRIC_CONFIG[metric]
    metric_slug = metric_cfg["slug"]
    metric_title = metric_cfg["title"]
    # Normalize metric title for consistent presentation
    metric_title = _normalize_metric_title(metric, metric_title)
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

    file_stem = f"forest_combined_vanilla_ordering_{metric_slug}"
    pdf_path = pdf_dir / f"{file_stem}.pdf"
    csv_path = csv_dir / f"{file_stem}.csv" if csv_dir is not None else None

    figsize = (panel_width * 2, height)
    direction = METRIC_DIRECTIONS.get(metric, "lower")

    def _callback(plt_mod: Any) -> None:
        setup_forest_plot_style()
        # Force font sizes via rcParams to override fastplot defaults
        plt_mod.rcParams['font.size'] = LABEL_FONT_SIZE_COMBINED
        plt_mod.rcParams['axes.labelsize'] = LABEL_FONT_SIZE_COMBINED
        plt_mod.rcParams['axes.titlesize'] = TITLE_FONT_SIZE_SUBPLOT_COMBINED
        plt_mod.rcParams['xtick.labelsize'] = TICK_FONT_SIZE_COMBINED
        plt_mod.rcParams['ytick.labelsize'] = TICK_FONT_SIZE_COMBINED
        plt_mod.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE_COMBINED
        plt_mod.rcParams['legend.title_fontsize'] = LEGEND_TITLE_FONT_SIZE_COMBINED

        fig = plt_mod.gcf()
        fig.clear()
        axes = fig.subplots(1, 2, sharey=True)

        # Custom color palette: Dark purple -> Blue -> Green -> Orange (replaces yellow) -> Red
        # Color-blind friendly palette that avoids yellow for better visibility
        # Based on viridis progression but with orange instead of yellow for Train 200
        # Custom color palette: Dark purple -> Blue -> Green -> Orange (replaces yellow) -> Red -> Brown
        # Color-blind friendly palette that avoids yellow for better visibility
        # Based on viridis progression but with orange instead of yellow for Train 200
        # Added Brown (#8c564b) for Train 1000
        custom_colors = ['#440154', '#31688e', '#35b779', '#f68f46', '#d73027', '#8c564b']  # Dark purple, Blue, Green, Orange, Red, Brown
        color_lookup = {
            ts: custom_colors[i % len(custom_colors)]
            for i, ts in enumerate(train_sizes)
        }

        combined_legend_entries: Dict[int, Line2D] = {}

        # Subplot 1: Vanilla Original vs Vanilla Topological
        ax1 = axes[0]
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        if not df_topo.empty:
            entries = render_forest_panel(
                ax1,
                df_topo,
                datasets,
                dataset_labels,
                train_sizes,
                offsets,
                marker_lookup,
                color_lookup,
                is_combined=True,
            )
            combined_legend_entries.update(entries)

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
                labelpad=_compute_left_adjustments(
                    [dataset_labels[name] for name in datasets],
                    Y_TICK_LABEL_OFFSET_COMBINED,
                    X_LABEL_PAD + Y_LABEL_EXTRA_PAD_COMBINED,
                    combined=True,
                )[1],
            )
            
            # Set tight y-axis limits
            n_datasets = len(datasets)
            ax1.set_ylim(0.5 * DATASET_SPACING, (n_datasets + 0.5) * DATASET_SPACING)
            
            # Explicitly set tick label sizes to ensure consistency across all plots
            y_positions = [(len(datasets) - i) * DATASET_SPACING for i in range(len(datasets))]
            labels = [dataset_labels[name] for name in datasets]
            tick_offset_combined, _ = _compute_left_adjustments(
                labels,
                Y_TICK_LABEL_OFFSET_COMBINED,
                X_LABEL_PAD + Y_LABEL_EXTRA_PAD_COMBINED,
                combined=True,
            )
            _draw_left_aligned_yticklabels(
                ax1,
                y_positions,
                labels,
                font_size=TICK_FONT_SIZE_COMBINED,
                x_offset_points=tick_offset_combined,
            )
            ax1.tick_params(axis="y", labelsize=TICK_FONT_SIZE_COMBINED)
            ax1.tick_params(axis="x", labelsize=TICK_FONT_SIZE_COMBINED, pad=TICK_PAD_X_COMBINED)
            # Force ticks outward for consistency - applied to both axes, all directions
            ax1.tick_params(axis="both", which="both", direction="out", left=True, right=False, bottom=True, top=False)

        # Subplot 2: Vanilla Original vs Vanilla Reverse Topological
        ax2 = axes[1]
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        if not df_worst.empty:
            entries = render_forest_panel(
                ax2,
                df_worst,
                datasets,
                dataset_labels,
                train_sizes,
                offsets,
                marker_lookup,
                color_lookup,
                is_combined=True,
            )
            combined_legend_entries.update(entries)

            comparison_worst = COMPARISON_SPECS["ordering_effects_vanilla_reverse_topological"]
            baseline_label = comparison_worst.baseline_label
            comparator_label = comparison_worst.comparator_label

            ax2.set_title(
                f"{baseline_label} vs {comparator_label}",
                fontsize=TITLE_FONT_SIZE_SUBPLOT_COMBINED,
                pad=COMBINED_TITLE_PAD,
                fontweight='normal',
            )
            ax2.set_ylabel("")
            
            # Set tight y-axis limits
            n_datasets = len(datasets)
            ax2.set_ylim(0.5 * DATASET_SPACING, (n_datasets + 0.5) * DATASET_SPACING)
            
            # Hide default y-axis tick labels for right subplot
            y_positions = [(len(datasets) - i) * DATASET_SPACING for i in range(len(datasets))]
            ax2.set_yticks(y_positions)
            ax2.set_yticklabels([])
            ax2.tick_params(axis="x", labelsize=TICK_FONT_SIZE_COMBINED, pad=TICK_PAD_X_COMBINED)
            ax2.tick_params(axis="y", labelleft=False)
            # Force ticks outward for consistency - applied to both axes, all directions
            ax2.tick_params(axis="both", which="both", direction="out", left=True, right=False, bottom=True, top=False)
            # Ensure labels are hidden even if render_forest_panel set them
            for label in ax2.get_yticklabels():
                label.set_visible(False)

        # Calculate shared x-axis limits from both dataframes for consistent visual scale
        shared_xlim = None
        dfs_for_shared = [df for df in [df_topo, df_worst] if not df.empty]
        if len(dfs_for_shared) > 0:
            # Get initial range to determine if we should use shared limits
            initial_range = abs(ax1.get_xlim()[1] - ax1.get_xlim()[0]) if not df_topo.empty else abs(ax2.get_xlim()[1] - ax2.get_xlim()[0])
            if 0.1 <= initial_range < 1.0:
                shared_xlim = _calculate_shared_xlim_from_dataframes(dfs_for_shared, step=0.1)
        
        # Apply shared limits and tick locator to both panels
        if shared_xlim is not None:
            apply_xaxis_tick_locator(ax1, shared_xlim=shared_xlim)
            apply_xaxis_tick_locator(ax2, shared_xlim=shared_xlim)
        else:
            # Fallback to individual limits if not in medium range
            if not df_topo.empty:
                apply_xaxis_tick_locator(ax1, df=df_topo)
            if not df_worst.empty:
                apply_xaxis_tick_locator(ax2, df=df_worst)

        # Shared x-axis label (baseline shared across both comparisons)
        shared_baseline = COMPARISON_SPECS["ordering_effects_vanilla_topological"].baseline_label
        if direction == "lower":
            shared_xlabel = f"Hodges–Lehmann diff ({shared_baseline} − comparator)"
        else:
            shared_xlabel = f"Hodges–Lehmann diff (comparator − {shared_baseline})"

        # Legend at bottom
        handles = [combined_legend_entries[ts] for ts in train_sizes if ts in combined_legend_entries]
        labels = [handle.get_label() for handle in handles]

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

        # Keep the same margins used by standard plots.
        left_margin = 0.20
        right_margin = 0.98
        wspace = COMBINED_WSPACE_DUO

        # Maximize horizontal space; use consistent top margin from config
        fig.subplots_adjust(
            left=left_margin,
            right=right_margin,
            top=COMBINED_TOP_MARGIN,
            bottom=bottom_margin,
            wspace=wspace,
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

        # Main title at top (metric name) - ADD AFTER subplots_adjust
        # Escape LaTeX special characters (especially % which is a comment character)
        escaped_metric_title = metric_title.replace('%', r'\%')
        fig.text(
            0.5, COMBINED_SUPTITLE_Y,
            escaped_metric_title,
            ha='center',
            va='top',
            fontsize=TITLE_FONT_SIZE_SUPTITLE_COMBINED,
            fontweight='normal',
            transform=fig.transFigure,
        )

        original_tight_layout = plt_mod.tight_layout
        def _no_tight_layout(*args: Any, **kwargs: Any) -> None:
            fig.subplots_adjust(
                left=left_margin,
                right=0.98,
                top=COMBINED_TOP_MARGIN,
                bottom=bottom_margin,
                wspace=COMBINED_WSPACE_DUO,
            )
            plt_mod.tight_layout = original_tight_layout
        plt_mod.tight_layout = _no_tight_layout

    # Save using fastplot
    save_with_fastplot(_callback, pdf_path, figsize, pad_inches=0.05)

    # Save combined CSV with only essential columns
    essential_columns = ["dataset", "train_size", "effect", "ci_lower", "ci_upper", "p_value", "p_holm", "comparison", "holm_significant_stepdown"]
    combined_rows = []
    for data, comparison_name in [(vanilla_orig_vs_topo, "vanilla_original_vs_topological"),
                                 (vanilla_orig_vs_worst, "vanilla_original_vs_worst")]:
        if data:
            for row in data:
                row_dict = dict(row.__dict__)
                row_dict["comparison"] = comparison_name
                # Keep only essential columns
                filtered_dict = {k: v for k, v in row_dict.items() if k in essential_columns}
                combined_rows.append(filtered_dict)

    if csv_path is not None:
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

            if metric == "correlation_matrix_difference":
                combined_df["metric"] = "correlation_matrix_difference"

            if "holm_significant_stepdown" in combined_df.columns:
                combined_df = combined_df.drop(columns=["holm_significant_stepdown"])

            combined_df.to_csv(csv_path, index=False)

    print(
        f"[SUCCESS] Saved Vanilla ordering combined forest plot for metric "
        f"{metric} in {pdf_path}"
    )


def plot_vanilla_topo_dag_combined_forest(
    metric: str,
    show_caption: bool = False,
    no_csv: bool = False,
) -> None:
    """Combined plot: Vanilla Original vs (Vanilla Topological, DAG)."""
    if metric not in METRIC_CONFIG:
        print(f"[WARN] Unknown metric '{metric}' for Vanilla+Dag combined forest plot.")
        return

    # Data for the two comparisons
    vanilla_orig_vs_topo = _collect_forest_rows(metric, COMPARISON_SPECS["ordering_effects_vanilla_topological"])
    vanilla_orig_vs_dag_topo = _collect_forest_rows(metric, COMPARISON_SPECS["cross_dag_topological_vs_vanilla_original"])

    if not vanilla_orig_vs_topo and not vanilla_orig_vs_dag_topo:
        print("[INFO] No data available for Vanilla vs (Topo,DAG-Topo) combined plot.")
        return

    # Build dataframes
    df_topo = build_forest_dataframe(vanilla_orig_vs_topo)
    df_dag = build_forest_dataframe(vanilla_orig_vs_dag_topo)

    # Filter out simglucose datasets to match other vanilla combined plots
    if not df_topo.empty:
        df_topo = df_topo[~df_topo["dataset"].str.startswith("simglucose", na=False)].copy()
    if not df_dag.empty:
        df_dag = df_dag[~df_dag["dataset"].str.startswith("simglucose", na=False)].copy()

    # Get all datasets
    all_datasets = set()
    for df in [df_topo, df_dag]:
        if not df.empty:
            all_datasets.update(df["dataset"].unique())

    datasets = sorted(all_datasets)
    if not datasets:
        print("[INFO] No datasets available for Vanilla vs (Topo,DAG-Topo) combined plot.")
        return

    # Build dataset labels
    dataset_labels: Dict[str, str] = {}
    for df in [df_topo, df_dag]:
        if not df.empty:
            df["dataset_label"] = df["dataset"].apply(_format_display_label)
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
        for df in [df_topo, df_dag]
        for ts in df["train_size"].unique()
        if not pd.isna(ts)
    })
    if not train_sizes:
        print("[INFO] No train sizes detected for Vanilla vs (Topo,DAG-Topo) combined plot.")
        return

    offsets = _build_offsets(train_sizes)
    marker_lookup = _build_marker_map(train_sizes)

    # Use fixed duo layout for all combined plots
    height = COMBINED_FIGSIZE_DUO[1]
    total_width = COMBINED_FIGSIZE_DUO[0]
    panel_width = total_width / 2

    # Check if metric should be exported to paper folder
    if metric not in FOREST_PAPER_METRICS:
        return

    metric_cfg = METRIC_CONFIG[metric]
    metric_slug = metric_cfg["slug"]
    metric_title = _normalize_metric_title(metric, metric_cfg["title"])
    # Save directly to paper folder
    paper_dir = PAPER_ROOT / "vanilla_topo_dag_topo_combined"
    _ensure_dir(paper_dir)
    if no_csv:
        pdf_dir = paper_dir
        csv_dir = None
    else:
        pdf_dir = paper_dir / "pdf"
        csv_dir = paper_dir / "csv"
        _ensure_dir(pdf_dir)
        _ensure_dir(csv_dir)

    file_stem = f"forest_combined_vanilla_topo_and_dag_topo_{metric_slug}"
    pdf_path = pdf_dir / f"{file_stem}.pdf"
    csv_path = csv_dir / f"{file_stem}.csv" if csv_dir is not None else None

    figsize = (panel_width * 2, height)
    direction = METRIC_DIRECTIONS.get(metric, "lower")
    
    # Prepare shared xlabel with explicit LaTeX command to prevent bold
    shared_baseline = COMPARISON_SPECS["ordering_effects_vanilla_topological"].baseline_label
    if direction == "lower":
        shared_xlabel = r"\textmd{Hodges–Lehmann diff} (" + f"{shared_baseline} − comparator)"
    else:
        shared_xlabel = r"\textmd{Hodges–Lehmann diff} (" + f"comparator − {shared_baseline})"

    def _callback(plt_mod: Any) -> None:
        setup_forest_plot_style()
        # Force font sizes via rcParams to override fastplot defaults
        plt_mod.rcParams['font.size'] = LABEL_FONT_SIZE_COMBINED
        plt_mod.rcParams['axes.labelsize'] = LABEL_FONT_SIZE_COMBINED
        plt_mod.rcParams['axes.titlesize'] = TITLE_FONT_SIZE_SUBPLOT_COMBINED
        plt_mod.rcParams['xtick.labelsize'] = TICK_FONT_SIZE_COMBINED
        plt_mod.rcParams['ytick.labelsize'] = TICK_FONT_SIZE_COMBINED
        plt_mod.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE_COMBINED
        plt_mod.rcParams['legend.title_fontsize'] = LEGEND_TITLE_FONT_SIZE_COMBINED

        fig = plt_mod.gcf()
        fig.clear()
        axes = fig.subplots(1, 2, sharey=True)

        # Custom color palette: Dark purple -> Blue -> Green -> Orange (replaces yellow) -> Red
        # Color-blind friendly palette that avoids yellow for better visibility
        # Based on viridis progression but with orange instead of yellow for Train 200
        # Custom color palette: Dark purple -> Blue -> Green -> Orange (replaces yellow) -> Red -> Brown
        # Color-blind friendly palette that avoids yellow for better visibility
        # Based on viridis progression but with orange instead of yellow for Train 200
        # Added Brown (#8c564b) for Train 1000
        custom_colors = ['#440154', '#31688e', '#35b779', '#f68f46', '#d73027', '#8c564b']  # Dark purple, Blue, Green, Orange, Red, Brown
        color_lookup = {
            ts: custom_colors[i % len(custom_colors)]
            for i, ts in enumerate(train_sizes)
        }

        combined_legend_entries: Dict[int, Line2D] = {}

        # Left: Vanilla Original vs Vanilla Topological
        ax1 = axes[0]
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        if not df_topo.empty:
            entries = render_forest_panel(
                ax1,
                df_topo,
                datasets,
                dataset_labels,
                train_sizes,
                offsets,
                marker_lookup,
                color_lookup,
                is_combined=True,
            )
            combined_legend_entries.update(entries)

            comparison_topo = COMPARISON_SPECS["ordering_effects_vanilla_topological"]
            baseline_label = comparison_topo.baseline_label
            comparator_label = comparison_topo.comparator_label
            ax1.set_title(
                f"{baseline_label} vs {comparator_label}",
                fontsize=TITLE_FONT_SIZE_SUBPLOT_COMBINED,
                pad=COMBINED_TITLE_PAD,
                fontweight='normal',
            )
            ax1.set_ylabel(
                "Dataset",
                fontsize=LABEL_FONT_SIZE_COMBINED,
                labelpad=_compute_left_adjustments(
                    [dataset_labels[name] for name in datasets],
                    Y_TICK_LABEL_OFFSET_COMBINED,
                    X_LABEL_PAD + Y_LABEL_EXTRA_PAD_COMBINED,
                    combined=True,
                )[1],
            )
            
            # Set tight y-axis limits
            n_datasets = len(datasets)
            ax1.set_ylim(0.5 * DATASET_SPACING, (n_datasets + 0.5) * DATASET_SPACING)
            
            # Explicitly set tick label sizes to ensure consistency across all plots
            y_positions = [(len(datasets) - i) * DATASET_SPACING for i in range(len(datasets))]
            labels = [dataset_labels[name] for name in datasets]
            tick_offset_combined, _ = _compute_left_adjustments(
                labels,
                Y_TICK_LABEL_OFFSET_COMBINED,
                X_LABEL_PAD + Y_LABEL_EXTRA_PAD_COMBINED,
                combined=True,
            )
            _draw_left_aligned_yticklabels(
                ax1,
                y_positions,
                labels,
                font_size=TICK_FONT_SIZE_COMBINED,
                x_offset_points=tick_offset_combined,
            )
            ax1.tick_params(axis="y", labelsize=TICK_FONT_SIZE_COMBINED)
            ax1.tick_params(axis="x", labelsize=TICK_FONT_SIZE_COMBINED, pad=TICK_PAD_X_COMBINED)
            # Force ticks outward for consistency - applied to both axes, all directions
            ax1.tick_params(axis="both", which="both", direction="out", left=True, right=False, bottom=True, top=False)

        # Right: Vanilla Original vs DAG
        ax2 = axes[1]
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        if not df_dag.empty:
            entries = render_forest_panel(
                ax2,
                df_dag,
                datasets,
                dataset_labels,
                train_sizes,
                offsets,
                marker_lookup,
                color_lookup,
                is_combined=True,
            )
            combined_legend_entries.update(entries)

            comparison_dag = COMPARISON_SPECS["cross_dag_topological_vs_vanilla_original"]
            baseline_label = comparison_dag.baseline_label
            comparator_label = comparison_dag.comparator_label
            ax2.set_title(
                f"{baseline_label} vs {comparator_label}",
                fontsize=TITLE_FONT_SIZE_SUBPLOT_COMBINED,
                pad=COMBINED_TITLE_PAD,
                fontweight='normal',
            )
            ax2.set_ylabel("")
            
            # Set tight y-axis limits
            n_datasets = len(datasets)
            ax2.set_ylim(0.5 * DATASET_SPACING, (n_datasets + 0.5) * DATASET_SPACING)
            
            # Hide default y-axis tick labels for right subplot
            y_positions = [(len(datasets) - i) * DATASET_SPACING for i in range(len(datasets))]
            ax2.set_yticks(y_positions)
            ax2.set_yticklabels([])
            ax2.tick_params(axis="x", labelsize=TICK_FONT_SIZE_COMBINED, pad=TICK_PAD_X_COMBINED)
            ax2.tick_params(axis="y", labelleft=False)
            # Force ticks outward for consistency - applied to both axes, all directions
            ax2.tick_params(axis="both", which="both", direction="out", left=True, right=False, bottom=True, top=False)
            # Ensure labels are hidden even if render_forest_panel set them
            for label in ax2.get_yticklabels():
                label.set_visible(False)

        # Calculate shared x-axis limits from both dataframes for consistent visual scale
        shared_xlim = None
        dfs_for_shared = [df for df in [df_topo, df_dag] if not df.empty]
        if len(dfs_for_shared) > 0:
            # Get initial range to determine if we should use shared limits
            initial_range = abs(ax1.get_xlim()[1] - ax1.get_xlim()[0]) if not df_topo.empty else abs(ax2.get_xlim()[1] - ax2.get_xlim()[0])
            if 0.1 <= initial_range < 1.0:
                shared_xlim = _calculate_shared_xlim_from_dataframes(dfs_for_shared, step=0.1)
        
        # Apply shared limits and tick locator to both panels
        if shared_xlim is not None:
            apply_xaxis_tick_locator(ax1, shared_xlim=shared_xlim)
            apply_xaxis_tick_locator(ax2, shared_xlim=shared_xlim)
        else:
            # Fallback to individual limits if not in medium range
            if not df_topo.empty:
                apply_xaxis_tick_locator(ax1, df=df_topo)
            if not df_dag.empty:
                apply_xaxis_tick_locator(ax2, df=df_dag)
        
        # Add extra margins for right panel (DAG) to improve spacing
        # Increase left margin (distance from Y-axis to zero line) and right margin
        if not df_dag.empty:
            xlim_dag = ax2.get_xlim()
            x_range_dag = abs(xlim_dag[1] - xlim_dag[0])
            # Add 5% margin on left and right for better spacing
            left_margin = x_range_dag * 0.05
            right_margin = x_range_dag * 0.05
            ax2.set_xlim(xlim_dag[0] - left_margin, xlim_dag[1] + right_margin)
            
            # Reapply locator and formatter after adding margins to ensure correct ticks
            # Get the current locator and formatter from ax2
            current_locator = ax2.xaxis.get_major_locator()
            current_formatter = ax2.xaxis.get_major_formatter()
            # Reapply them to ensure ticks are correct with new limits
            ax2.xaxis.set_major_locator(current_locator)
            ax2.xaxis.set_major_formatter(current_formatter)

        # Legend at bottom
        handles = [combined_legend_entries[ts] for ts in train_sizes if ts in combined_legend_entries]
        labels = [handle.get_label() for handle in handles]

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

        # Keep the same margins used by standard plots.
        left_margin = 0.20
        right_margin = 0.98
        wspace = COMBINED_WSPACE_DUO

        # Maximize horizontal space; use consistent top margin from config
        fig.subplots_adjust(
            left=left_margin,
            right=right_margin,
            top=COMBINED_TOP_MARGIN,
            bottom=bottom_margin,
            wspace=wspace,
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

        # Main title at top (metric name) - ADD AFTER subplots_adjust
        # Escape LaTeX special characters (especially % which is a comment character)
        escaped_metric_title = metric_title.replace('%', r'\%')
        fig.text(
            0.5, COMBINED_SUPTITLE_Y,
            escaped_metric_title,
            ha='center',
            va='top',
            fontsize=TITLE_FONT_SIZE_SUPTITLE_COMBINED,
            fontweight='normal',
            transform=fig.transFigure,
        )

        original_tight_layout = plt_mod.tight_layout
        def _no_tight_layout(*args: Any, **kwargs: Any) -> None:
            fig.subplots_adjust(
                left=left_margin,
                right=0.98,
                top=COMBINED_TOP_MARGIN,
                bottom=bottom_margin,
                wspace=COMBINED_WSPACE_DUO,
            )
            plt_mod.tight_layout = original_tight_layout
        plt_mod.tight_layout = _no_tight_layout

    # Save using fastplot
    save_with_fastplot(_callback, pdf_path, figsize, pad_inches=0.05)

    # Save combined CSV with only essential columns
    essential_columns = ["dataset", "train_size", "effect", "ci_lower", "ci_upper", "p_value", "p_holm", "comparison", "holm_significant_stepdown"]
    combined_rows = []
    for data, comparison_name in [
        (vanilla_orig_vs_topo, "vanilla_original_vs_topological"),
        (vanilla_orig_vs_dag_topo, "cross_dag_topological_vs_vanilla_original"),
    ]:
        if data:
            for row in data:
                row_dict = dict(row.__dict__)
                row_dict["comparison"] = comparison_name
                # Keep only essential columns
                filtered_dict = {k: v for k, v in row_dict.items() if k in essential_columns}
                combined_rows.append(filtered_dict)

    if csv_path is not None:
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

            if metric == "correlation_matrix_difference":
                combined_df["metric"] = "correlation_matrix_difference"

            if "holm_significant_stepdown" in combined_df.columns:
                combined_df = combined_df.drop(columns=["holm_significant_stepdown"])

            combined_df.to_csv(csv_path, index=False)

    print(
        f"[SUCCESS] Saved Vanilla vs (Topo,DAG-Topo) combined forest plot for metric "
        f"{metric} in {pdf_path}"
    )


def plot_vanilla_topo_simglucose_combined_forest(
    metric: str,
    show_caption: bool = False,
    no_csv: bool = False,
) -> None:
    """Combined plot: Vanilla Original vs Vanilla Topological (left: other datasets, right: simglucose only)."""
    if metric not in METRIC_CONFIG:
        print(f"[WARN] Unknown metric '{metric}' for Vanilla+Simglucose combined forest plot.")
        return

    # Data for the comparison
    vanilla_orig_vs_topo = _collect_forest_rows(metric, COMPARISON_SPECS["ordering_effects_vanilla_topological"])

    if not vanilla_orig_vs_topo:
        print("[INFO] No data available for Vanilla vs Topological combined plot.")
        return

    # Build dataframe
    df_all = build_forest_dataframe(vanilla_orig_vs_topo)

    if df_all.empty:
        print("[INFO] No data available for Vanilla vs Topological combined plot.")
        return

    # Separate datasets: other datasets (left) and simglucose (right)
    df_left = df_all[~df_all["dataset"].str.startswith("simglucose", na=False)].copy()
    df_right = df_all[df_all["dataset"].str.startswith("simglucose", na=False)].copy()

    # Get datasets for each panel
    datasets_left = sorted(df_left["dataset"].unique()) if not df_left.empty else []
    datasets_right = sorted(df_right["dataset"].unique()) if not df_right.empty else []

    if not datasets_left or not datasets_right:
        print("[INFO] Missing data for one or both panels in Vanilla+Simglucose combined plot.")
        return

    # Build dataset labels
    dataset_labels_left: Dict[str, str] = {}
    if not df_left.empty:
        df_left["dataset_label"] = df_left["dataset"].apply(_format_display_label)
        dataset_labels_left = {
            name: _abbreviate_dataset_label_for_combined_plots(
                name, df_left.loc[df_left["dataset"] == name, "dataset_label"].iloc[0]
            )
            for name in datasets_left
        }

    dataset_labels_right: Dict[str, str] = {}
    if not df_right.empty:
        df_right["dataset_label"] = df_right["dataset"].apply(_format_display_label)
        dataset_labels_right = {
            name: _abbreviate_dataset_label_for_combined_plots(
                name, df_right.loc[df_right["dataset"] == name, "dataset_label"].iloc[0]
            )
            for name in datasets_right
        }

    # Get train sizes
    train_sizes = sorted({
        int(ts)
        for ts in df_all["train_size"].unique()
        if not pd.isna(ts)
    })
    if not train_sizes:
        print("[INFO] No train sizes detected for Vanilla+Simglucose combined plot.")
        return

    offsets = _build_offsets(train_sizes)
    marker_lookup = _build_marker_map(train_sizes)

    # Use fixed duo layout for all combined plots
    height = COMBINED_FIGSIZE_DUO[1]
    total_width = COMBINED_FIGSIZE_DUO[0]
    panel_width = total_width / 2

    # Check if metric should be exported to paper folder
    if metric not in FOREST_PAPER_METRICS:
        return

    metric_cfg = METRIC_CONFIG[metric]
    metric_slug = metric_cfg["slug"]
    metric_title = _normalize_metric_title(metric, metric_cfg["title"])
    # Save directly to paper folder
    paper_dir = PAPER_ROOT / "vanilla_topo_simglucose_combined"
    _ensure_dir(paper_dir)
    if no_csv:
        pdf_dir = paper_dir
        csv_dir = None
    else:
        pdf_dir = paper_dir / "pdf"
        csv_dir = paper_dir / "csv"
        _ensure_dir(pdf_dir)
        _ensure_dir(csv_dir)

    file_stem = f"forest_combined_vanilla_topo_and_simglucose_{metric_slug}"
    pdf_path = pdf_dir / f"{file_stem}.pdf"
    csv_path = csv_dir / f"{file_stem}.csv" if csv_dir is not None else None

    figsize = (panel_width * 2, height)
    direction = METRIC_DIRECTIONS.get(metric, "lower")
    
    # Prepare shared xlabel
    comparison = COMPARISON_SPECS["ordering_effects_vanilla_topological"]
    shared_baseline = comparison.baseline_label
    comparator_label = comparison.comparator_label  # "Vanilla Topological"
    if direction == "lower":
        shared_xlabel = r"\textmd{Hodges–Lehmann diff} (" + f"{shared_baseline} − {comparator_label})"
    else:
        shared_xlabel = r"\textmd{Hodges–Lehmann diff} (" + f"{comparator_label} − {shared_baseline})"

    def _callback(plt_mod: Any) -> None:
        setup_forest_plot_style()
        # Force font sizes via rcParams to override fastplot defaults
        plt_mod.rcParams['font.size'] = LABEL_FONT_SIZE_COMBINED
        plt_mod.rcParams['axes.labelsize'] = LABEL_FONT_SIZE_COMBINED
        plt_mod.rcParams['axes.titlesize'] = TITLE_FONT_SIZE_SUBPLOT_COMBINED
        plt_mod.rcParams['xtick.labelsize'] = TICK_FONT_SIZE_COMBINED
        plt_mod.rcParams['ytick.labelsize'] = TICK_FONT_SIZE_COMBINED
        plt_mod.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE_COMBINED
        plt_mod.rcParams['legend.title_fontsize'] = LEGEND_TITLE_FONT_SIZE_COMBINED

        fig = plt_mod.gcf()
        fig.clear()
        # Do NOT share y-axis because datasets are different
        axes = fig.subplots(1, 2, sharey=False)

        # Custom color palette
        # Custom color palette: Dark purple -> Blue -> Green -> Orange (replaces yellow) -> Red -> Brown
        # Color-blind friendly palette that avoids yellow for better visibility
        # Based on viridis progression but with orange instead of yellow for Train 200
        # Added Brown (#8c564b) for Train 1000
        custom_colors = ['#440154', '#31688e', '#35b779', '#f68f46', '#d73027', '#8c564b']
        color_lookup = {
            ts: custom_colors[i % len(custom_colors)]
            for i, ts in enumerate(train_sizes)
        }

        combined_legend_entries: Dict[int, Line2D] = {}

        # Left: Vanilla Original vs Vanilla Topological (other datasets)
        ax1 = axes[0]
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        if not df_left.empty:
            entries = render_forest_panel(
                ax1,
                df_left,
                datasets_left,
                dataset_labels_left,
                train_sizes,
                offsets,
                marker_lookup,
                color_lookup,
                is_combined=True,
            )
            combined_legend_entries.update(entries)

            # No individual title for left panel - we'll use a shared title above
            ax1.set_ylabel(
                "Dataset",
                fontsize=LABEL_FONT_SIZE_COMBINED,
                labelpad=_compute_left_adjustments(
                    [dataset_labels_left[name] for name in datasets_left],
                    Y_TICK_LABEL_OFFSET_COMBINED,
                    X_LABEL_PAD + Y_LABEL_EXTRA_PAD_COMBINED,
                    combined=True,
                )[1],
            )
            
            # Set tight y-axis limits for left panel
            n_datasets_left = len(datasets_left)
            ax1.set_ylim(0.5 * DATASET_SPACING, (n_datasets_left + 0.5) * DATASET_SPACING)
            
            # Set tick labels for left panel
            y_positions_left = [(len(datasets_left) - i) * DATASET_SPACING for i in range(len(datasets_left))]
            labels_left = [dataset_labels_left[name] for name in datasets_left]
            tick_offset_combined, _ = _compute_left_adjustments(
                labels_left,
                Y_TICK_LABEL_OFFSET_COMBINED,
                X_LABEL_PAD + Y_LABEL_EXTRA_PAD_COMBINED,
                combined=True,
            )
            _draw_left_aligned_yticklabels(
                ax1,
                y_positions_left,
                labels_left,
                font_size=TICK_FONT_SIZE_COMBINED,
                x_offset_points=tick_offset_combined,
            )
            ax1.tick_params(axis="y", labelsize=TICK_FONT_SIZE_COMBINED)
            ax1.tick_params(axis="x", labelsize=TICK_FONT_SIZE_COMBINED, pad=TICK_PAD_X_COMBINED)
            ax1.tick_params(axis="both", which="both", direction="out", left=True, right=False, bottom=True, top=False)

            # Apply x-axis tick locator for left panel
            apply_xaxis_tick_locator(ax1, df=df_left)

        # Right: Vanilla Original vs Vanilla Topological (simglucose only)
        ax2 = axes[1]
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        if not df_right.empty:
            entries = render_forest_panel(
                ax2,
                df_right,
                datasets_right,
                dataset_labels_right,
                train_sizes,
                offsets,
                marker_lookup,
                color_lookup,
                is_combined=True,
            )
            combined_legend_entries.update(entries)

            # No individual title for right panel - we'll use a shared title above
            ax2.set_ylabel("")
            
            # Set tight y-axis limits for right panel (simglucose)
            n_datasets_right = len(datasets_right)
            ax2.set_ylim(0.5 * DATASET_SPACING, (n_datasets_right + 0.5) * DATASET_SPACING)
            
            # Set tick labels for right panel (simglucose)
            y_positions_right = [(len(datasets_right) - i) * DATASET_SPACING for i in range(len(datasets_right))]
            labels_right = [dataset_labels_right[name] for name in datasets_right]
            tick_offset_combined_right, _ = _compute_left_adjustments(
                labels_right,
                Y_TICK_LABEL_OFFSET_COMBINED,
                X_LABEL_PAD + Y_LABEL_EXTRA_PAD_COMBINED,
                combined=True,
            )
            _draw_left_aligned_yticklabels(
                ax2,
                y_positions_right,
                labels_right,
                font_size=TICK_FONT_SIZE_COMBINED,
                x_offset_points=tick_offset_combined_right,
            )
            ax2.tick_params(axis="y", labelsize=TICK_FONT_SIZE_COMBINED)
            ax2.tick_params(axis="x", labelsize=TICK_FONT_SIZE_COMBINED, pad=TICK_PAD_X_COMBINED)
            ax2.tick_params(axis="both", which="both", direction="out", left=True, right=False, bottom=True, top=False)

            # Apply simglucose-specific x-axis settings: start from -10, step 10
            apply_xaxis_tick_locator(ax2, df=df_right)
            current_xlim = ax2.get_xlim()
            ax2.set_xlim(left=-10, right=current_xlim[1])
            ax2.xaxis.set_major_locator(MultipleLocator(10))

        # Legend at bottom
        handles = [combined_legend_entries[ts] for ts in train_sizes if ts in combined_legend_entries]
        labels = [handle.get_label() for handle in handles]

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

        left_margin = 0.20

        # Maximize horizontal space; use consistent top margin from config
        fig.subplots_adjust(
            left=left_margin,
            right=0.98,
            top=COMBINED_TOP_MARGIN,
            bottom=bottom_margin,
            wspace=COMBINED_WSPACE_DUO,
        )

        # Align x-axis tick labels vertically across all panels
        fig.align_xlabels(axes)
        fig.align_ylabels(axes)

        # Place shared x-axis label AFTER subplots_adjust
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

        # Main title at top (comparison and metric name)
        comparison_topo = COMPARISON_SPECS["ordering_effects_vanilla_topological"]
        # Escape LaTeX special characters (especially % which is a comment character)
        escaped_metric_title = metric_title.replace('%', r'\%')
        main_title = f"{comparison_topo.title} · {escaped_metric_title}"
        fig.text(
            0.5, COMBINED_SUPTITLE_Y,
            main_title,
            ha='center',
            va='top',
            fontsize=TITLE_FONT_SIZE_SUPTITLE_COMBINED,
            fontweight='normal',
            transform=fig.transFigure,
        )

        original_tight_layout = plt_mod.tight_layout
        def _no_tight_layout(*args: Any, **kwargs: Any) -> None:
            fig.subplots_adjust(
                left=left_margin,
                right=0.98,
                top=COMBINED_TOP_MARGIN,
                bottom=bottom_margin,
                wspace=COMBINED_WSPACE_DUO,
            )
            plt_mod.tight_layout = original_tight_layout
        plt_mod.tight_layout = _no_tight_layout

    # Save using fastplot
    save_with_fastplot(_callback, pdf_path, figsize, pad_inches=0.05)

    # Save combined CSV with only essential columns
    essential_columns = ["dataset", "train_size", "effect", "ci_lower", "ci_upper", "p_value", "p_holm", "comparison", "holm_significant_stepdown"]
    combined_rows = []
    for row in vanilla_orig_vs_topo:
        row_dict = dict(row.__dict__)
        row_dict["comparison"] = "vanilla_original_vs_topological"
        # Keep only essential columns
        filtered_dict = {k: v for k, v in row_dict.items() if k in essential_columns}
        combined_rows.append(filtered_dict)

    if csv_path is not None:
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

            if metric == "correlation_matrix_difference":
                combined_df["metric"] = "correlation_matrix_difference"

            if "holm_significant_stepdown" in combined_df.columns:
                combined_df = combined_df.drop(columns=["holm_significant_stepdown"])

            combined_df.to_csv(csv_path, index=False)

    print(
        f"[SUCCESS] Saved Vanilla vs Topological (with Simglucose) combined forest plot for metric "
        f"{metric} in {pdf_path}"
    )


def main(
    metric_names: Iterable[str],
    comparison_keys: Iterable[str],
    show_caption: bool = False,
    recompute_stats: bool = True,
    no_csv: bool = False,
    result_files: Sequence[str] | None = None,
    dataset_slugs: Sequence[str] | None = None,
    paper_root: Path | None = None,
) -> None:
    global PAPER_ROOT, DATASET_FILTER
    metric_names = tuple(metric_names)
    comparison_keys = tuple(comparison_keys)
    active_paper_root = PAPER_ROOT
    if paper_root is not None:
        PAPER_ROOT = _normalize_noise_layout_fix_root(Path(paper_root))
        active_paper_root = PAPER_ROOT
    if dataset_slugs is not None:
        DATASET_FILTER = {str(slug).strip() for slug in dataset_slugs if str(slug).strip()}
    else:
        DATASET_FILTER = None
    if result_files is None:
        discovered = interventional_stats.discover_result_files()
    else:
        discovered = list(result_files)
    _set_expected_dataset_slugs(discovered)
    # Guardrail: keep the main paper run free from noise-1e-2 datasets unless explicitly requested.
    # Noise-specific runs use paper_noise1e-2 roots or explicit dataset/result-file overrides.
    if DATASET_FILTER is None and "paper_noise1e-2" not in str(active_paper_root):
        filtered_slugs = {
            _dataset_slug_from_filename(Path(name))
            for name in discovered
            if "noise1e-2" not in _dataset_slug_from_filename(Path(name))
        }
        if filtered_slugs:
            DATASET_FILTER = filtered_slugs
    _ensure_dir(STAT_TESTS_ROOT)
    if recompute_stats:
        _ensure_stat_tests_from_csvs(discovered, metric_names)
    _reset_paper_root()

    for metric in metric_names:
        for key in comparison_keys:
            if key == "ordering_effects_vanilla_topological":
                continue
            try:
                comparison = COMPARISON_SPECS[key]
            except KeyError as exc:
                raise SystemExit(f"Unknown comparison '{key}'. Available: {sorted(COMPARISON_SPECS)}") from exc

            rows = _collect_forest_rows(metric, comparison)
            plot_forest(metric, rows, comparison, show_caption=show_caption, no_csv=no_csv)

    # Generate CPDAG combined plots if CPDAG comparisons are requested
    if ("original_cpdag_minimal_vs_vanilla" in comparison_keys or
        "original_cpdag_discovered_vs_vanilla" in comparison_keys):
        for metric in metric_names:
            plot_cpdag_combined_forest(metric, show_caption=show_caption, no_csv=no_csv)

    # Generate DAG+Minimal CPDAG combined plots if both comparisons are requested
    if ("cross_dag_topological_vs_vanilla_original" in comparison_keys and
        "original_cpdag_minimal_vs_vanilla" in comparison_keys):
        for metric in metric_names:
            plot_dag_cpdag_minimal_combined_forest(metric, show_caption=show_caption, no_csv=no_csv)

    # Generate Vanilla ordering combined plots if Vanilla ordering comparisons are requested
    if ("ordering_effects_vanilla_topological" in comparison_keys or
        "ordering_effects_vanilla_reverse_topological" in comparison_keys):
        for metric in metric_names:
            plot_vanilla_ordering_combined_forest(metric, show_caption=show_caption, no_csv=no_csv)

    # Generate Vanilla vs (Topological, DAG) combined plots
    if ("ordering_effects_vanilla_topological" in comparison_keys or
        "cross_dag_topological_vs_vanilla_original" in comparison_keys):
        for metric in metric_names:
            plot_vanilla_topo_dag_combined_forest(metric, show_caption=show_caption, no_csv=no_csv)

    # Generate Vanilla vs Topological (with Simglucose) combined plots
    if "ordering_effects_vanilla_topological" in comparison_keys:
        for metric in metric_names:
            plot_vanilla_topo_simglucose_combined_forest(metric, show_caption=show_caption, no_csv=no_csv)


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
        "--result-files",
        nargs="+",
        default=None,
        help="Override the list of interventional result CSVs to process.",
    )
    parser.add_argument(
        "--dataset-slugs",
        nargs="+",
        default=None,
        help="Restrict plots to the specified dataset slugs.",
    )
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=None,
        help="Override the output folder for paper-ready plots.",
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
    main(
        args.metrics,
        args.comparisons,
        show_caption=args.caption,
        recompute_stats=not args.skip_stats,
        no_csv=args.no_csv,
        result_files=args.result_files,
        dataset_slugs=args.dataset_slugs,
        paper_root=args.paper_root,
    )
