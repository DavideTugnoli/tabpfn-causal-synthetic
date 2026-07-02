
import re
import shutil
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, FuncFormatter, MultipleLocator

# Try to import fastplot, but fallback if not available
try:
    import fastplot
    HAS_FASTPLOT = True
except ImportError:
    HAS_FASTPLOT = False

from causal_experiments.utils.visualization_config import (
    METRIC_CONFIG,
    DPI,
)
from causal_experiments.utils.forest_plot_config import (
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

# Orientation config
METRIC_DIRECTIONS: Dict[str, str] = {
    "correlation_matrix_difference": "lower",
    "k_marginal_tvd": "lower",
    "mi_matrix_difference": "lower",
    "nnaa": "lower",
    "ate_difference": "lower",
    "ate_relative_error": "lower",
}

@dataclass
class ForestRow:
    dataset: str
    train_size: int
    effect: float
    ci_lower: float
    ci_upper: float
    n_pairs: int
    ci_source: str | None = None
    p_value: float | None = None
    p_holm: float | None = None
    statistic: float | None = None
    holm_alpha: float | None = None
    holm_stage_threshold: float | None = None
    holm_significant: bool | None = None
    holm_significant_stepdown: bool | None = None
    ci_lower_uncorrected: float | None = None
    ci_upper_uncorrected: float | None = None
    ci_level_uncorrected: float | None = None
    ci_lower_holm: float | None = None
    ci_upper_holm: float | None = None
    ci_level_holm: float | None = None
    effect_baseline_minus_comparator: float | None = None
    median_baseline: float | None = None
    median_comparator: float | None = None
    median_diff_baseline_minus_comparator: float | None = None
    median_diff_oriented: float | None = None
    mean_diff_baseline_minus_comparator: float | None = None
    mean_diff_oriented: float | None = None
    direction: str | None = None
    metric: str | None = None
    metric_slug: str | None = None
    comparison_id: str | None = None
    comparison_slug: str | None = None
    comparison_title: str | None = None
    baseline_label: str | None = None
    comparator_label: str | None = None
    baseline_condition: str | None = None
    comparator_condition: str | None = None
    group_slug: str | None = None
    group_label: str | None = None


@dataclass(frozen=True)
class NnaaCheckSummary:
    min_value: float
    max_value: float
    high_points: List[str]


@dataclass(frozen=True)
class ComparisonSpec:
    slug: str
    baseline: str
    comparator: str
    baseline_label: str
    comparator_label: str
    title: str
    group_slug: str
    group_label: str

    @property
    def output_slug(self) -> str:
        """Descriptive slug combining baseline and comparator labels."""
        return _comparison_output_slug(self)


def _slugify_label(text: str) -> str:
    """Create a filesystem-friendly slug from a human-readable label."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).lower())
    slug = slug.strip("_")
    return slug or "comparison"


def _comparison_output_slug(comparison: ComparisonSpec) -> str:
    """Return an order-independent slug built from the condition identifiers."""
    condition_slugs = sorted(
        [_slugify_label(comparison.baseline), _slugify_label(comparison.comparator)]
    )
    return f"{condition_slugs[0]}_vs_{condition_slugs[1]}"


def _build_caption_text(metric_title: str, comparison: ComparisonSpec, direction: str) -> str:
    """Construct a concise caption describing the comparison and orientation."""
    orientation_note = (
        f"Positive values indicate {comparison.comparator_label} performs better (metric lower is better)."
        if direction == "lower"
        else f"Positive values indicate {comparison.comparator_label} performs better (metric higher is better)."
    )
    return (
        f"{metric_title} · {comparison.baseline_label} vs {comparison.comparator_label}. "
        f"{orientation_note}"
    )


def _format_display_label(text: str) -> str:
    """Replace underscores with spaces and collapse repeated whitespace for presentation."""
    return " ".join(str(text).replace("_", " ").split())


def _wrap_xlabel(label: str, max_chars: int = 36) -> str:
    """
    Wrap long x-axis labels to avoid clipping/overlap while keeping font sizes intact.
    """
    return "\n".join(textwrap.fill(label, width=max_chars).splitlines())


def _normalize_metric_title(metric: str, metric_title: str) -> str:
    """
    Normalize metric titles for consistent presentation in plots.
    Expands common acronyms to full names for clarity and consistency.
    """
    # Expand MI to Mutual Information
    if "MI Matrix Difference" in metric_title:
        metric_title = metric_title.replace("MI Matrix Difference", "Mutual Information Matrix Difference")
    
    # Expand TVD to Total Variation Distance
    if "TVD" in metric_title:
        metric_title = metric_title.replace("TVD", "Total Variation Distance")
    
    # Expand NNAA to Nearest-Neighbor Adversarial Accuracy
    if metric_title == "NNAA":
        metric_title = "Nearest-Neighbor Adversarial Accuracy"
    
    # Special handling for k-marginal: use consistent format
    if metric == "k_marginal_tvd":
        metric_title = "k-Marginal Total Variation Distance"
    
    return metric_title


def _first_valid_value_with_source(row: pd.Series, columns: Tuple[str, ...]) -> Tuple[float, str]:
    """Return the first non-null value together with its column name."""
    for col in columns:
        if col in row.index and pd.notna(row[col]):
            return float(row[col]), col
    raise ValueError(f"Missing CI columns among {columns}")


def _first_valid_value(row: pd.Series, columns: Tuple[str, ...]) -> float:
    """Return the first non-null value from ``columns`` present in ``row``."""
    value, _ = _first_valid_value_with_source(row, columns)
    return value


def _normalize_effect(
    row: pd.Series,
    comparison: ComparisonSpec,
    ci_lower_cols: Tuple[str, ...],
    ci_upper_cols: Tuple[str, ...],
) -> Tuple[float, float, float, str | None, str | None]:
    """
    Return (baseline - comparator, ci_lower, ci_upper, lower_source, upper_source)
    regardless of original ordering.
    """
    raw_effect = float(row["effect_hl"])
    ci_lower_value, ci_lower_source = _first_valid_value_with_source(row, ci_lower_cols)
    ci_upper_value, ci_upper_source = _first_valid_value_with_source(row, ci_upper_cols)

    cond_a = str(row["condition_a"])
    cond_b = str(row["condition_b"])

    baseline_condition = comparison.baseline
    comparator_condition = comparison.comparator

    if cond_a == baseline_condition and cond_b == comparator_condition:
        baseline_minus_comparator = raw_effect
        lower, upper = ci_lower_value, ci_upper_value
    elif cond_a == comparator_condition and cond_b == baseline_condition:
        # Flip the sign because the stored effect was comparator - baseline
        baseline_minus_comparator = -raw_effect
        lower, upper = -ci_upper_value, -ci_lower_value
    else:
        raise ValueError(
            f"Unexpected condition pair ({cond_a}, {cond_b}); expected {comparison.baseline_label} "
            f"vs {comparison.comparator_label}."
        )

    if lower > upper:
        lower, upper = upper, lower
        ci_lower_source, ci_upper_source = ci_upper_source, ci_lower_source

    return baseline_minus_comparator, lower, upper, ci_lower_source, ci_upper_source


def _orient_scalar(value: float, direction: str) -> float:
    """Orient a scalar so positive values indicate comparator better."""
    if direction == "higher":
        return -value
    return value


def _maybe_float(value: Any) -> float | None:
    """Convert to float if possible, returning None for NaN or invalid inputs."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_bool(value: Any) -> bool | None:
    """Convert to bool if the value is not NaN."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return bool(value)


def _extract_oriented_ci(
    rec: pd.Series,
    comparison: ComparisonSpec,
    direction: str,
    lower_col: str,
    upper_col: str,
) -> Tuple[float | None, float | None]:
    """Return oriented CI bounds for the specified columns, if available."""
    if lower_col not in rec.index or upper_col not in rec.index:
        return None, None
    if pd.isna(rec.get(lower_col)) or pd.isna(rec.get(upper_col)):
        return None, None

    try:
        baseline_minus_comp, ci_lower, ci_upper, _, _ = _normalize_effect(
            rec,
            comparison,
            (lower_col,),
            (upper_col,),
        )
    except ValueError:
        return None, None

    _, oriented_lower, oriented_upper = _transform_effect_for_direction(
        baseline_minus_comp,
        ci_lower,
        ci_upper,
        direction,
    )
    return float(oriented_lower), float(oriented_upper)


def _transform_effect_for_direction(
    effect: float,
    lower: float,
    upper: float,
    direction: str,
) -> Tuple[float, float, float]:
    """Orient the effect so that positive values indicate the comparator is better."""
    if direction not in {"lower", "higher"}:
        raise ValueError(f"Unsupported direction '{direction}'. Use 'lower' or 'higher'.")

    if direction == "lower":
        return effect, lower, upper

    flipped_effect = -effect
    flipped_lower = -upper
    flipped_upper = -lower
    if flipped_lower > flipped_upper:
        flipped_lower, flipped_upper = flipped_upper, flipped_lower
    return flipped_effect, flipped_lower, flipped_upper


def _build_offsets(train_sizes: Iterable[int]) -> Dict[int, float]:
    """Assign small vertical offsets per train size so markers do not overlap."""
    sizes = sorted(set(train_sizes))
    if not sizes:
        return {}
    if len(sizes) == 1:
        return {sizes[0]: 0.0}
    # Spread train sizes more aggressively while keeping them within the dataset band.
    half_span = min(0.38, 0.16 * len(sizes))
    # Assign offsets so that smaller train sizes appear slightly above the dataset center
    # (i.e., ordering goes 20 -> 500 from top to bottom when read visually).
    offsets = np.linspace(half_span, -half_span, len(sizes))
    return {ts: float(offset) for ts, offset in zip(sizes, offsets)}


def _build_marker_map(train_sizes: Iterable[int]) -> Dict[int, str]:
    """
    Provide distinct markers per train size to remain distinguishable without color.
    The list is cycled if there are more train sizes than available symbols.
    """
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", ">", "<", "*"]
    sizes = sorted(set(train_sizes))
    if not sizes:
        return {}
    markers = {}
    for idx, ts in enumerate(sizes):
        marker = marker_cycle[idx % len(marker_cycle)]
        markers[ts] = marker
    return markers


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_forest_plot_style() -> None:
    """
    Configure matplotlib for publication-quality forest plots.
    Enforces Times font via LaTeX.
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "text.latex.preamble": r"\usepackage{times} \usepackage{calc}",
        "axes.labelsize": LABEL_FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.title_fontsize": LEGEND_TITLE_FONT_SIZE,
        "figure.titlesize": TITLE_FONT_SIZE_SUPTITLE_COMBINED,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def save_with_fastplot(
    callback: Callable[[Any], None],
    path: Path,
    figsize: Tuple[float, float],
    style: str = "latex",
    fontsize: int = 14,
    pad_inches: float = 0.05,
    use_tight_bbox: bool = True,
) -> None:
    """Persist the figure produced inside the callback using FastPlot with tight bbox.
    
    Args:
        callback: Function that receives plt and draws the plot
        path: Output file path
        figsize: Figure size as (width, height) tuple
        style: Font style - "latex" uses Times via mathptmx package
        fontsize: Base font size for the plot (default 18 for publication)
        pad_inches: Padding around the figure
        use_tight_bbox: Whether to use tight bounding box (handled by fastplot internally)
    """
    def _wrapped_callback(plt_mod: Any) -> None:
        # Disable tight_layout - we manage layout manually with subplots_adjust
        original_tight_layout = plt_mod.tight_layout
        plt_mod.tight_layout = lambda *args, **kwargs: None
        
        try:
            callback(plt_mod)
        finally:
            # Restore original tight_layout
            plt_mod.tight_layout = original_tight_layout
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    pad = max(pad_inches, 0.12)
    rc_params = {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{times}",
        "font.family": "serif",
        "font.serif": ["Times"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "savefig.bbox": "tight" if use_tight_bbox else "standard",
        "savefig.pad_inches": pad,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }

    if HAS_FASTPLOT:
        # Use fastplot.plot() with callback mode to ensure pgfplots compatibility
        # Pass xtick_direction and ytick_direction explicitly to override fastplot defaults
        fastplot.plot(
            data=None,
            path=str(path),
            mode="callback",
            callback=_wrapped_callback,
            style=style,
            figsize=figsize,
            fontsize=fontsize,
            dpi=DPI,
            rcParams=rc_params,
            xtick_direction="out",
            ytick_direction="out",
        )
    else:
        # Fallback to standard matplotlib if fastplot is not available
        plt.rcParams.update(rc_params)
        plt.figure(figsize=figsize)
        _wrapped_callback(plt)
        plt.savefig(path, dpi=DPI, bbox_inches="tight" if use_tight_bbox else None, pad_inches=pad)
        plt.close()


def build_forest_dataframe(rows: List[ForestRow]) -> pd.DataFrame:
    """Return a dataframe ready for plotting from the provided rows."""
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([r.__dict__ for r in rows])
    df.sort_values(["dataset", "train_size"], inplace=True)
    df["dataset_label"] = df["dataset"].apply(_format_display_label)
    return df


def _calculate_max_decimal_places(values: List[float], max_decimals: int) -> int:
    """Calculate the maximum number of decimal places needed for consistent formatting.
    
    Analyzes all values to find the maximum number of significant decimal places.
    If any value requires max_decimals places, all values will use max_decimals for consistency.
    
    Args:
        values: List of float values to analyze
        max_decimals: Maximum allowed decimal places
        
    Returns:
        Number of decimal places to use (0 to max_decimals)
    """
    max_needed = 0
    for val in values:
        # Format with max_decimals to see what's needed
        formatted = f"{val:.{max_decimals}f}"
        if '.' in formatted:
            # Count significant digits after decimal (non-zero digits)
            decimal_part = formatted.split('.')[1]
            # Find the rightmost non-zero digit
            for i in range(len(decimal_part) - 1, -1, -1):
                if decimal_part[i] != '0':
                    # Need i+1 decimal places (0-indexed)
                    max_needed = max(max_needed, i + 1)
                    break
    
    return max_needed


def _create_xaxis_formatter(max_decimals: int) -> FuncFormatter:
    """Create a formatter that formats x-axis tick labels with consistent decimal places.
    
    Always shows exactly max_decimals places for consistency within a plot.
    If max_decimals is 0, shows integers without decimal point.
    
    Args:
        max_decimals: Number of decimal places to always show
        
    Returns:
        A FuncFormatter that formats numbers with exactly max_decimals places.
    """
    def formatter(x: float, pos: int) -> str:
        if max_decimals == 0:
            return f"{int(x)}"
        return f"{x:.{max_decimals}f}"
    
    return FuncFormatter(formatter)


def _calculate_tight_xlim_from_data(df: pd.DataFrame, step: float | None = None) -> Tuple[float, float]:
    """Calculate tight x-axis limits based on actual data, rounded to multiples of step.
    
    Args:
        df: DataFrame with 'effect', 'ci_lower', 'ci_upper' columns
        step: Step size for rounding. If None, auto-determines (0.05 for range <0.1, 0.1 otherwise)
        
    Returns:
        Tuple of (xmin, xmax) rounded to multiples of step
    """
    if df.empty:
        return -0.1, 0.1
    
    # Get all relevant x values (effect and CI bounds)
    x_values = []
    for col in ['effect', 'ci_lower', 'ci_upper']:
        if col in df.columns:
            x_values.extend(df[col].dropna().tolist())
    
    if not x_values:
        return -0.1, 0.1
    
    x_min = min(x_values)
    x_max = max(x_values)
    x_range = x_max - x_min
    
    # Auto-determine step if not provided
    if step is None:
        if x_range < 0.08:
            # For very small ranges (< 0.08), use very fine step (0.01) for better granularity
            # This handles cases like nnaa where data goes from -0.01 to 0.05
            step = 0.01
        elif x_range < 0.1:
            # For small ranges (0.08-0.1), use finer step (0.05)
            step = 0.05
        else:
            # For larger ranges, use standard step (0.1)
            step = 0.1
    
    # Round down to nearest multiple of step for min, round up for max
    # But preserve very small effects: if min/max are very close to zero but not exactly zero,
    # ensure they are included in the range (don't round them to zero)
    x_min_rounded = np.floor(x_min / step) * step
    x_max_rounded = np.ceil(x_max / step) * step
    
    # Preserve very small effects: if x_min is positive but very small (< step), 
    # ensure we don't round it to zero by including zero in the range
    if 0 < x_min < step:
        x_min_rounded = min(0.0, x_min_rounded)  # Include zero and preserve the small positive value
    # Similarly for x_max if negative but very small
    if -step < x_max < 0:
        x_max_rounded = max(0.0, x_max_rounded)  # Include zero and preserve the small negative value
    
    # Ensure we include 0 if it's within the range
    if x_min <= 0 <= x_max:
        # Already included since we round outward
        pass
    elif x_min > 0:
        # All positive, ensure we start at a nice boundary
        # But preserve very small positive values (already handled above)
        if x_min >= step:
            x_min_rounded = max(0.0, x_min_rounded)
    elif x_max < 0:
        # All negative, ensure we end at a nice boundary
        # But preserve very small negative values (already handled above)
        if x_max <= -step:
            x_max_rounded = min(0.0, x_max_rounded)
    
    # For very small ranges with step 0.01 or 0.05, ensure appropriate span
    # Use symmetric range -0.10 to 0.10 only when data is roughly symmetric around 0
    # For asymmetric data (e.g., -0.01 to 0.05), use tighter asymmetric range
    if step == 0.01:
        # For step 0.01, just round to nearest step - no special handling needed
        # This gives precise control for very small ranges like nnaa
        pass
    elif step == 0.05:
        if x_min <= 0 <= x_max:
            # Data includes 0: check if data is roughly symmetric
            # If max is much larger than abs(min), use asymmetric range
            asymmetry_ratio = abs(x_max) / abs(x_min) if x_min < 0 else float('inf')
            if asymmetry_ratio > 3.0 or abs(x_min) < 0.02:
                # Data is asymmetric or very close to 0 on negative side
                # Use tighter range: round to nearest 0.05 but don't force full symmetry
                # Ensure minimum span for readability
                if x_max_rounded - x_min_rounded < 0.10:
                    # Expand slightly but maintain asymmetry
                    if x_min_rounded > -0.05:
                        x_min_rounded = -0.05  # At least show some negative range
                    if x_max_rounded < 0.05:
                        x_max_rounded = max(0.05, x_max_rounded)  # Ensure reasonable positive range
            else:
                # Data is roughly symmetric: use symmetric range -0.10 to 0.10
                x_min_rounded = -0.10
                x_max_rounded = 0.10
        else:
            # Data doesn't include 0: check if range is too small
            rounded_range = x_max_rounded - x_min_rounded
            if rounded_range < 0.15:  # Less than 3 steps (0.05 * 3)
                # Expand symmetrically around data center
                center = (x_min + x_max) / 2
                x_min_rounded = np.floor((center - 0.075) / step) * step
                x_max_rounded = np.ceil((center + 0.075) / step) * step
    
    return float(x_min_rounded), float(x_max_rounded)


def _dataframes_xranges_overlap(dfs: List[pd.DataFrame], min_fraction: float = 0.1) -> bool:
    """Decide whether a combined plot's panels should share one x-axis.

    Returns True only if the panels' x-ranges (over 'effect', 'ci_lower', 'ci_upper')
    overlap by at least ``min_fraction`` of their union. Panels that sit in largely
    disjoint regions (e.g. one all-positive, the other all-negative) return False so
    each panel keeps its own scale instead of leaving half the panel empty.
    """
    ranges = []
    for df in dfs:
        if df is None or getattr(df, "empty", True):
            continue
        vals = []
        for col in ("effect", "ci_lower", "ci_upper"):
            if col in df.columns:
                vals.extend(pd.to_numeric(df[col], errors="coerce").dropna().tolist())
        if vals:
            ranges.append((min(vals), max(vals)))
    if len(ranges) < 2:
        return True
    intersection = min(r[1] for r in ranges) - max(r[0] for r in ranges)
    union = max(r[1] for r in ranges) - min(r[0] for r in ranges)
    if union <= 0:
        return True
    return (intersection / union) >= min_fraction


def _calculate_shared_xlim_from_dataframes(dfs: List[pd.DataFrame], step: float | None = None) -> Tuple[float, float]:
    """Calculate shared x-axis limits from multiple dataframes, rounded to multiples of step.

    Useful for combined plots where all panels should have the same x-axis scale.
    If step is None, automatically determines appropriate step based on data range.
    
    Args:
        dfs: List of DataFrames with 'effect', 'ci_lower', 'ci_upper' columns
        step: Step size for rounding. If None, auto-determines (0.05 for range <0.1, 0.1 otherwise)
        
    Returns:
        Tuple of (xmin, xmax) rounded to multiples of step
    """
    if not dfs:
        return -0.1, 0.1
    
    # Collect all x values from all dataframes
    all_x_values = []
    for df in dfs:
        if df.empty:
            continue
        for col in ['effect', 'ci_lower', 'ci_upper']:
            if col in df.columns:
                all_x_values.extend(df[col].dropna().tolist())
    
    if not all_x_values:
        return -0.1, 0.1
    
    x_min = min(all_x_values)
    x_max = max(all_x_values)
    x_range = x_max - x_min
    
    # Auto-determine step if not provided
    if step is None:
        if x_range < 0.1:
            # For very small ranges, use finer step (0.05)
            step = 0.05
        else:
            # For larger ranges, use standard step (0.1)
            step = 0.1
    
    # Round down to nearest multiple of step for min, round up for max
    x_min_rounded = np.floor(x_min / step) * step
    x_max_rounded = np.ceil(x_max / step) * step
    
    # Ensure we include 0 if it's within the range
    if x_min <= 0 <= x_max:
        # Already included since we round outward
        pass
    elif x_min > 0:
        # All positive, ensure we start at a nice boundary
        x_min_rounded = max(0.0, x_min_rounded)
    elif x_max < 0:
        # All negative, ensure we end at a nice boundary
        x_max_rounded = min(0.0, x_max_rounded)
    
    return float(x_min_rounded), float(x_max_rounded)


def apply_xaxis_tick_locator(ax: Any, df: pd.DataFrame | None = None, shared_xlim: Tuple[float, float] | None = None) -> None:
    """Apply appropriate x-axis tick locator based on the current x-axis range.
    
    If df is provided and range is medium (0.1-1.0), also adjust x-axis limits to fit data 
    tightly (rounded to multiples of 0.1) without extra margins.
    
    If shared_xlim is provided, use those limits instead of calculating from df.
    This is useful for combined plots where all panels should have the same x-axis scale.
    
    This function should be called after render_forest_panel, before or instead of adding margins.
    
    Args:
        ax: Matplotlib axes object
        df: Optional DataFrame with 'effect', 'ci_lower', 'ci_upper' columns for tight limits
        shared_xlim: Optional tuple of (xmin, xmax) to use as shared limits for combined plots
    """
    xlim = ax.get_xlim()
    x_range = abs(xlim[1] - xlim[0])
    
    # If shared_xlim is provided, use it (for combined plots)
    if shared_xlim is not None:
        ax.set_xlim(shared_xlim[0], shared_xlim[1])
        xlim = shared_xlim
        x_range = abs(xlim[1] - xlim[0])
    # If df is provided, calculate tight limits (for any range)
    elif df is not None:
        tight_xmin, tight_xmax = _calculate_tight_xlim_from_data(df, step=None)
        
        # Balance asymmetric limits for noise=1e-2 CPDAG discovered plots
        # Check if this is a noise=1e-2 plot (custom_scm_noise1e-2 dataset)
        is_noise_1e2 = False
        if 'dataset' in df.columns:
            is_noise_1e2 = df['dataset'].str.contains('custom_scm_noise1e-2', na=False).any()
        
        if is_noise_1e2:
            # Check if limits are highly asymmetric (e.g., 0 is far to the right, only left tail)
            # Balance by ensuring reasonable space on both sides of 0
            if tight_xmin <= 0 <= tight_xmax:
                # Data spans zero: check asymmetry
                left_span = abs(tight_xmin)
                right_span = abs(tight_xmax)
                # If left span is much smaller than right span, expand left side
                if left_span > 0 and right_span > 0 and right_span / left_span > 3.0:
                    # Balance by expanding left side to at least 1/3 of right span
                    balanced_left = -right_span / 3.0
                    # Round to nearest 0.05 for consistency
                    balanced_left = np.floor(balanced_left / 0.05) * 0.05
                    tight_xmin = min(tight_xmin, balanced_left)
                # If right span is much smaller than left span, expand right side
                elif left_span > 0 and right_span > 0 and left_span / right_span > 3.0:
                    # Balance by expanding right side to at least 1/3 of left span
                    balanced_right = left_span / 3.0
                    # Round to nearest 0.05 for consistency
                    balanced_right = np.ceil(balanced_right / 0.05) * 0.05
                    tight_xmax = max(tight_xmax, balanced_right)
            elif tight_xmin > 0:
                # All positive: ensure some negative space for visual balance
                if tight_xmin < tight_xmax * 0.1:  # Very close to zero
                    tight_xmin = -tight_xmax * 0.2  # Add 20% negative space
                    tight_xmin = np.floor(tight_xmin / 0.05) * 0.05
            elif tight_xmax < 0:
                # All negative: ensure some positive space for visual balance
                if abs(tight_xmax) < abs(tight_xmin) * 0.1:  # Very close to zero
                    tight_xmax = -tight_xmin * 0.2  # Add 20% positive space
                    tight_xmax = np.ceil(tight_xmax / 0.05) * 0.05
        
        # Set tight limits without margins
        ax.set_xlim(tight_xmin, tight_xmax)
        xlim = (tight_xmin, tight_xmax)
        x_range = abs(xlim[1] - xlim[0])

    # Guard against degenerate ranges (e.g., all effects/CI exactly zero).
    # A null span can produce tick labels collapsed to repeated "0".
    if x_range < 1e-12:
        ax.set_xlim(-0.1, 0.1)
        xlim = (-0.1, 0.1)
        x_range = 0.2
    
    # Set locator first to determine tick values
    # Check if limits are exactly -0.10 to 0.10 (or very close) - use step 0.05 for fine granularity
    is_fine_scale = (abs(xlim[0] + 0.10) < 1e-6 and abs(xlim[1] - 0.10) < 1e-6) or \
                    (abs(xlim[0] + 0.05) < 1e-6 and abs(xlim[1] - 0.05) < 1e-6)
    
    if x_range < 0.08:
        # For very small ranges (< 0.08), use step 0.01 for very fine granularity
        # This handles cases like nnaa where data goes from -0.01 to 0.05
        locator = MultipleLocator(0.01)
        max_decimals = 2
    elif x_range < 0.1 or is_fine_scale:
        # For small ranges (0.08-0.1) or when explicitly set to fine scale (-0.10 to 0.10),
        # use MultipleLocator with step 0.05 for finer granularity
        locator = MultipleLocator(0.05)
        max_decimals = 2
    elif x_range < 1.0:
        # For medium ranges (like ATE difference, frobenius_corr_norm_2marginal),
        # force step 0.1 instead of 0.2 using MultipleLocator
        locator = MultipleLocator(0.1)
        max_decimals = 1
    else:
        # For larger ranges, use standard bins
        locator = MaxNLocator(nbins=7, steps=[1, 2, 5, 10])
        max_decimals = 2
    
    ax.xaxis.set_major_locator(locator)
    
    # Get tick values to determine actual decimal places needed
    tick_values = locator.tick_values(xlim[0], xlim[1])
    # Filter ticks within the axis limits
    tick_values = [t for t in tick_values if xlim[0] <= t <= xlim[1]]
    
    # Calculate actual decimal places needed for consistency
    actual_decimals = _calculate_max_decimal_places(tick_values, max_decimals)
    
    # Apply formatter with consistent decimal places
    ax.xaxis.set_major_formatter(_create_xaxis_formatter(actual_decimals))


def render_forest_panel(
    ax: Any,
    df: pd.DataFrame,
    datasets: List[str],
    dataset_labels: Dict[str, str],
    train_sizes: List[int],
    offsets: Dict[int, float],
    marker_lookup: Dict[int, str],
    color_lookup: Dict[int, Any],
    is_combined: bool = False,
    tick_font_size_override: int | None = None,
    marker_size_legend_override: int | None = None,
) -> Dict[int, Any]:
    """Draw a single forest panel and return legend handles keyed by train size."""
    legend_entries: Dict[int, Any] = {}

    # Increase spacing between datasets
    for idx, dataset in enumerate(datasets):
        base_y = (len(datasets) - idx) * DATASET_SPACING
        if idx % 2 == 1:
            ax.axhspan(
                base_y - 0.5 * DATASET_SPACING,
                base_y + 0.5 * DATASET_SPACING,
                color="#f2f2f2",
                alpha=0.35,
                zorder=0,
            )

        subset = df[df["dataset"] == dataset]
        if subset.empty:
            continue

        for _, rec in subset.iterrows():
            ts = rec["train_size"]
            marker = marker_lookup.get(ts, "o")
            effect = rec["effect"]
            lower = rec["ci_lower"]
            upper = rec["ci_upper"]
            y_pos = base_y + offsets.get(ts, 0.0)

            # Numerical rounding in CSV inputs can occasionally make the point estimate
            # appear (slightly) outside the CI bounds (e.g., effect=0 but ci_lower=8e-5).
            # That would make the symmetric error lengths negative and silently drop the
            # marker. Clamp at zero so we still render the point/CI when bounds are valid.
            if upper < lower:
                continue
            err_low = max(0.0, effect - lower)
            err_high = max(0.0, upper - effect)

            color = color_lookup.get(ts, "#1f77b4")
            is_significant = (
                (rec.get("p_holm") is not None and rec["p_holm"] < 0.05)
                or bool(rec.get("holm_significant_stepdown"))
            )
            # Visual fix: if effect is exactly zero (or effectively zero), treated as non-significant
            # to avoid confusing "significant zero" points in the plot.
            if abs(effect) < 1e-9:
                is_significant = False
            dash_style = "solid" if is_significant else (0, (3, 2))
            marker_face = color if is_significant else "white"

            first_entry = ts not in legend_entries
            label = f"Train {ts}" if first_entry else "_nolegend_"
            handle = ax.errorbar(
                effect,
                y_pos,
                xerr=np.array([[err_low, err_high]]).T,
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=ERROR_BAR_LINEWIDTH,
                capsize=ERROR_BAR_CAPSIZE,
                markersize=MARKER_SIZE_PLOT,
                marker=marker,
                markeredgecolor="black",
                markerfacecolor=marker_face,
                markeredgewidth=MARKER_EDGEWIDTH,
                linestyle="none",
                label=label,
            )

            for bar in handle[2]:
                bar.set_linestyle(dash_style)
                bar.set_color(color)
            for cap in handle[1]:
                cap.set_linestyle(dash_style)
                cap.set_color(color)

            if first_entry:
                core_line = handle.lines[0]
                core_line.set_label(label)
                legend_entries[ts] = Line2D(
                    [],
                    [],
                    marker=marker,
                    linestyle="none",
                    markersize=marker_size_legend_override or MARKER_SIZE_LEGEND_SINGLE,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=MARKER_EDGEWIDTH_LEGEND,
                    color=color,
                    label=label,
                )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.grid(axis="y", linestyle=":", alpha=0.25)
    ax.grid(axis="x", linestyle=":", alpha=0.35)

    # Use larger fonts for combined plots; allow override for very dense layouts
    tick_font_size = (
        tick_font_size_override
        if tick_font_size_override is not None
        else (TICK_FONT_SIZE_COMBINED if is_combined else TICK_FONT_SIZE_SINGLE_ENHANCED)
    )
    
    ax.set_yticks([(len(datasets) - idx) * DATASET_SPACING for idx in range(len(datasets))])
    ax.set_yticklabels(
        [dataset_labels.get(name, name) for name in datasets],
        fontsize=tick_font_size,
    )
    ax.tick_params(axis="y", labelsize=tick_font_size, pad=TICK_PAD_Y, direction="out")
    ax.tick_params(axis="x", labelsize=tick_font_size, pad=TICK_PAD_X, direction="out")
    
    # Use appropriate number format based on x-axis range
    # Remove trailing zeros for cleaner scientific presentation
    # Note: Tick locator will be set definitively after margins are added
    # via apply_xaxis_tick_locator() called from the plotting functions
    xlim = ax.get_xlim()
    x_range = abs(xlim[1] - xlim[0])
    if x_range < 0.1:
        # For small ranges (like k-marginal), use 3 decimal places
        ax.xaxis.set_major_formatter(_create_xaxis_formatter(3))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))
    else:
        # For other ranges, use 2 decimal places
        # Locator will be set definitively after margins are added
        ax.xaxis.set_major_formatter(_create_xaxis_formatter(2))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))

    return legend_entries


def _validate_nnaa_metric(df: pd.DataFrame) -> NnaaCheckSummary | None:
    """
    Check if NNAA medians fall within the expected range (0.5, 0.7].
    Returns a summary object if the metric is NNAA, otherwise None.
    """
    if df.empty:
        return None

    # Filter for NNAA metric if mixed (though usually df is pre-filtered)
    if "metric" in df.columns:
        nnaa_df = df[df["metric"] == "nnaa"]
    else:
        nnaa_df = df

    if nnaa_df.empty:
        return None

    # Check median values
    # We look at 'median_comparator' (the method being evaluated)
    # and 'median_baseline' (the baseline, e.g. Vanilla)
    medians = []
    if "median_comparator" in nnaa_df.columns:
        medians.extend(nnaa_df["median_comparator"].dropna().tolist())
    if "median_baseline" in nnaa_df.columns:
        medians.extend(nnaa_df["median_baseline"].dropna().tolist())

    if not medians:
        return None

    min_val = min(medians)
    max_val = max(medians)
    
    # Identify datasets/train_sizes where value > 0.7
    high_points = []
    for _, row in nnaa_df.iterrows():
        dataset = row.get("dataset", "unknown")
        ts = row.get("train_size", "unknown")
        val_comp = row.get("median_comparator")
        val_base = row.get("median_baseline")
        
        if pd.notna(val_comp) and val_comp > 0.7:
            high_points.append(f"{dataset} (train {ts}, comp={val_comp:.3f})")
        if pd.notna(val_base) and val_base > 0.7:
            high_points.append(f"{dataset} (train {ts}, base={val_base:.3f})")

    return NnaaCheckSummary(
        min_value=min_val,
        max_value=max_val,
        high_points=high_points
    )
