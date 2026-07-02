#!/usr/bin/env python3
"""
Statistical testing utilities for the comparison experiment.

This module centralizes the computation of Friedman and Wilcoxon tests,
including Hodges–Lehmann effect sizes with confidence intervals that are
consistent with the Holm step-down correction used for multiplicity control.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy.stats import friedmanchisquare, wilcoxon

try:
    import scikit_posthocs as sp  # type: ignore
except Exception as exc:  # noqa: BLE001 - match legacy behaviour
    raise RuntimeError(
        "scikit-posthocs is not available. Install the package in the same Python environment before running the script."
    ) from exc

CI_ALPHA_DEFAULT = 0.05


@dataclass
class StatTestConfig:
    """Configuration bundle for statistical testing."""

    metrics: Sequence[str]
    metric_config: Dict[str, Dict[str, Any]]
    condition_order: Sequence[str]
    condition_colors: Dict[str, str]
    alpha: float = CI_ALPHA_DEFAULT


def hodges_lehmann_ci_from_diffs(
    diffs: np.ndarray,
    alpha: float = CI_ALPHA_DEFAULT,
) -> Tuple[float, float, float]:
    """Compute Hodges–Lehmann pseudo-median and CI from paired differences."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return float("nan"), float("nan"), float("nan")
    if np.all(np.abs(diffs) < 1e-12):
        return 0.0, 0.0, 0.0
    if diffs.size == 1:
        single = float(diffs[0])
        return single, single, single

    pairwise = np.add.outer(diffs, diffs) * 0.5
    tri = pairwise[np.triu_indices(diffs.size)]
    pseudo_vals = np.unique(np.sort(tri))
    hl = float(np.median(tri))

    alpha_half = alpha / 2.0
    nvals = pseudo_vals.size
    pv_cache: Dict[Tuple[int, str], float] = {}

    def pvalue_at(idx: int, alternative: str) -> float:
        key = (idx, alternative)
        if key in pv_cache:
            return pv_cache[key]
        theta = float(pseudo_vals[idx])
        try:
            res = wilcoxon(
                diffs - theta,
                zero_method="wilcox",
                correction=False,
                alternative=alternative,
                method="auto",
            )
            p = float(res.pvalue)
        except ValueError:
            p = 1.0
        pv_cache[key] = p
        return p

    if pvalue_at(nvals - 1, "greater") <= alpha_half:
        lower_idx = nvals - 1
    else:
        lo, hi = 0, nvals - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if pvalue_at(mid, "greater") > alpha_half:
                hi = mid
            else:
                lo = mid + 1
        lower_idx = lo

    if pvalue_at(0, "less") <= alpha_half:
        upper_idx = 0
    else:
        lo, hi = 0, nvals - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if pvalue_at(mid, "less") > alpha_half:
                lo = mid
            else:
                hi = mid - 1
        upper_idx = lo

    lower = float(pseudo_vals[lower_idx])
    upper = float(pseudo_vals[upper_idx])
    return hl, lower, upper


def _holm_correction(p_values: Sequence[float]) -> List[float]:
    """Apply Holm–Bonferroni correction to a sequence of p-values."""
    p_values = [float(p) for p in p_values]
    m = len(p_values)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda idx: (p_values[idx], idx))
    adjusted_sorted: List[float] = []
    for rank, idx in enumerate(order):
        factor = m - rank
        adjusted_sorted.append(min(1.0, p_values[idx] * factor))

    for i in range(1, m):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])

    adjusted = [0.0] * m
    for sorted_idx, original_idx in enumerate(order):
        adjusted[original_idx] = adjusted_sorted[sorted_idx]
    return adjusted


def _assign_holm_confidence(
    df: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """
    Enrich ``df`` with Holm-adjusted alpha allocations per comparison.

    Expected columns in ``df``: ``p_value``.
    Adds: ``holm_alpha``, ``holm_rank``, ``holm_stage_threshold``, ``holm_significant_stepdown``.
    """
    if df.empty:
        return df

    df = df.copy()
    df["holm_alpha"] = np.nan
    df["holm_rank"] = np.nan
    df["holm_stage_threshold"] = np.nan
    df["holm_significant_stepdown"] = False

    key_columns = ["dataset", "metric", "train_size"]
    group_keys = [col for col in key_columns if col in df.columns]

    for _, group in df.groupby(group_keys):
        indices = group.index.to_numpy()
        p_values = group["p_value"].astype(float).to_numpy()
        m = len(p_values)
        if m == 0:
            continue

        order = np.argsort(p_values, kind="mergesort")
        thresholds = alpha / (m - np.arange(m))
        failure_rank: Optional[int] = None
        failure_threshold: Optional[float] = None

        for rank_pos, sorted_idx in enumerate(order):
            global_idx = indices[sorted_idx]
            threshold = float(thresholds[rank_pos])
            p_val = p_values[sorted_idx]
            df.at[global_idx, "holm_rank"] = rank_pos + 1
            df.at[global_idx, "holm_stage_threshold"] = threshold

            exceeds = not np.isfinite(p_val) or p_val > threshold
            if failure_rank is None and exceeds:
                failure_rank = rank_pos
                failure_threshold = threshold

            if failure_rank is None:
                df.at[global_idx, "holm_alpha"] = threshold
                df.at[global_idx, "holm_significant_stepdown"] = True
            else:
                df.at[global_idx, "holm_alpha"] = failure_threshold
                df.at[global_idx, "holm_significant_stepdown"] = False

    return df


def _compute_holm_conf_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Holm-adjusted Hodges–Lehmann confidence intervals."""
    if "_diffs" not in df.columns:
        return df

    df = df.copy()
    lower_vals: List[float] = []
    upper_vals: List[float] = []
    level_vals: List[float] = []

    for _, row in df.iterrows():
        diffs = row.get("_diffs")
        alpha = row.get("holm_alpha")
        if diffs is None or not isinstance(diffs, (list, tuple, np.ndarray)):
            lower_vals.append(float("nan"))
            upper_vals.append(float("nan"))
            level_vals.append(float("nan"))
            continue

        diffs_array = np.asarray(diffs, dtype=float)
        if not np.isfinite(alpha) or alpha is None or alpha <= 0:
            lower_vals.append(float("nan"))
            upper_vals.append(float("nan"))
            level_vals.append(float("nan"))
            continue

        _, lower_holm, upper_holm = hodges_lehmann_ci_from_diffs(diffs_array, alpha=float(alpha))
        lower_vals.append(float(lower_holm))
        upper_vals.append(float(upper_holm))
        level_vals.append(float(1.0 - float(alpha)))

    df["effect_ci_lower_holm"] = lower_vals
    df["effect_ci_upper_holm"] = upper_vals
    df["effect_ci_level_holm"] = level_vals
    df = df.drop(columns=["_diffs"])
    return df


def run_stat_tests_for_dataset(
    dataset_df: pd.DataFrame,
    dataset_slug: str,
    output_dir: Path,
    config: StatTestConfig,
) -> None:
    """Run Friedman and Wilcoxon tests for a dataset and persist the results."""
    if "seed" not in dataset_df.columns:
        print(f"[INFO] No seed column for dataset {dataset_slug}; skipping statistical tests.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = dataset_df.copy()
    dataset_df = dataset_df[dataset_df["seed"].notna()]
    if dataset_df.empty:
        print(f"[INFO] No valid data for tests in dataset {dataset_slug}.")
        return

    friedman_records: List[Dict[str, Any]] = []
    posthoc_records: List[Dict[str, Any]] = []

    available_train_sizes = sorted(dataset_df["train_size"].dropna().unique())

    for metric in config.metrics:
        if metric not in dataset_df.columns:
            continue
        metric_slug = config.metric_config.get(metric, {}).get("slug", metric)

        for train_size in available_train_sizes:
            subset = dataset_df[dataset_df["train_size"] == train_size]
            valid = subset[subset[metric].notna()]
            valid = valid[~valid[metric].isin({-1.0, -1})]
            if valid.empty:
                continue

            pivot = valid.pivot_table(index="seed", columns="condition", values=metric)
            if pivot.empty:
                continue

            ordered_conditions = [c for c in config.condition_order if c in pivot.columns]
            extra_conditions = [c for c in pivot.columns if c not in ordered_conditions]
            conditions = ordered_conditions + sorted(extra_conditions)
            if len(conditions) < 2:
                continue

            data_clean = pivot[conditions].dropna()
            if data_clean.shape[0] < 2:
                continue

            posthoc_matrix = None

            if len(conditions) >= 3:
                try:
                    samples = [data_clean[cond].values for cond in conditions]
                    stat, p_value = friedmanchisquare(*samples)
                    friedman_records.append(
                        {
                            "dataset": dataset_slug,
                            "metric": metric,
                            "metric_slug": metric_slug,
                            "train_size": int(train_size),
                            "n_conditions": len(conditions),
                            "n_pairs": int(data_clean.shape[0]),
                            "statistic": float(stat),
                            "p_value": float(p_value),
                        }
                    )
                except ValueError:
                    pass

                index_name = data_clean.index.name or "seed"
                long_df = data_clean.reset_index()
                if index_name != "seed":
                    long_df = long_df.rename(columns={index_name: "seed"})
                long_df = long_df.melt(
                    id_vars="seed",
                    value_vars=conditions,
                    var_name="condition",
                    value_name="value",
                )
                long_df = long_df.dropna(subset=["value"])
                if not long_df.empty:
                    posthoc_matrix = sp.posthoc_wilcoxon(
                        long_df,
                        group_col="condition",
                        val_col="value",
                        p_adjust="holm",
                    )
                    posthoc_matrix.index = conditions
                    posthoc_matrix.columns = conditions

            for cond_a, cond_b in combinations(conditions, 2):
                paired = data_clean[[cond_a, cond_b]].dropna()
                if paired.empty:
                    continue

                values_a = paired[cond_a].values
                values_b = paired[cond_b].values
                diffs = values_a - values_b

                if (np.abs(diffs) < 1e-12).all():
                    stat = 0.0
                    p_raw = 1.0
                else:
                    try:
                        stat, p_raw = wilcoxon(
                            values_a,
                            values_b,
                            zero_method="pratt",
                            correction=False,
                        )
                    except ValueError:
                        continue

                if posthoc_matrix is not None:
                    p_holm_matrix = float(posthoc_matrix.loc[cond_a, cond_b])
                else:
                    p_holm_matrix = float("nan")

                hl_estimate, ci_lower, ci_upper = hodges_lehmann_ci_from_diffs(
                    diffs,
                    alpha=config.alpha,
                )

                posthoc_records.append(
                    {
                        "dataset": dataset_slug,
                        "metric": metric,
                        "metric_slug": metric_slug,
                        "train_size": int(train_size),
                        "condition_a": cond_a,
                        "condition_b": cond_b,
                        "n_pairs": int(len(paired)),
                        "median_a": float(paired[cond_a].median()),
                        "median_b": float(paired[cond_b].median()),
                        "median_diff": float((paired[cond_a] - paired[cond_b]).median()),
                        "mean_diff": float((paired[cond_a] - paired[cond_b]).mean()),
                        "effect_hl": float(hl_estimate),
                        "effect_ci_lower": float(ci_lower),
                        "effect_ci_upper": float(ci_upper),
                        "effect_ci_level": float(1.0 - config.alpha),
                        "statistic": float(stat),
                        "p_value": float(p_raw),
                        "p_value_holm": p_holm_matrix,
                        "_diffs": diffs.tolist(),
                    }
                )

    if not posthoc_records and not friedman_records:
        print(f"[INFO] No comparisons available for dataset {dataset_slug}.")
        return

    if posthoc_records:
        posthoc_df = pd.DataFrame.from_records(posthoc_records)

        # Filter to prespecified comparisons before applying Holm correction
        # Keep only:
        # 1. All comparisons against vanilla_original
        # 2. The comparison vanilla_topological vs dag_topological
        # 3. The comparison cpdag_discovered_original vs dag_discovered_topological
        # 4. The comparison vanilla_random vs dag_topological (random-order sensitivity)
        prespecified_mask = (
            (posthoc_df["condition_a"] == "vanilla_original") |
            (posthoc_df["condition_b"] == "vanilla_original") |
            ((posthoc_df["condition_a"] == "vanilla_topological") &
             (posthoc_df["condition_b"] == "dag_topological")) |
            ((posthoc_df["condition_a"] == "dag_topological") &
             (posthoc_df["condition_b"] == "vanilla_topological")) |
            ((posthoc_df["condition_a"] == "cpdag_discovered_original") &
             (posthoc_df["condition_b"] == "dag_discovered_topological")) |
            ((posthoc_df["condition_a"] == "dag_discovered_topological") &
             (posthoc_df["condition_b"] == "cpdag_discovered_original")) |
            ((posthoc_df["condition_a"] == "vanilla_random") &
             (posthoc_df["condition_b"] == "dag_topological")) |
            ((posthoc_df["condition_a"] == "dag_topological") &
             (posthoc_df["condition_b"] == "vanilla_random"))
        )
        posthoc_df = posthoc_df[prespecified_mask].copy()

        # Ensure Holm correction is available for every comparison
        for (metric, train_size), group in posthoc_df.groupby(["metric", "train_size"]):
            adjusted = _holm_correction(group["p_value"].tolist())
            posthoc_df.loc[group.index, "p_value_holm"] = adjusted

        # Allocate Holm-specific alpha thresholds and compute adjusted CIs
        posthoc_df = _assign_holm_confidence(posthoc_df, alpha=config.alpha)
        posthoc_df["holm_significant"] = posthoc_df["p_value_holm"] <= config.alpha
        posthoc_df = _compute_holm_conf_intervals(posthoc_df)

        posthoc_df.sort_values(
            ["metric", "train_size", "condition_a", "condition_b"],
            inplace=True,
        )
        posthoc_path = output_dir / "posthoc_wilcoxon_summary.csv"
        posthoc_df.to_csv(posthoc_path, index=False)
        print(f"[INFO] Post-hoc Wilcoxon tests saved to {posthoc_path} ({len(posthoc_df)} comparisons)")

    if friedman_records:
        friedman_df = pd.DataFrame.from_records(friedman_records)
        friedman_df.sort_values(["metric", "train_size"], inplace=True)
        friedman_path = output_dir / "friedman_summary.csv"
        friedman_df.to_csv(friedman_path, index=False)
        print(f"[INFO] Friedman tests saved to {friedman_path} ({len(friedman_df)} configurations)")


def compute_statistical_tests(
    df: pd.DataFrame,
    output_root: Path,
    config: StatTestConfig,
    normalize_slug: Optional[Callable[[Any], str]] = None,
) -> None:
    """Compute statistical tests for every dataset present in ``df``."""
    if "dataset_slug" not in df.columns:
        print("[INFO] Column dataset_slug missing; skipping test computation.")
        return

    if normalize_slug is None:
        normalize_slug = lambda x: str(x).strip().lower().replace(" ", "_")

    df = df.copy()
    df["dataset_slug"] = df["dataset_slug"].apply(normalize_slug)
    config = config or build_stat_test_config()
    output_root = Path(output_root)

    for dataset_slug, dataset_df in df.groupby("dataset_slug"):
        safe_slug = _normalize_dataset_slug(dataset_slug)
        dataset_output_dir = output_root / safe_slug
        run_stat_tests_for_dataset(dataset_df, safe_slug, dataset_output_dir, config=config)


def build_stat_test_config(metrics: Iterable[str] | None = None) -> StatTestConfig:
    """Build a StatTestConfig from metrics."""
    from causal_experiments.utils.visualization_config import METRIC_CONFIG
    from causal_experiments.results.comparison_experiment.forest_plots import (
        CONDITION_COLORS,
        CONDITION_ORDER,
    )

    metric_list: List[str] = []
    if metrics is not None:
        for name in metrics:
            if name in METRIC_CONFIG and name not in metric_list:
                metric_list.append(name)

    return StatTestConfig(
        metrics=metric_list,
        metric_config=METRIC_CONFIG,
        condition_order=list(CONDITION_ORDER),
        condition_colors=CONDITION_COLORS,
        alpha=CI_ALPHA_DEFAULT,
    )


def _normalize_dataset_slug(value: Any) -> str:
    """Normalize a dataset slug to a safe filesystem name."""
    if value is None:
        return "dataset_unknown"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "dataset_unknown"
    cleaned = text.replace("\\", "/").split("/")[-1]
    import re

    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned.lower() if cleaned else "dataset_unknown"
