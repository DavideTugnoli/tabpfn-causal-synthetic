#!/usr/bin/env python3
"""Statistical testing utilities for the interventional experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STAT_TESTS_ROOT = SCRIPT_DIR / "stat_tests"

from causal_experiments.results.comparison_experiment.statistical_tests import (  # noqa: E402
    CI_ALPHA_DEFAULT,
    StatTestConfig,
    run_stat_tests_for_dataset as _comparison_run_stat_tests,
)
from causal_experiments.utils.visualization_config import METRIC_CONFIG  # noqa: E402


CONDITION_COLORS: Dict[str, str] = {
    "vanilla_original": "#1f77b4",
    "vanilla_topological": "#ff7f0e",
    "vanilla_reverse_topological": "#d62728",
    "dag_topological": "#8c564b",
    "cpdag_minimal_original": "#7f7f7f",
    "cpdag_discovered_original": "#9467bd",
}

CONDITION_ORDER: Sequence[str] = tuple(CONDITION_COLORS.keys())

DEFAULT_METRICS: Sequence[str] = (
    "ate_difference",
    "ate_relative_error",
    "ate_synthetic",
)

def discover_result_files() -> List[str]:
    """Return the list of available interventional result CSV filenames.

    We now store cleaned result CSVs under ``SCRIPT_DIR / "data"`` for
    consistency with the comparison experiment. To remain backward-compatible
    with older layouts, we also look in the root directory.
    """

    search_roots = [SCRIPT_DIR / "data", SCRIPT_DIR]
    patterns = (
        "result_*_intervention_experiment_cleaned*.csv",
        "result_*_interventional_experiment_cleaned*.csv",
    )
    files: set[str] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_file():
                    # store relative path from SCRIPT_DIR so callers can join
                    rel = path.relative_to(SCRIPT_DIR)
                    files.add(str(rel))
    return sorted(files)


INTERVENTIONAL_RESULT_FILES: Sequence[str] = tuple(discover_result_files())


def _normalize_dataset_slug(value: Any) -> str:
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


def _ensure_condition_column(df: pd.DataFrame) -> pd.DataFrame:
    if "condition" in df.columns:
        return df.copy()

    df = df.copy()
    algo = df.get("algorithm", pd.Series(["unknown"] * len(df), dtype="object"))
    order = df.get("column_order", pd.Series(["unknown"] * len(df), dtype="object"))
    df["condition"] = (
        algo.astype(str).str.strip()
        + "_"
        + order.astype(str).str.strip()
    ).str.replace("__+", "_", regex=True).str.strip("_")
    return df


def build_stat_test_config(metrics: Iterable[str] | None = None) -> StatTestConfig:
    metric_list: List[str] = []
    combined: List[str] = list(DEFAULT_METRICS)
    if metrics is not None:
        combined.extend(list(metrics))
    for name in combined:
        if name in METRIC_CONFIG and name not in metric_list:
            metric_list.append(name)

    return StatTestConfig(
        metrics=metric_list,
        metric_config=METRIC_CONFIG,
        condition_order=list(CONDITION_ORDER),
        condition_colors=CONDITION_COLORS,
        alpha=CI_ALPHA_DEFAULT,
    )


def run_stat_tests_for_dataset(
    dataset_df: pd.DataFrame,
    dataset_slug: str,
    output_dir: Path,
    config: StatTestConfig | None = None,
) -> None:
    df_prepared = _ensure_condition_column(dataset_df)
    config = config or build_stat_test_config()
    safe_slug = _normalize_dataset_slug(dataset_slug)
    _comparison_run_stat_tests(df_prepared, safe_slug, Path(output_dir), config)


def compute_statistical_tests(
    df: pd.DataFrame,
    output_root: Path,
    config: StatTestConfig | None = None,
) -> None:
    if "dataset_slug" not in df.columns:
        print("[INFO] Missing dataset_slug column; skipping interventional test computation.")
        return

    config = config or build_stat_test_config()
    output_root = Path(output_root)

    for dataset_slug, dataset_df in df.groupby("dataset_slug"):
        safe_slug = _normalize_dataset_slug(dataset_slug)
        dataset_output_dir = output_root / safe_slug
        run_stat_tests_for_dataset(dataset_df, safe_slug, dataset_output_dir, config=config)


def ensure_stat_tests_from_csvs(
    csv_files: Iterable[str],
    metrics: Iterable[str] | None = None,
) -> None:
    config = build_stat_test_config(metrics)
    for relative in csv_files:
        csv_path = SCRIPT_DIR / relative
        if not csv_path.exists():
            print(f"[WARN] Interventional CSV missing: {relative}")
            continue

        dataset_slug = _dataset_slug_from_filename(csv_path)
        dataset_dir = STAT_TESTS_ROOT / dataset_slug
        posthoc_path = dataset_dir / "posthoc_wilcoxon_summary.csv"
        # Regenerate stat tests if CSV has been updated (e.g., algorithm names changed)
        # Remove old stat test files to force regeneration
        if posthoc_path.exists():
            posthoc_path.unlink()
        # Also remove friedman summary if it exists
        friedman_path = dataset_dir / "friedman_summary.csv"
        if friedman_path.exists():
            friedman_path.unlink()

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Unable to read '{csv_path.name}': {exc}")
            continue

        if df.empty:
            print(f"[INFO] CSV '{csv_path.name}' is empty; skipping test computation.")
            continue

        df = _ensure_condition_column(df)
        try:
            run_stat_tests_for_dataset(df, dataset_slug, dataset_dir, config=config)
            print(f"[INFO] Statistical tests generated for dataset '{dataset_slug}'.")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to compute tests for '{dataset_slug}': {exc}")
