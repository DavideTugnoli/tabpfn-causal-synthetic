#!/usr/bin/env python3
"""Compare downloaded TabularARGN metrics against local TabPFN cleaned results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[2]
TABULARARGN_RAW = ROOT / "raw"
ANALYSIS_DIR = ROOT / "analysis"
FIGURES_DIR = ROOT / "figures"

TABPFN_DATA_DIR = (
    PROJECT_ROOT
    / "tabpfn-causal-synthetic"
    / "causal_experiments"
    / "results"
    / "comparison_experiment"
    / "data"
)

DATASET_FILES = {
    "custom_scm": "result_custom_scm_comparison_experiment_cleaned_reps_100.csv",
    "custom_scm_noise1e-2": "result_custom_scm_noise1e-2_comparison_experiment_cleaned_reps_100.csv",
    "csuite_symprod_simpson": "result_csuite_symprod_simpson_comparison_experiment_cleaned_reps_100.csv",
    "csuite_mixed_confounding": "result_csuite_mixed_confounding_comparison_experiment_cleaned_reps_100.csv",
    "csuite_large_backdoor": "result_csuite_large_backdoor_comparison_experiment_cleaned_reps_100.csv",
}

METHOD_ORDER = [
    "TabularARGN",
    "TabPFN vanilla original",
    "TabPFN vanilla topological",
    "TabPFN DAG-aware",
]

TABPFN_METHODS = {
    ("vanilla", "original"): "TabPFN vanilla original",
    ("vanilla", "topological"): "TabPFN vanilla topological",
    ("dag", "topological"): "TabPFN DAG-aware",
}

METRIC_INFO = {
    "correlation_matrix_difference": ("CMD", "lower"),
    "k_marginal_tvd": ("kMTVD", "lower"),
    "nnaa": ("NNAA distance to 0.5", "distance_to_half"),
}


def load_tabularargn() -> pd.DataFrame:
    frames = []
    for csv_path in sorted(TABULARARGN_RAW.glob("*/*.csv")):
        frame = pd.read_csv(csv_path)
        frame["source_file"] = str(csv_path.relative_to(ROOT))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No TabularARGN CSV files found under {TABULARARGN_RAW}")
    data = pd.concat(frames, ignore_index=True)
    data["method"] = "TabularARGN"
    return data


def load_tabpfn() -> pd.DataFrame:
    frames = []
    for dataset, filename in DATASET_FILES.items():
        csv_path = TABPFN_DATA_DIR / filename
        frame = pd.read_csv(csv_path)
        frame["dataset"] = dataset
        frame["method"] = [
            TABPFN_METHODS.get((algorithm, column_order), "")
            for algorithm, column_order in zip(frame["algorithm"], frame["column_order"])
        ]
        frame = frame[frame["method"] != ""].copy()
        frame["source_file"] = str(csv_path.relative_to(PROJECT_ROOT))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def add_metric_losses(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["cmd_loss"] = data["correlation_matrix_difference"]
    data["kmtvd_loss"] = data["k_marginal_tvd"]
    data["nnaa_loss"] = (data["nnaa"] - 0.5).abs()
    return data


def build_paired_long(tabularargn: pd.DataFrame, tabpfn: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["dataset", "train_size", "seed"]
    tgn_cols = key_cols + [
        "correlation_matrix_difference",
        "k_marginal_tvd",
        "nnaa",
        "cmd_loss",
        "kmtvd_loss",
        "nnaa_loss",
        "fit_seconds",
        "sample_seconds",
        "total_seconds",
    ]
    tabular = tabularargn[tgn_cols].rename(
        columns={
            "correlation_matrix_difference": "tabularargn_correlation_matrix_difference",
            "k_marginal_tvd": "tabularargn_k_marginal_tvd",
            "nnaa": "tabularargn_nnaa",
            "cmd_loss": "tabularargn_cmd_loss",
            "kmtvd_loss": "tabularargn_kmtvd_loss",
            "nnaa_loss": "tabularargn_nnaa_loss",
        }
    )

    rows = []
    for method, method_df in tabpfn.groupby("method"):
        pfn_cols = key_cols + [
            "correlation_matrix_difference",
            "k_marginal_tvd",
            "nnaa",
            "cmd_loss",
            "kmtvd_loss",
            "nnaa_loss",
        ]
        merged = tabular.merge(
            method_df[pfn_cols].rename(
                columns={
                    "correlation_matrix_difference": "tabpfn_correlation_matrix_difference",
                    "k_marginal_tvd": "tabpfn_k_marginal_tvd",
                    "nnaa": "tabpfn_nnaa",
                    "cmd_loss": "tabpfn_cmd_loss",
                    "kmtvd_loss": "tabpfn_kmtvd_loss",
                    "nnaa_loss": "tabpfn_nnaa_loss",
                }
            ),
            on=key_cols,
            how="inner",
        )
        merged["tabpfn_method"] = method
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def summarize_methods(all_methods: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("correlation_matrix_difference", "CMD median"),
        ("k_marginal_tvd", "kMTVD median"),
        ("nnaa", "NNAA median"),
        ("nnaa_loss", "NNAA |x-0.5| median"),
        ("total_seconds", "Total seconds median"),
    ]
    rows = []
    for (dataset, train_size, method), group in all_methods.groupby(["dataset", "train_size", "method"]):
        row = {
            "dataset": dataset,
            "train_size": train_size,
            "method": method,
            "n": len(group),
        }
        for column, label in metrics:
            if column in group.columns and group[column].notna().any():
                row[label] = group[column].median()
            else:
                row[label] = np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["method"] = pd.Categorical(summary["method"], METHOD_ORDER, ordered=True)
    return summary.sort_values(["dataset", "train_size", "method"]).reset_index(drop=True)


def summarize_pairwise(paired: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        ("CMD", "cmd_loss"),
        ("kMTVD", "kmtvd_loss"),
        ("NNAA |x-0.5|", "nnaa_loss"),
    ]
    rows = []
    for (dataset, train_size, method), group in paired.groupby(["dataset", "train_size", "tabpfn_method"]):
        for metric_label, loss_col in metric_columns:
            t_col = f"tabularargn_{loss_col}"
            p_col = f"tabpfn_{loss_col}"
            metric_group = group[
                np.isfinite(group[t_col].to_numpy(dtype=float))
                & np.isfinite(group[p_col].to_numpy(dtype=float))
            ].copy()
            diff = metric_group[t_col] - metric_group[p_col]
            diff_values = diff.to_numpy(dtype=float)
            rows.append(
                {
                    "dataset": dataset,
                    "train_size": train_size,
                    "tabpfn_method": method,
                    "metric": metric_label,
                    "n_pairs": len(metric_group),
                    "tabularargn_median_loss": metric_group[t_col].median(),
                    "tabpfn_median_loss": metric_group[p_col].median(),
                    "median_loss_diff_tabularargn_minus_tabpfn": diff.median(),
                    "hodges_lehmann_diff_tabularargn_minus_tabpfn": hodges_lehmann_one_sample(diff_values),
                    "wilcoxon_pratt_p_tabularargn_greater": wilcoxon_pratt_greater(diff_values),
                    "tabpfn_better_fraction": (diff > 0).mean(),
                    "tabularargn_better_fraction": (diff < 0).mean(),
                    "tie_fraction": (diff == 0).mean(),
                }
            )
    summary = pd.DataFrame(rows).sort_values(["metric", "dataset", "train_size", "tabpfn_method"])
    summary["wilcoxon_pratt_p_holm_by_tabpfn_method"] = np.nan
    for method, index in summary.groupby("tabpfn_method").groups.items():
        adjusted = holm_adjust(summary.loc[index, "wilcoxon_pratt_p_tabularargn_greater"].to_numpy(dtype=float))
        summary.loc[index, "wilcoxon_pratt_p_holm_by_tabpfn_method"] = adjusted
    return summary


def hodges_lehmann_one_sample(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    walsh = (values[:, None] + values[None, :]) / 2.0
    return float(np.median(walsh))


def wilcoxon_pratt_greater(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0 or np.all(values == 0):
        return np.nan
    return float(wilcoxon(values, zero_method="pratt", alternative="greater").pvalue)


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(pvalues, np.nan, dtype=float)
    finite = np.isfinite(pvalues)
    finite_positions = np.where(finite)[0]
    finite_values = pvalues[finite]
    order = np.argsort(finite_values)
    ordered_positions = finite_positions[order]
    ordered_values = finite_values[order]
    m = len(ordered_values)
    running_max = 0.0
    for rank, (position, value) in enumerate(zip(ordered_positions, ordered_values)):
        candidate = min(1.0, (m - rank) * value)
        running_max = max(running_max, candidate)
        adjusted[position] = running_max
    return adjusted


def summarize_wins(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, metric), group in pairwise.groupby(["tabpfn_method", "metric"]):
        rows.append(
            {
                "tabpfn_method": method,
                "metric": metric,
                "cells": len(group),
                "tabpfn_better_cells": int((group["median_loss_diff_tabularargn_minus_tabpfn"] > 0).sum()),
                "tabularargn_better_cells": int((group["median_loss_diff_tabularargn_minus_tabpfn"] < 0).sum()),
                "tied_cells": int((group["median_loss_diff_tabularargn_minus_tabpfn"] == 0).sum()),
                "median_of_cellwise_differences": group[
                    "median_loss_diff_tabularargn_minus_tabpfn"
                ].median(),
            }
        )
    return pd.DataFrame(rows).sort_values(["metric", "tabpfn_method"])


def summarize_dataset_ratios(pairwise: pd.DataFrame, method: str) -> pd.DataFrame:
    subset = pairwise[pairwise["tabpfn_method"] == method].copy()
    subset["loss_ratio_tabularargn_over_tabpfn"] = (
        subset["tabularargn_median_loss"] / subset["tabpfn_median_loss"].replace(0, np.nan)
    )
    rows = []
    for (dataset, metric), group in subset.groupby(["dataset", "metric"]):
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "cells": len(group),
                "tabpfn_better_cells": int((group["median_loss_diff_tabularargn_minus_tabpfn"] > 0).sum()),
                "median_tabularargn_loss": group["tabularargn_median_loss"].median(),
                "median_tabpfn_loss": group["tabpfn_median_loss"].median(),
                "significant_cells_holm_0p05": int(
                    (group["wilcoxon_pratt_p_holm_by_tabpfn_method"] < 0.05).sum()
                ),
                "median_loss_ratio_tabularargn_over_tabpfn": group[
                    "loss_ratio_tabularargn_over_tabpfn"
                ].median(),
                "min_loss_ratio_tabularargn_over_tabpfn": group[
                    "loss_ratio_tabularargn_over_tabpfn"
                ].min(),
                "max_loss_ratio_tabularargn_over_tabpfn": group[
                    "loss_ratio_tabularargn_over_tabpfn"
                ].max(),
            }
        )
    return pd.DataFrame(rows).sort_values(["metric", "dataset"]).reset_index(drop=True)


def make_heatmap(pairwise: pd.DataFrame, method: str, metric: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    subset = pairwise[(pairwise["tabpfn_method"] == method) & (pairwise["metric"] == metric)].copy()
    if subset.empty:
        return
    subset["ratio"] = subset["tabularargn_median_loss"] / subset["tabpfn_median_loss"].replace(0, np.nan)
    table = subset.pivot(index="dataset", columns="train_size", values="ratio").sort_index()
    if table.empty:
        return

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    values = np.log10(table.to_numpy(dtype=float))
    if np.all(np.isnan(values)):
        plt.close(fig)
        return
    vmax = np.nanmax(np.abs(values))
    vmax = max(vmax, 0.1)
    im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_xlabel("Train size")
    ax.set_title(f"TabularARGN / {method} median loss ratio - {metric}")

    for i, dataset in enumerate(table.index):
        for j, train_size in enumerate(table.columns):
            ratio = table.loc[dataset, train_size]
            if pd.isna(ratio):
                text = "NA"
            else:
                text = f"{ratio:.1f}x"
            ax.text(j, i, text, ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10 ratio; positive = TabPFN better")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def dataframe_to_markdown(data: pd.DataFrame) -> str:
    if data.empty:
        return "_No rows._"

    rows = []
    columns = list(data.columns)
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in data.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_markdown_summary(
    wins: pd.DataFrame,
    pairwise: pd.DataFrame,
    dag_dataset_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# TabularARGN vs TabPFN comparison",
        "",
        "This analysis uses only downloaded metric CSV/JSON files from Leonardo; synthetic NPZ files are not copied.",
        "Rows are paired by `(dataset, train_size, seed)` against the local cleaned TabPFN comparison CSVs.",
        "For CMD and kMTVD lower is better; for NNAA the comparison uses `abs(NNAA - 0.5)`.",
        "",
        "## Cell-wise win counts",
        "",
        dataframe_to_markdown(wins),
        "",
        "## Dataset summary against TabPFN DAG-aware",
        "",
        dataframe_to_markdown(dag_dataset_summary),
        "",
        "## Strongest TabularARGN losses against TabPFN DAG-aware",
        "",
    ]
    dag = pairwise[pairwise["tabpfn_method"] == "TabPFN DAG-aware"].copy()
    dag = dag.sort_values("median_loss_diff_tabularargn_minus_tabpfn", ascending=False)
    cols = [
        "dataset",
        "train_size",
        "metric",
        "n_pairs",
        "tabularargn_median_loss",
        "tabpfn_median_loss",
        "median_loss_diff_tabularargn_minus_tabpfn",
        "tabpfn_better_fraction",
    ]
    lines.append(dataframe_to_markdown(dag[cols].head(20)))
    lines.append("")
    output_path.write_text("\n".join(lines))


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tabularargn = add_metric_losses(load_tabularargn())
    tabpfn = add_metric_losses(load_tabpfn())

    common_datasets = sorted(set(tabularargn["dataset"]).intersection(tabpfn["dataset"]))
    tabularargn = tabularargn[tabularargn["dataset"].isin(common_datasets)].copy()
    tabpfn = tabpfn[tabpfn["dataset"].isin(common_datasets)].copy()

    all_methods = pd.concat(
        [
            tabularargn,
            tabpfn,
        ],
        ignore_index=True,
        sort=False,
    )

    paired = build_paired_long(tabularargn, tabpfn)
    method_summary = summarize_methods(all_methods)
    pairwise = summarize_pairwise(paired)
    wins = summarize_wins(pairwise)
    dag_dataset_summary = summarize_dataset_ratios(pairwise, "TabPFN DAG-aware")

    tabularargn.to_csv(ANALYSIS_DIR / "tabularargn_metrics_long.csv", index=False)
    paired.to_csv(ANALYSIS_DIR / "tabularargn_vs_tabpfn_paired_long.csv", index=False)
    method_summary.to_csv(ANALYSIS_DIR / "median_metrics_by_method.csv", index=False)
    pairwise.to_csv(ANALYSIS_DIR / "pairwise_tabularargn_vs_tabpfn_by_cell.csv", index=False)
    wins.to_csv(ANALYSIS_DIR / "tabularargn_vs_tabpfn_win_counts.csv", index=False)
    dag_dataset_summary.to_csv(
        ANALYSIS_DIR / "tabularargn_vs_tabpfn_dag_aware_dataset_summary.csv", index=False
    )
    write_markdown_summary(
        wins,
        pairwise,
        dag_dataset_summary,
        ANALYSIS_DIR / "tabularargn_vs_tabpfn_summary.md",
    )

    for method in ["TabPFN vanilla original", "TabPFN DAG-aware"]:
        slug = method.lower().replace(" ", "_").replace("-", "_")
        for metric in ["CMD", "kMTVD", "NNAA |x-0.5|"]:
            metric_slug = metric.lower().replace(" ", "_").replace("|", "").replace(".", "p5")
            make_heatmap(pairwise, method, metric, FIGURES_DIR / f"loss_ratio_tabularargn_vs_{slug}_{metric_slug}.pdf")

    print(f"Wrote analysis outputs to {ANALYSIS_DIR}")
    print(f"Wrote figures to {FIGURES_DIR}")
    print(wins.to_string(index=False))


if __name__ == "__main__":
    main()
