#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FixedFormatter


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = EXPERIMENT_ROOT / "data"
RESULTS_DIR = EXPERIMENT_ROOT / "statistics"
OUTPUT_DIR = EXPERIMENT_ROOT / "figures" / "robustness_summary"

TRAIN_SIZES = [20, 50, 100, 200, 500]
NOISE_CONFIGS = {
    0.1: {
        "csv": DATA_DIR
        / "result_custom_scm_noise0p1_robustness_comparison_experiment_cleaned_reps_100.csv",
        "stats": RESULTS_DIR
        / "custom_scm_noise0p1_robustness"
        / "stat_tests"
        / "posthoc_wilcoxon_summary.csv",
    },
    0.2: {
        "csv": DATA_DIR
        / "result_custom_scm_noise0p2_comparison_experiment_cleaned_reps_100.csv",
        "stats": RESULTS_DIR
        / "custom_scm_noise0p2"
        / "stat_tests"
        / "posthoc_wilcoxon_summary.csv",
    },
    0.5: {
        "csv": DATA_DIR
        / "result_custom_scm_noise0p5_robustness_comparison_experiment_cleaned_reps_100.csv",
        "stats": RESULTS_DIR
        / "custom_scm_noise0p5_robustness"
        / "stat_tests"
        / "posthoc_wilcoxon_summary.csv",
    },
}

EXPECTED_CPDAG = {
    0: {"parents": [], "undirected": []},
    1: {"parents": [0, 2], "undirected": []},
    2: {"parents": [], "undirected": [3]},
    3: {"parents": [], "undirected": [2]},
}


def normalize_graph(graph: object) -> dict[int, dict[str, list[int]]]:
    parsed = ast.literal_eval(str(graph))
    return {
        int(node): {
            "parents": sorted(int(parent) for parent in data.get("parents", [])),
            "undirected": sorted(int(neighbor) for neighbor in data.get("undirected", [])),
        }
        for node, data in parsed.items()
    }


def graph_matches_expected(graph: object) -> bool:
    try:
        return normalize_graph(graph) == EXPECTED_CPDAG
    except Exception:
        return False


def metric_rows(noise: float, paths: dict[str, Path]) -> list[dict[str, float | int]]:
    data = pd.read_csv(paths["csv"])
    stats = pd.read_csv(paths["stats"])

    rows: list[dict[str, float | int]] = []
    cpdag_data = data[
        (data["algorithm"] == "cpdag_discovered")
        & (data["column_order"] == "original")
    ].copy()

    stats = stats[
        (stats["metric"] == "correlation_matrix_difference")
        & (stats["condition_a"] == "vanilla_original")
        & (stats["condition_b"] == "cpdag_discovered_original")
    ].copy()

    for train_size in TRAIN_SIZES:
        stat_row = stats[stats["train_size"] == train_size].iloc[0]
        cpdag_subset = cpdag_data[cpdag_data["train_size"] == train_size]
        recovery = float(cpdag_subset["graph_structure"].map(graph_matches_expected).mean())
        ci_low = float(stat_row["effect_ci_lower_holm"])
        ci_high = float(stat_row["effect_ci_upper_holm"])
        rows.append(
            {
                "noise_level": noise,
                "train_size": train_size,
                "cmd_hl_vanilla_minus_cpdag": float(stat_row["effect_hl"]),
                "cmd_ci_lower_holm": min(ci_low, ci_high, float(stat_row["effect_hl"])),
                "cmd_ci_upper_holm": max(ci_low, ci_high, float(stat_row["effect_hl"])),
                "cmd_p_holm": float(stat_row["p_value_holm"]),
                "cmd_holm_significant": bool(stat_row["holm_significant_stepdown"]),
                "cpdag_exact_recovery": recovery,
            }
        )
    return rows


def plot_summary(summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "custom_scm_noise_recoverability_summary.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times"],
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{times}",
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = {0.1: "#31688e", 0.2: "#35b779", 0.5: "#d95f02"}
    markers = {0.1: "s", 0.2: "o", 0.5: "^"}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), sharex=True)

    for noise in sorted(summary["noise_level"].unique()):
        subset = summary[summary["noise_level"] == noise].sort_values("train_size")
        label = r"$\sigma=0.2$ (CSMr)" if float(noise) == 0.2 else rf"$\sigma={noise:g}$"
        axes[0].errorbar(
            subset["train_size"],
            subset["cmd_hl_vanilla_minus_cpdag"],
            yerr=[
                subset["cmd_hl_vanilla_minus_cpdag"] - subset["cmd_ci_lower_holm"],
                subset["cmd_ci_upper_holm"] - subset["cmd_hl_vanilla_minus_cpdag"],
            ],
            marker=markers[float(noise)],
            color=colors[float(noise)],
            linewidth=1.4,
            capsize=2.5,
            label=label,
        )
        axes[1].plot(
            subset["train_size"],
            100.0 * subset["cpdag_exact_recovery"],
            marker=markers[float(noise)],
            color=colors[float(noise)],
            linewidth=1.4,
            label=label,
        )

    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    axes[0].set_title("CMD improvement")
    axes[0].set_ylabel("HL diff (Vanilla - CPDAG)")
    axes[0].set_xlabel("Train size")

    axes[1].set_title("PC recovery")
    axes[1].set_ylabel(r"Exact CPDAG recovery (\%)")
    axes[1].set_xlabel("Train size")
    axes[1].set_ylim(-2, 102)

    for ax in axes:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(TRAIN_SIZES))
        ax.xaxis.set_major_formatter(FixedFormatter([str(size) for size in TRAIN_SIZES]))
        ax.minorticks_off()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="0.9", linewidth=0.8)

    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUTPUT_DIR / "custom_scm_noise_recoverability_cmd_recovery.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "custom_scm_noise_recoverability_cmd_recovery.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows: list[dict[str, float | int]] = []
    for noise, paths in NOISE_CONFIGS.items():
        rows.extend(metric_rows(noise, paths))
    plot_summary(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
