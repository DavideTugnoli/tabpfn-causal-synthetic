#!/usr/bin/env python3
"""Build the random-order sensitivity forest plots with the official pipeline.

The random-order vanilla rows (fixed pool of 10 permutations, cyclically
assigned across the 100 paired cleaned seeds) come from the final full-scope
bundle (`final_100_paired_20260608/tables/random_order_full_scope_paired_long.csv`)
and are appended to the canonical cleaned CSVs as condition
``vanilla_random`` (algorithm="vanilla", column_order="random").

The comparison plotted is ``cross_dag_topological_vs_vanilla_random`` (DAG-aware
topological vs vanilla with random ordering) on the same seven datasets shown
by the paper's main DAG figure (SimGlucose has no full DAG and is excluded,
as in that figure). Statistics are recomputed by the official
`statistical_tests.py` (Wilcoxon-Pratt, Holm within each dataset x metric x
train-size family of prespecified comparisons, paper convention).

Outputs (inside this bundle):
- `data/raw` and `data/nnaa_distance_0p5` plot inputs;
- `forest_plots/{cmd_kmtvd,nnaa_raw,nnaa_distance_0p5}` + `comparison_results/`;
- `tables/forest_significance_counts.csv` (W/L/NS read back from forest CSVs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
REPO = BUNDLE.parents[3]
COMPARISON_ROOT = REPO / "causal_experiments/results/comparison_experiment"
OFFICIAL_DATA = COMPARISON_ROOT / "data"
PAIRED_LONG = BUNDLE / "final_100_paired_20260608/tables/random_order_full_scope_paired_long.csv"

DATASETS = (
    "custom_scm",
    "csuite_large_backdoor",
    "csuite_mixed_confounding",
    "csuite_mixed_simpson",
    "csuite_nonlin_simpson",
    "csuite_symprod_simpson",
    "csuite_weak_arrows",
)
METRICS = ("correlation_matrix_difference", "k_marginal_tvd", "nnaa")
COMPARISON_KEY = "cross_dag_topological_vs_vanilla_random"
FOREST_SUBDIR = "vanilla_random_vs_dag_topological"
FOREST_CSVS = {
    "correlation_matrix_difference": "forest_dag_topological_vs_vanilla_random_correlation_matrix_difference.csv",
    "k_marginal_tvd": "forest_dag_topological_vs_vanilla_random_2marginal.csv",
    "nnaa": "forest_dag_topological_vs_vanilla_random_nnaa.csv",
}

sys.path.insert(0, str(REPO))
from causal_experiments.results.comparison_experiment import forest_plots  # noqa: E402


def build_inputs() -> list[str]:
    raw_dir = BUNDLE / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(PAIRED_LONG)
    random_rows = paired[paired["method"] == "random_order"]

    outputs = []
    for dataset in DATASETS:
        cleaned = pd.read_csv(
            OFFICIAL_DATA / f"result_{dataset}_comparison_experiment_cleaned_reps_100.csv"
        )
        rnd = random_rows[random_rows["dataset"] == dataset].copy()
        rnd["algorithm"] = "vanilla"
        rnd["column_order"] = "random"

        vanilla = cleaned[
            cleaned["algorithm"].astype(str).eq("vanilla")
            & cleaned["column_order"].astype(str).eq("original")
        ]
        for train_size, group in vanilla.groupby("train_size"):
            rnd_seeds = set(rnd.loc[rnd["train_size"].eq(train_size), "seed"])
            if rnd_seeds != set(group["seed"]):
                raise RuntimeError(f"{dataset} N={train_size}: random-order seeds do not match vanilla_original")
        if (rnd[list(METRICS)] < 0).any().any():
            raise RuntimeError(f"{dataset}: negative metric in random-order rows")

        common = [c for c in cleaned.columns if c in rnd.columns]
        combined = pd.concat([cleaned[common], rnd[common]], ignore_index=True)
        out_path = raw_dir / f"result_{dataset}_comparison_experiment_cleaned_reps_100.csv"
        combined.to_csv(out_path, index=False)
        outputs.append(str(out_path))
    return outputs


def configure(inputs: list[str], paper_root: Path, stats_root: Path) -> None:
    forest_plots.PAPER_ROOT = paper_root
    forest_plots.COMPARISON_RESULTS_DIR = stats_root
    forest_plots.COMPARISON_RESULT_FILES = tuple(inputs)
    forest_plots._set_expected_dataset_slugs(tuple(inputs))


def run_forest(inputs: list[str], label: str, metrics: tuple[str, ...]) -> None:
    configure(inputs, BUNDLE / f"forest_plots/{label}", BUNDLE / f"comparison_results/{label}")
    forest_plots.main(
        metric_names=metrics,
        comparison_keys=(COMPARISON_KEY,),
        show_caption=False,
        recompute_stats=True,
        no_csv=False,
        single_column=False,
    )


def collect_counts() -> pd.DataFrame:
    rows = []
    for metric, fname in FOREST_CSVS.items():
        label = "nnaa_raw" if metric == "nnaa" else "cmd_kmtvd"
        df = pd.read_csv(BUNDLE / f"forest_plots/{label}/{FOREST_SUBDIR}/csv/{fname}")
        wins = int(((df["effect"] > 0) & df["is_significant"]).sum())
        losses = int(((df["effect"] < 0) & df["is_significant"]).sum())
        rows.append(
            {
                "metric": metric,
                "cells": len(df),
                "dag_aware_wins": wins,
                "dag_aware_losses": losses,
                "not_significant": int((~df["is_significant"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    inputs = build_inputs()

    run_forest(inputs, "cmd_kmtvd", ("correlation_matrix_difference", "k_marginal_tvd"))
    run_forest(inputs, "nnaa_raw", ("nnaa",))

    distance_dir = BUNDLE / "data/nnaa_distance_0p5"
    distance_dir.mkdir(parents=True, exist_ok=True)
    distance_inputs = []
    for path in inputs:
        df = pd.read_csv(path)
        df["nnaa_raw"] = pd.to_numeric(df["nnaa"], errors="raise")
        df["nnaa"] = (df["nnaa_raw"] - 0.5).abs().where(df["nnaa_raw"] >= 0, df["nnaa_raw"])
        out = distance_dir / Path(path).name
        df.to_csv(out, index=False)
        distance_inputs.append(str(out))
    # usetex is active: bare "|" in text mode renders as an em dash, so wrap
    # the bars and the minus sign in math mode.
    forest_plots.METRIC_CONFIG["nnaa"]["title"] = (
        "$|$Nearest-Neighbor Adversarial Accuracy $-$ 0.5$|$"
    )
    run_forest(distance_inputs, "nnaa_distance_0p5", ("nnaa",))

    tables = BUNDLE / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    counts = collect_counts()
    counts.to_csv(tables / "forest_significance_counts.csv", index=False)
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
