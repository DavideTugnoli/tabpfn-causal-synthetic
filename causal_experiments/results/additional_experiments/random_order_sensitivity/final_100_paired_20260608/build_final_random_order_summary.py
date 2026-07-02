#!/usr/bin/env python3
"""Build exact-100-paired full-scope random-order sensitivity tables."""

from __future__ import annotations

import argparse
import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRICS = ("correlation_matrix_difference", "k_marginal_tvd", "nnaa")
REGULAR_DATASETS = (
    "custom_scm",
    "csuite_large_backdoor",
    "csuite_mixed_confounding",
    "csuite_mixed_simpson",
    "csuite_nonlin_simpson",
    "csuite_symprod_simpson",
    "csuite_weak_arrows",
)
PAPER_DATASETS = (*REGULAR_DATASETS, "simglucose")


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values, kind="mergesort")
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * float(p_values[idx]))
        adjusted[idx] = min(running, 1.0)
    return adjusted


def hodges_lehmann(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    walsh = [(left + right) / 2.0 for left, right in combinations_with_replacement(diffs, 2)]
    return float(np.median(walsh))


def read_random_rows(six_dataset_csv: Path, full_scope_root: Path) -> pd.DataFrame:
    six = pd.read_csv(six_dataset_csv)
    six = six[six["dataset"].isin(REGULAR_DATASETS)].copy()

    frames = [six]
    for path in sorted(full_scope_root.glob("*/N*/random_order_sensitivity_results.csv")):
        frame = pd.read_csv(path)
        frame["dataset"] = frame["dataset"].replace({"simglucose_complete": "simglucose"})
        frames.append(frame)
    random_rows = pd.concat(frames, ignore_index=True)
    random_rows = random_rows[random_rows["dataset"].isin(PAPER_DATASETS)].copy()
    random_rows = random_rows.rename(columns={"row_seed": "seed"})
    random_rows["method"] = "random_order"
    return random_rows[["dataset", "train_size", "seed", "method", *METRICS]]


def read_cleaned_rows(cleaned_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset in PAPER_DATASETS:
        path = cleaned_root / f"result_{dataset}_comparison_experiment_cleaned_reps_100.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "dataset", dataset)

        original = frame[(frame["algorithm"] == "vanilla") & (frame["column_order"] == "original")].copy()
        original["method"] = "vanilla_original"

        if dataset == "simglucose":
            comparator = frame[
                (frame["algorithm"] == "vanilla") & (frame["column_order"] == "topological")
            ].copy()
            comparator["method"] = "vanilla_topological"
        else:
            comparator = frame[(frame["algorithm"] == "dag") & (frame["column_order"] == "topological")].copy()
            comparator["method"] = "dag_topological"

        frames.extend([original, comparator])
    return pd.concat(frames, ignore_index=True)[
        ["dataset", "train_size", "seed", "method", *METRICS]
    ]


def validate_and_combine(random_rows: pd.DataFrame, cleaned_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([random_rows, cleaned_rows], ignore_index=True)
    validation: list[dict[str, object]] = []

    for (dataset, train_size), group in combined.groupby(["dataset", "train_size"], sort=True):
        comparator = "vanilla_topological" if dataset == "simglucose" else "dag_topological"
        required = ("random_order", "vanilla_original", comparator)
        seed_sets: dict[str, set[int]] = {}
        for method in required:
            rows = group[group["method"] == method]
            finite = np.isfinite(rows[list(METRICS)].apply(pd.to_numeric, errors="coerce")).all(axis=1)
            rows = rows[finite]
            seeds = set(rows["seed"].astype(int))
            seed_sets[method] = seeds
            validation.append(
                {
                    "dataset": dataset,
                    "train_size": int(train_size),
                    "method": method,
                    "n_rows": int(len(rows)),
                    "n_unique_seeds": int(len(seeds)),
                    "n_duplicate_seed_rows": int(rows.duplicated("seed", keep=False).sum()),
                }
            )

        common = set.intersection(*seed_sets.values())
        union = set.union(*seed_sets.values())
        valid = len(common) == 100 and len(union) == 100 and all(seeds == common for seeds in seed_sets.values())
        for row in validation[-len(required) :]:
            row["n_common_seeds"] = len(common)
            row["n_union_seeds"] = len(union)
            row["paired_cell_valid"] = valid
        if not valid:
            raise RuntimeError(
                f"Invalid paired cell {dataset} N={train_size}: "
                f"counts={ {method: len(seeds) for method, seeds in seed_sets.items()} }, "
                f"common={len(common)}, union={len(union)}"
            )

    return combined, pd.DataFrame(validation)


def pairwise_detail(combined: pd.DataFrame, datasets: tuple[str, ...], scope: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset in datasets:
        comparator = "vanilla_topological" if dataset == "simglucose" else "dag_topological"
        for train_size in sorted(combined.loc[combined["dataset"] == dataset, "train_size"].unique()):
            cell = combined[(combined["dataset"] == dataset) & (combined["train_size"] == train_size)]
            for baseline in ("random_order", "vanilla_original"):
                left = cell[cell["method"] == baseline].set_index("seed")
                right = cell[cell["method"] == comparator].set_index("seed")
                if set(left.index) != set(right.index) or len(left) != 100:
                    raise RuntimeError(f"Unpaired contrast {dataset} N={train_size} {baseline} vs {comparator}")
                for metric in METRICS:
                    diffs = left.loc[sorted(left.index), metric].to_numpy(float) - right.loc[
                        sorted(right.index), metric
                    ].to_numpy(float)
                    try:
                        p_value = float(
                            wilcoxon(
                                diffs,
                                zero_method="pratt",
                                correction=False,
                                alternative="two-sided",
                                method="auto",
                            ).pvalue
                        )
                    except ValueError:
                        p_value = 1.0
                    records.append(
                        {
                            "scope": scope,
                            "dataset": dataset,
                            "train_size": int(train_size),
                            "metric": metric,
                            "baseline": baseline,
                            "comparator": comparator,
                            "n_pairs": len(diffs),
                            "median_baseline": float(np.median(left[metric])),
                            "median_comparator": float(np.median(right[metric])),
                            "hl_baseline_minus_comparator": hodges_lehmann(diffs),
                            "p_raw": p_value,
                        }
                    )

    detail = pd.DataFrame(records)
    detail["p_holm"] = np.nan
    detail["significant"] = False
    for (_, metric, baseline), indices in detail.groupby(["scope", "metric", "baseline"]).groups.items():
        adjusted = holm_adjust(detail.loc[indices, "p_raw"].to_numpy(float))
        detail.loc[indices, "p_holm"] = adjusted
        detail.loc[indices, "significant"] = adjusted <= 0.05
    detail["direction"] = np.where(
        ~detail["significant"],
        "not_significant",
        np.where(detail["hl_baseline_minus_comparator"] > 0, "comparator_wins", "comparator_loses"),
    )
    return detail


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby(["scope", "baseline", "metric"], sort=True)
        .agg(
            n_cells=("dataset", "size"),
            n_with_100_pairs=("n_pairs", lambda values: int((values == 100).sum())),
            comparator_wins=("direction", lambda values: int((values == "comparator_wins").sum())),
            comparator_losses=("direction", lambda values: int((values == "comparator_loses").sum())),
            not_significant=("direction", lambda values: int((values == "not_significant").sum())),
            median_hl=("hl_baseline_minus_comparator", "median"),
            min_hl=("hl_baseline_minus_comparator", "min"),
            max_hl=("hl_baseline_minus_comparator", "max"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[5]
    parser.add_argument(
        "--six-dataset-csv",
        type=Path,
        default=Path("/path/to/random_order_patch_results_6datasets_long.csv"),
    )
    parser.add_argument(
        "--full-scope-root",
        type=Path,
        default=Path("/path/to/full_scope_patched"),
    )
    parser.add_argument(
        "--cleaned-root",
        type=Path,
        default=root / "causal_experiments/results/comparison_experiment/data",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "tables")
    args = parser.parse_args()

    random_rows = read_random_rows(args.six_dataset_csv, args.full_scope_root)
    cleaned_rows = read_cleaned_rows(args.cleaned_root)
    combined, validation = validate_and_combine(random_rows, cleaned_rows)

    regular = pairwise_detail(combined, REGULAR_DATASETS, "regular_7_dag_aware")
    paper = pairwise_detail(combined, PAPER_DATASETS, "paper_8_comparator")
    detail = pd.concat([regular, paper], ignore_index=True)
    summary = summarize(detail)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "random_order_full_scope_paired_long.csv", index=False)
    validation.to_csv(args.output_dir / "random_order_full_scope_pairing_validation.csv", index=False)
    detail.to_csv(args.output_dir / "random_order_full_scope_pairwise_detail.csv", index=False)
    summary.to_csv(args.output_dir / "random_order_full_scope_pairwise_summary.csv", index=False)
    metadata = {
        "six_dataset_csv": str(args.six_dataset_csv.resolve()),
        "full_scope_root": str(args.full_scope_root.resolve()),
        "cleaned_root": str(args.cleaned_root.resolve()),
        "metrics": list(METRICS),
        "regular_scope": list(REGULAR_DATASETS),
        "paper_scope": list(PAPER_DATASETS),
        "pairing_rule": "exact same 100 canonical cleaned seeds for random, original, and comparator",
        "simglucose_comparator": "vanilla_topological",
    }
    (args.output_dir / "random_order_full_scope_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
