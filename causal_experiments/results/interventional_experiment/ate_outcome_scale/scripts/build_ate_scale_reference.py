#!/usr/bin/env python3
"""Build outcome-scale ATE reference tables from saved interventional results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

MPL_CACHE_DIR = Path("/tmp/tabpfn_ate_outcome_scale_matplotlib")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()


def find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "causal_experiments").exists():
            return parent
    raise RuntimeError(f"Could not locate repository root from {start}")


REPO_ROOT = find_repo_root(SCRIPT_PATH)
DEFAULT_INPUT_DIR = REPO_ROOT / "causal_experiments/results/interventional_experiment/data"
DEFAULT_OUTPUT_DIR = SCRIPT_PATH.parents[1]
DEFAULT_BOOTSTRAPS = 10_000
RANDOM_SEED = 20260424


DATASET_DISPLAY_NAMES = {
    "custom_scm": "Custom SCM",
    "custom_scm_noise1e-2": "Custom SCM (noise 1e-2)",
    "csuite_large_backdoor": "CSuite large backdoor",
    "csuite_mixed_confounding": "CSuite mixed confounding",
    "csuite_mixed_simpson": "CSuite mixed Simpson",
    "csuite_nonlin_simpson": "CSuite nonlinear Simpson",
    "csuite_symprod_simpson": "CSuite symprod Simpson",
    "csuite_weak_arrows": "CSuite weak arrows",
    "simglucose": "SimGlucose",
}

OUTCOME_UNITS = {
    "custom_scm": "SCM outcome units",
    "custom_scm_noise1e-2": "SCM outcome units",
    "simglucose": "mg/dL",
}

CUSTOM_SCM_METADATA = {
    "intervention_variable": "X3",
    "target_variable": "X1",
    "intervention_values": "0,1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ATE scale reference tables from saved interventional CSVs."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=DEFAULT_BOOTSTRAPS,
        help="Bootstrap resamples for percentile CIs of medians.",
    )
    return parser.parse_args()


def dataset_slug_from_path(path: Path) -> str:
    name = path.stem
    if name.startswith("result_"):
        name = name[len("result_") :]
    suffixes = (
        "_intervention_experiment_cleaned_reps_100",
        "_interventional_experiment_cleaned_reps_100",
    )
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) + RANDOM_SEED) % (2**32)


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.median,
    *,
    n_bootstraps: int,
    seed: int,
) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan")
    if clean.size == 1 or n_bootstraps <= 0:
        val = float(statistic(clean))
        return val, val
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, clean.size, size=(n_bootstraps, clean.size))
    samples = clean[sample_idx]
    stats = np.apply_along_axis(statistic, 1, samples)
    return tuple(float(x) for x in np.percentile(stats, [2.5, 97.5]))


def hodges_lehmann(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan")
    pairwise = (clean[:, None] + clean[None, :]) * 0.5
    return float(np.median(pairwise[np.triu_indices(clean.size)]))


def condition_id(algorithm: str, column_order: str) -> str:
    return f"{algorithm}_{column_order}"


def load_raw_results(input_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    paths = sorted(input_dir.glob("result_*intervention*_cleaned_reps_100.csv"))
    if not paths:
        raise FileNotFoundError(f"No interventional result CSVs found in {input_dir}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        dataset = dataset_slug_from_path(path)
        frame = pd.read_csv(path)
        frame["dataset"] = dataset
        frame["dataset_display"] = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        frame["condition"] = [
            condition_id(str(alg), str(order))
            for alg, order in zip(frame["algorithm"], frame["column_order"], strict=False)
        ]
        frame["is_noise_robustness"] = dataset.endswith("noise1e-2")
        if "dataset_name" not in frame.columns:
            frame["dataset_name"] = dataset
        for key, value in CUSTOM_SCM_METADATA.items():
            if key not in frame.columns:
                frame[key] = value if dataset.startswith("custom_scm") else ""
        frame["outcome_units"] = frame["dataset"].map(
            lambda x: OUTCOME_UNITS.get(str(x), "native outcome units")
        )
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "train_size",
        "seed",
        "ate_test",
        "ate_synthetic",
        "ate_difference",
        "ate_relative_error",
    ]
    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    valid = raw[
        np.isfinite(raw["ate_test"])
        & np.isfinite(raw["ate_synthetic"])
        & np.isfinite(raw["ate_difference"])
        & (raw["ate_test"] != -1.0)
        & (raw["ate_synthetic"] != -1.0)
        & (raw["ate_difference"] >= 0.0)
    ].copy()
    valid["train_size"] = valid["train_size"].astype(int)
    valid["seed"] = valid["seed"].astype(int)
    valid["abs_ate_test"] = valid["ate_test"].abs()
    valid["relative_error_percent"] = valid["ate_relative_error"].abs() * 100.0

    return valid, paths


def summarize_methods(raw: pd.DataFrame, n_bootstraps: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = [
        "dataset",
        "dataset_display",
        "is_noise_robustness",
        "outcome_units",
        "intervention_variable",
        "target_variable",
        "intervention_values",
        "algorithm",
        "column_order",
        "condition",
        "train_size",
    ]
    for keys, group in raw.groupby(group_cols, dropna=False, sort=True):
        key_dict = dict(zip(group_cols, keys, strict=False))
        error = group["ate_difference"].to_numpy(dtype=float)
        synthetic_ate = group["ate_synthetic"].to_numpy(dtype=float)
        rel_error = group["relative_error_percent"].to_numpy(dtype=float)
        ate_test = float(group["ate_test"].median())
        seed_parts = (
            key_dict["dataset"],
            key_dict["condition"],
            key_dict["train_size"],
        )
        error_ci = bootstrap_ci(
            error,
            np.median,
            n_bootstraps=n_bootstraps,
            seed=stable_seed(*seed_parts, "ate_difference"),
        )
        synthetic_ci = bootstrap_ci(
            synthetic_ate,
            np.median,
            n_bootstraps=n_bootstraps,
            seed=stable_seed(*seed_parts, "ate_synthetic"),
        )
        rel_error_ci = bootstrap_ci(
            rel_error,
            np.median,
            n_bootstraps=n_bootstraps,
            seed=stable_seed(*seed_parts, "relative_error"),
        )
        rows.append(
            {
                **key_dict,
                "n_seeds": int(group["seed"].nunique()),
                "ate_test": ate_test,
                "abs_ate_test": abs(ate_test),
                "median_ate_synthetic": float(np.median(synthetic_ate)),
                "median_ate_synthetic_ci_low": synthetic_ci[0],
                "median_ate_synthetic_ci_high": synthetic_ci[1],
                "median_abs_error": float(np.median(error)),
                "median_abs_error_ci_low": error_ci[0],
                "median_abs_error_ci_high": error_ci[1],
                "iqr_abs_error_low": float(np.percentile(error, 25)),
                "iqr_abs_error_high": float(np.percentile(error, 75)),
                "mean_abs_error": float(np.mean(error)),
                "median_relative_error_percent": float(np.median(rel_error)),
                "median_relative_error_percent_ci_low": rel_error_ci[0],
                "median_relative_error_percent_ci_high": rel_error_ci[1],
            }
        )
    return pd.DataFrame(rows)


def add_primary_comparisons(raw: pd.DataFrame, summary: pd.DataFrame, n_bootstraps: int) -> pd.DataFrame:
    primary_rows: list[dict[str, object]] = []
    datasets = sorted(summary["dataset"].unique())
    for dataset in datasets:
        comparator = "vanilla_topological" if dataset == "simglucose" else "dag_topological"
        comparator_label = "Vanilla topological" if dataset == "simglucose" else "DAG-aware"
        baseline = "vanilla_original"
        dataset_summary = summary[summary["dataset"] == dataset]
        train_sizes = sorted(dataset_summary["train_size"].unique())
        for train_size in train_sizes:
            base = dataset_summary[
                (dataset_summary["condition"] == baseline)
                & (dataset_summary["train_size"] == train_size)
            ]
            comp = dataset_summary[
                (dataset_summary["condition"] == comparator)
                & (dataset_summary["train_size"] == train_size)
            ]
            if base.empty or comp.empty:
                continue
            base_row = base.iloc[0].to_dict()
            comp_row = comp.iloc[0].to_dict()

            paired_base = raw[
                (raw["dataset"] == dataset)
                & (raw["train_size"] == train_size)
                & (raw["condition"] == baseline)
            ][["seed", "ate_difference"]].rename(columns={"ate_difference": "baseline_abs_error"})
            paired_comp = raw[
                (raw["dataset"] == dataset)
                & (raw["train_size"] == train_size)
                & (raw["condition"] == comparator)
            ][["seed", "ate_difference"]].rename(columns={"ate_difference": "comparator_abs_error"})
            paired = paired_base.merge(paired_comp, on="seed", how="inner")
            paired["abs_error_reduction"] = (
                paired["baseline_abs_error"] - paired["comparator_abs_error"]
            )
            paired = paired[paired["baseline_abs_error"] > 0].copy()
            paired["relative_reduction_percent"] = (
                paired["abs_error_reduction"] / paired["baseline_abs_error"] * 100.0
            )

            baseline_median = float(base_row["median_abs_error"])
            comparator_median = float(comp_row["median_abs_error"])
            if baseline_median > 0:
                median_ratio_reduction = (
                    (baseline_median - comparator_median) / baseline_median * 100.0
                )
            else:
                median_ratio_reduction = float("nan")

            paired_rel = paired["relative_reduction_percent"].to_numpy(dtype=float)
            paired_rel_ci = bootstrap_ci(
                paired_rel,
                np.median,
                n_bootstraps=n_bootstraps,
                seed=stable_seed(dataset, train_size, comparator, "paired_relative_reduction"),
            )

            primary_rows.append(
                {
                    "dataset": dataset,
                    "dataset_display": base_row["dataset_display"],
                    "is_noise_robustness": base_row["is_noise_robustness"],
                    "train_size": int(train_size),
                    "outcome_units": base_row["outcome_units"],
                    "intervention_variable": base_row["intervention_variable"],
                    "target_variable": base_row["target_variable"],
                    "intervention_values": base_row["intervention_values"],
                    "ate_test": base_row["ate_test"],
                    "abs_ate_test": base_row["abs_ate_test"],
                    "baseline_condition": baseline,
                    "baseline_label": "Vanilla original",
                    "baseline_median_abs_error": base_row["median_abs_error"],
                    "baseline_median_abs_error_ci_low": base_row["median_abs_error_ci_low"],
                    "baseline_median_abs_error_ci_high": base_row["median_abs_error_ci_high"],
                    "baseline_median_relative_error_percent": base_row[
                        "median_relative_error_percent"
                    ],
                    "comparator_condition": comparator,
                    "comparator_label": comparator_label,
                    "comparator_median_abs_error": comp_row["median_abs_error"],
                    "comparator_median_abs_error_ci_low": comp_row[
                        "median_abs_error_ci_low"
                    ],
                    "comparator_median_abs_error_ci_high": comp_row[
                        "median_abs_error_ci_high"
                    ],
                    "comparator_median_relative_error_percent": comp_row[
                        "median_relative_error_percent"
                    ],
                    "relative_reduction_from_median_errors_percent": median_ratio_reduction,
                    "paired_n_seeds": int(len(paired)),
                    "paired_median_abs_error_reduction": float(
                        np.median(paired["abs_error_reduction"])
                    )
                    if not paired.empty
                    else float("nan"),
                    "paired_hl_abs_error_reduction": hodges_lehmann(
                        paired["abs_error_reduction"].to_numpy(dtype=float)
                    )
                    if not paired.empty
                    else float("nan"),
                    "paired_median_relative_reduction_percent": float(np.median(paired_rel))
                    if paired_rel.size
                    else float("nan"),
                    "paired_median_relative_reduction_percent_ci_low": paired_rel_ci[0],
                    "paired_median_relative_reduction_percent_ci_high": paired_rel_ci[1],
                }
            )
    return pd.DataFrame(primary_rows)


def fmt_num(value: object, digits: int = 3) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.{digits}g}"


def fmt_ci(row: pd.Series, prefix: str) -> str:
    median = fmt_num(row[f"{prefix}"])
    low = fmt_num(row[f"{prefix}_ci_low"])
    high = fmt_num(row[f"{prefix}_ci_high"])
    return f"{median} [{low}, {high}]"


def build_main_table(primary: pd.DataFrame) -> pd.DataFrame:
    table = primary[~primary["is_noise_robustness"]].copy()
    table = table.sort_values(["dataset_display", "train_size"])
    rows: list[dict[str, object]] = []
    for _, row in table.iterrows():
        rows.append(
            {
                "Dataset": row["dataset_display"],
                "N": int(row["train_size"]),
                "Outcome units": row["outcome_units"],
                "|ATE_test|": fmt_num(row["abs_ate_test"]),
                "Vanilla |Delta_ATE| median [95% CI]": fmt_ci(
                    row, "baseline_median_abs_error"
                ),
                "Comparator": row["comparator_label"],
                "Comparator |Delta_ATE| median [95% CI]": fmt_ci(
                    row, "comparator_median_abs_error"
                ),
                "Reduction from medians (%)": fmt_num(
                    row["relative_reduction_from_median_errors_percent"], 3
                ),
                "Paired median reduction (%)": fmt_num(
                    row["paired_median_relative_reduction_percent"], 3
                ),
            }
        )
    return pd.DataFrame(rows)


def write_markdown_table(table: pd.DataFrame, path: Path) -> None:
    if table.empty:
        path.write_text("_No rows._\n", encoding="utf-8")
        return
    columns = [str(col) for col in table.columns]

    def escape_cell(value: object) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(escape_cell(row[col]) for col in table.columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(table: pd.DataFrame, path: Path) -> None:
    latex = table.to_latex(index=False, escape=True)
    path.write_text(latex, encoding="utf-8")


def plot_simglucose(primary: pd.DataFrame, summary: pd.DataFrame, figures_dir: Path) -> None:
    sim_primary = primary[primary["dataset"] == "simglucose"].sort_values("train_size")
    if sim_primary.empty:
        return

    x = np.arange(len(sim_primary))
    labels = [str(int(v)) for v in sim_primary["train_size"]]
    ate_reference = float(sim_primary["abs_ate_test"].iloc[0])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for offset, prefix, label, color, marker in [
        (-0.08, "baseline", "Vanilla original", "#4C78A8", "o"),
        (0.08, "comparator", "Vanilla topological", "#F58518", "s"),
    ]:
        med = sim_primary[f"{prefix}_median_abs_error"].to_numpy(dtype=float)
        low = sim_primary[f"{prefix}_median_abs_error_ci_low"].to_numpy(dtype=float)
        high = sim_primary[f"{prefix}_median_abs_error_ci_high"].to_numpy(dtype=float)
        yerr = np.vstack([med - low, high - med])
        ax.errorbar(
            x + offset,
            med,
            yerr=yerr,
            marker=marker,
            linewidth=1.8,
            capsize=3,
            label=label,
            color=color,
        )
    ax.axhline(
        ate_reference,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="|ATE_test|",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Train size")
    ax.set_ylabel("Absolute ATE error (mg/dL, log scale)")
    ax.set_title("SimGlucose ATE Error on Outcome Scale")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"simglucose_ate_error_scale.{ext}", dpi=300)
    plt.close(fig)

    sim_summary = summary[
        (summary["dataset"] == "simglucose")
        & (summary["condition"].isin(["vanilla_original", "vanilla_topological"]))
    ].copy()
    sim_summary = sim_summary.sort_values(["condition", "train_size"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for offset, condition, label, color, marker in [
        (-0.08, "vanilla_original", "Vanilla original", "#4C78A8", "o"),
        (0.08, "vanilla_topological", "Vanilla topological", "#F58518", "s"),
    ]:
        subset = sim_summary[sim_summary["condition"] == condition].sort_values("train_size")
        med = subset["median_ate_synthetic"].to_numpy(dtype=float)
        low = subset["median_ate_synthetic_ci_low"].to_numpy(dtype=float)
        high = subset["median_ate_synthetic_ci_high"].to_numpy(dtype=float)
        yerr = np.vstack([med - low, high - med])
        ax.errorbar(
            x + offset,
            med,
            yerr=yerr,
            marker=marker,
            linewidth=1.8,
            capsize=3,
            label=label,
            color=color,
        )
    ax.axhline(
        float(sim_summary["ate_test"].median()),
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="ATE_test",
    )
    ax.set_yscale("symlog", linthresh=5.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Train size")
    ax.set_ylabel("ATE estimate (mg/dL, symlog scale)")
    ax.set_title("SimGlucose Synthetic ATE Estimates")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"simglucose_ate_estimates_scale.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw, input_paths = load_raw_results(args.input_dir)
    summary = summarize_methods(raw, args.n_bootstraps)
    primary = add_primary_comparisons(raw, summary, args.n_bootstraps)
    main_table = build_main_table(primary)

    raw.to_csv(tables_dir / "ate_raw_long_validated.csv", index=False)
    summary.to_csv(tables_dir / "method_error_summary_all_conditions.csv", index=False)
    primary.to_csv(tables_dir / "primary_scale_reference_all_train_sizes.csv", index=False)
    main_table.to_csv(
        tables_dir / "primary_scale_reference_main_all_train_sizes.csv",
        index=False,
    )
    write_markdown_table(main_table, tables_dir / "primary_scale_reference_main_all_train_sizes.md")
    write_latex_table(
        main_table,
        tables_dir / "primary_scale_reference_main_all_train_sizes.tex",
    )

    sim_primary = primary[primary["dataset"] == "simglucose"].copy()
    sim_primary.to_csv(tables_dir / "simglucose_scale_reference_all_train_sizes.csv", index=False)
    plot_simglucose(primary, summary, figures_dir)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        "input_dir": str(args.input_dir.relative_to(REPO_ROOT)),
        "input_files": [str(path.relative_to(REPO_ROOT)) for path in input_paths],
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "included_train_sizes": sorted(int(x) for x in primary["train_size"].unique()),
        "main_table_note": "All available train sizes are included; noise robustness rows are kept in the full CSV.",
        "n_bootstraps": args.n_bootstraps,
        "baseline_condition": "vanilla_original",
        "primary_comparator_rule": {
            "custom_scm_and_csuite": "dag_topological",
            "simglucose": "vanilla_topological",
        },
        "simglucose_units_note": (
            "subcut_glucose is the CGM-like subcutaneous glucose outcome in mg/dL; "
            "intervention action_insulin_U_per_min is U/min."
        ),
    }
    (tables_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote tables to {tables_dir}")
    print(f"Wrote figures to {figures_dir}")
    print("Main table:")
    print(main_table.to_string(index=False))


if __name__ == "__main__":
    main()
