#!/usr/bin/env python3
"""Rebuild the noise=1e-2 Custom SCM spurious-correlation table."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


PAPER_CONFIGS = [
    ("vanilla", "original", "Vanilla original"),
    ("vanilla", "topological", "Vanilla topological"),
    ("vanilla", "reverse_topological", "Vanilla reverse top."),
    ("dag", "topological", "DAG-aware"),
    ("cpdag_minimal", "original", "Oracle PDAG"),
    ("cpdag_discovered", "original", "Discovered CPDAG"),
]
PAPER_TRAIN_SIZES = [20, 100, 500]
METRICS = ["corr_X0_X3", "corr_X0_X2"]
TEX_LABELS = {
    "DAG-aware": r"\gls{dag}-aware",
    "Oracle PDAG": r"\Gls{opdag}",
    "Discovered CPDAG": r"Discovered \gls{cpdag}",
}


def load_npz_frame(path: Path, data_key: str) -> pd.DataFrame:
    with np.load(path, allow_pickle=True) as data:
        values = data[data_key]
        columns = data["column_names"].tolist()
    return pd.DataFrame(values, columns=columns)


def correlations(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "corr_X0_X3": float(pearsonr(frame["X0"], frame["X3"]).statistic),
        "corr_X0_X2": float(pearsonr(frame["X0"], frame["X2"]).statistic),
    }


def synthetic_name(algorithm: str, order: str, train_size: int, seed: int) -> str:
    return f"synthetic_{algorithm}_{order}_ts{train_size}_s{seed}.npz"


def render(value: float) -> str:
    return f"{value:.3f}"


def build_tex(summary: pd.DataFrame, test_corr: dict[str, float]) -> str:
    lines: list[str] = []
    for train_size in PAPER_TRAIN_SIZES:
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{Train size $N = {train_size}$}}}} \\")
        for algorithm, order, label in PAPER_CONFIGS:
            row = summary[
                (summary["algorithm"] == algorithm)
                & (summary["column_order"] == order)
                & (summary["train_size"] == train_size)
            ].iloc[0]
            tex_label = TEX_LABELS.get(label, label)
            lines.append(
                f"{tex_label} & {render(row.mean_corr_X0_X3)} & ({render(row.std_corr_X0_X3)}) "
                f"& {render(row.mean_corr_X0_X2)} & ({render(row.std_corr_X0_X2)}) \\\\"
            )
        lines.append(r"\midrule")
    lines.append(
        f"Test set & {render(test_corr['corr_X0_X3'])} & {{--}} "
        f"& {render(test_corr['corr_X0_X2'])} & {{--}} \\\\"
    )
    return "\n".join(lines) + "\n"


def build_markdown(summary: pd.DataFrame, test_corr: dict[str, float]) -> str:
    lines = [
        "| Method | N | rho(X0,X3) mean | std | rho(X0,X2) mean | std | reps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for train_size in PAPER_TRAIN_SIZES:
        for algorithm, order, label in PAPER_CONFIGS:
            row = summary[
                (summary["algorithm"] == algorithm)
                & (summary["column_order"] == order)
                & (summary["train_size"] == train_size)
            ].iloc[0]
            lines.append(
                f"| {label} | {train_size} | {render(row.mean_corr_X0_X3)} | "
                f"{render(row.std_corr_X0_X3)} | {render(row.mean_corr_X0_X2)} | "
                f"{render(row.std_corr_X0_X2)} | {int(row.n_reps)} |"
            )
    lines.append(
        f"| Test set | -- | {render(test_corr['corr_X0_X3'])} | -- | "
        f"{render(test_corr['corr_X0_X2'])} | -- | -- |"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    archive_root = args.archive_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = archive_root / "cleaned_csv/result_custom_scm_noise1e-2_comparison_experiment_cleaned_reps_100.csv"
    dataset_root = archive_root / "datasets/custom_scm_noise1e-2"
    synthetic_root = dataset_root / "synthetic"
    test_path = dataset_root / "global_test_set.npz"

    cleaned = pd.read_csv(cleaned_path)
    wanted = {(algorithm, order) for algorithm, order, _ in PAPER_CONFIGS}
    cleaned = cleaned[
        cleaned.apply(lambda row: (row["algorithm"], row["column_order"]) in wanted, axis=1)
    ].copy()

    coverage = (
        cleaned.groupby(["algorithm", "column_order", "train_size"])["seed"]
        .agg(n_rows="size", n_unique_seeds="nunique")
        .reset_index()
    )
    if not coverage["n_rows"].eq(100).all() or not coverage["n_unique_seeds"].eq(100).all():
        raise RuntimeError("Canonical cleaned coverage is not exactly 100 unique seeds per cell.")

    test_frame = load_npz_frame(test_path, "X_test")
    test_corr = correlations(test_frame)

    rows: list[dict[str, object]] = []
    for row in cleaned.itertuples(index=False):
        path = synthetic_root / synthetic_name(
            str(row.algorithm), str(row.column_order), int(row.train_size), int(row.seed)
        )
        frame = load_npz_frame(path, "synthetic_data")
        corr = correlations(frame)
        rows.append(
            {
                "algorithm": row.algorithm,
                "column_order": row.column_order,
                "train_size": int(row.train_size),
                "seed": int(row.seed),
                **corr,
                "synthetic_npz": str(path),
                "graph_structure": row.graph_structure,
            }
        )

    per_run = pd.DataFrame(rows)
    summary = (
        per_run.groupby(["algorithm", "column_order", "train_size"])
        .agg(
            n_reps=("seed", "size"),
            mean_corr_X0_X3=("corr_X0_X3", "mean"),
            std_corr_X0_X3=("corr_X0_X3", "std"),
            mean_corr_X0_X2=("corr_X0_X2", "mean"),
            std_corr_X0_X2=("corr_X0_X2", "std"),
        )
        .reset_index()
    )

    per_run.to_csv(output_dir / "spurious_correlations_per_run.csv", index=False)
    summary.to_csv(output_dir / "spurious_correlations_summary.csv", index=False)
    coverage.to_csv(output_dir / "coverage.csv", index=False)
    pd.DataFrame([test_corr]).to_csv(output_dir / "test_set_correlations.csv", index=False)
    (output_dir / "spurious_correlations_table.md").write_text(
        build_markdown(summary, test_corr), encoding="utf-8"
    )
    (output_dir / "spurious_correlations_table_rows.tex").write_text(
        build_tex(summary, test_corr), encoding="utf-8"
    )

    graph_audit = []
    for row in per_run.itertuples(index=False):
        if not isinstance(row.graph_structure, str) or row.graph_structure == "no_graph":
            continue
        graph = ast.literal_eval(row.graph_structure)
        directed = sum(
            len(info.get("parents", [])) if isinstance(info, dict) else len(info)
            for info in graph.values()
        )
        graph_audit.append(
            {
                "algorithm": row.algorithm,
                "column_order": row.column_order,
                "train_size": row.train_size,
                "seed": row.seed,
                "directed_edges_from_graph_structure": directed,
            }
        )
    pd.DataFrame(graph_audit).to_csv(output_dir / "graph_structure_audit.csv", index=False)

    print(f"Processed {len(per_run)} canonical rows.")
    print(f"All cells have 100 repetitions: {summary['n_reps'].eq(100).all()}")
    print(build_markdown(summary, test_corr))


if __name__ == "__main__":
    main()
