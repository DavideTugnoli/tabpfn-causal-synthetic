#!/usr/bin/env python3
"""Consolidate chunked external-baseline CSVs into a validated public result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["correlation_matrix_difference", "k_marginal_tvd", "nnaa"]
PUBLIC_COLUMNS = [
    "dataset",
    "algorithm",
    "column_order",
    "train_size",
    "seed",
    "fit_sample_seconds",
    "total_seconds",
    *METRICS,
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-glob", default="result_*.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--dataset", default="custom_scm")
    parser.add_argument("--expected-train-sizes", default="20,50,100,200,500")
    parser.add_argument("--expected-repetitions", type=int, default=100)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No result CSVs found in {args.input_dir}")

    frames = [pd.read_csv(path).assign(source_file=path.name) for path in paths]
    results = pd.concat(frames, ignore_index=True)
    expected_sizes = [int(value) for value in args.expected_train_sizes.split(",")]
    key = ["dataset", "algorithm", "column_order", "train_size", "seed"]

    failures = results[
        results.get("error", pd.Series(index=results.index, dtype=object)).notna()
        | results.get("traceback", pd.Series(index=results.index, dtype=object)).notna()
    ]
    duplicates = results[results.duplicated(key, keep=False)]
    finite_metrics = np.isfinite(results[METRICS].to_numpy(dtype=float)).all(axis=1)
    counts = (
        results.groupby(["dataset", "algorithm", "column_order", "train_size"], dropna=False)
        .agg(n_rows=("seed", "size"), n_unique_seeds=("seed", "nunique"))
        .reset_index()
        .sort_values("train_size")
    )

    invalid_counts = counts[
        (counts["n_rows"] != args.expected_repetitions)
        | (counts["n_unique_seeds"] != args.expected_repetitions)
    ]
    observed_sizes = sorted(results["train_size"].unique().tolist())
    valid = (
        results["dataset"].eq(args.dataset).all()
        and observed_sizes == expected_sizes
        and failures.empty
        and duplicates.empty
        and finite_metrics.all()
        and invalid_counts.empty
    )

    summary = {
        "valid": bool(valid),
        "input_csvs": len(paths),
        "rows": len(results),
        "dataset": args.dataset,
        "expected_train_sizes": expected_sizes,
        "observed_train_sizes": observed_sizes,
        "expected_repetitions_per_cell": args.expected_repetitions,
        "duplicate_rows": len(duplicates),
        "failure_rows": len(failures),
        "nonfinite_metric_rows": int((~finite_metrics).sum()),
        "cell_counts": counts.to_dict(orient="records"),
    }
    if not valid:
        raise SystemExit(json.dumps(summary, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results[PUBLIC_COLUMNS].sort_values(["train_size", "seed"]).to_csv(
        args.output_dir / args.output_name, index=False
    )
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
