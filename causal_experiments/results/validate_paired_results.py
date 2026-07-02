#!/usr/bin/env python3
"""Validate exact paired-seed completeness before promoting final result CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def validate(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    data = pd.read_csv(args.csv)
    group_cols = parse_list(args.group_cols)
    metric_cols = parse_list(args.metric_cols)
    required_methods = parse_list(args.required_methods) if args.required_methods else sorted(
        data[args.method_col].dropna().astype(str).unique()
    )

    required = set(group_cols + [args.method_col, args.seed_col] + metric_cols)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.copy()
    data[args.method_col] = data[args.method_col].astype(str)
    records: list[dict[str, object]] = []

    for group_key, group in data.groupby(group_cols, dropna=False, sort=True):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_cols, key_values))
        method_seeds: dict[str, set[object]] = {}

        for method in required_methods:
            rows = group[group[args.method_col] == method]
            duplicate_count = int(rows.duplicated([args.seed_col], keep=False).sum())
            finite_metrics = np.isfinite(rows[metric_cols].apply(pd.to_numeric, errors="coerce")).all(axis=1)
            valid_rows = rows[finite_metrics]
            seeds = set(valid_rows[args.seed_col].tolist())
            method_seeds[method] = seeds
            records.append(
                {
                    **base,
                    "method": method,
                    "n_rows": int(len(rows)),
                    "n_valid_rows": int(len(valid_rows)),
                    "n_unique_valid_seeds": int(len(seeds)),
                    "n_duplicate_seed_rows": duplicate_count,
                }
            )

        seed_sets = list(method_seeds.values())
        common = set.intersection(*seed_sets) if seed_sets else set()
        union = set.union(*seed_sets) if seed_sets else set()
        group_ok = (
            len(common) == args.expected_seeds
            and len(union) == args.expected_seeds
            and all(seeds == common for seeds in seed_sets)
        )
        for record in records[-len(required_methods) :]:
            record["n_common_seeds"] = len(common)
            record["n_union_seeds"] = len(union)
            record["paired_cell_valid"] = group_ok

    report = pd.DataFrame.from_records(records)
    summary = {
        "source_csv": str(args.csv.resolve()),
        "expected_seeds": args.expected_seeds,
        "group_columns": group_cols,
        "method_column": args.method_col,
        "seed_column": args.seed_col,
        "required_methods": required_methods,
        "metric_columns": metric_cols,
        "n_groups": int(report[group_cols].drop_duplicates().shape[0]),
        "n_valid_groups": int(
            report.loc[report["paired_cell_valid"], group_cols].drop_duplicates().shape[0]
        ),
        "all_groups_valid": bool(report["paired_cell_valid"].all()),
    }
    return report, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--group-cols", default="dataset,train_size")
    parser.add_argument("--method-col", default="condition")
    parser.add_argument("--seed-col", default="seed")
    parser.add_argument("--required-methods", default=None)
    parser.add_argument("--metric-cols", required=True)
    parser.add_argument("--expected-seeds", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report, summary = validate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_dir / "paired_seed_validation.csv", index=False)
    (args.output_dir / "paired_seed_validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not summary["all_groups_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
