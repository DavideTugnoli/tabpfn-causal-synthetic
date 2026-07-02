#!/usr/bin/env python3
"""Recompute CMD from saved TabularARGN synthetic NPZ files without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_npz_frame(path: Path, keys: tuple[str, ...]) -> pd.DataFrame:
    with np.load(path, allow_pickle=True) as data:
        for key in keys:
            if key in data:
                values = data[key]
                break
        else:
            raise KeyError(f"None of {keys} found in {path}")
        columns = data["column_names"].tolist()
    return pd.DataFrame(values, columns=columns)


def finite_spearman_cmd(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    if list(real.columns) != list(synthetic.columns):
        raise ValueError("Real and synthetic columns differ")
    real_corr = real.corr(method="spearman").to_numpy(copy=True)
    synthetic_corr = synthetic.corr(method="spearman").to_numpy(copy=True)
    np.fill_diagonal(real_corr, 1.0)
    np.fill_diagonal(synthetic_corr, 1.0)
    real_corr = np.nan_to_num(real_corr, nan=0.0, posinf=0.0, neginf=0.0)
    synthetic_corr = np.nan_to_num(synthetic_corr, nan=0.0, posinf=0.0, neginf=0.0)
    value = float(np.linalg.norm(real_corr - synthetic_corr, ord="fro"))
    if not np.isfinite(value):
        raise ValueError("CMD remains non-finite after deterministic fallback")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-size", type=int, required=True)
    seed_source = parser.add_mutually_exclusive_group(required=True)
    seed_source.add_argument("--seeds", help="Comma-separated exact seed list")
    seed_source.add_argument("--seeds-file", type=Path, help="One exact seed per line")
    parser.add_argument("--synthetic-dir", type=Path, required=True)
    parser.add_argument("--global-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.seeds_file:
        seeds = [
            int(value.strip())
            for value in args.seeds_file.read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
    else:
        seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seed list must be non-empty and unique")

    real = load_npz_frame(args.global_test, ("X_test", "test_data", "data"))
    records: list[dict[str, object]] = []
    for seed in seeds:
        synthetic_path = (
            args.synthetic_dir
            / f"synthetic_tabularargn_unconditional_ts{args.train_size}_s{seed}.npz"
        )
        if not synthetic_path.exists():
            raise FileNotFoundError(synthetic_path)
        synthetic = load_npz_frame(synthetic_path, ("synthetic_data", "data"))
        records.append(
            {
                "dataset": args.dataset,
                "train_size": args.train_size,
                "seed": seed,
                "correlation_matrix_difference": finite_spearman_cmd(real, synthetic),
                "metric_patch": "constant_spearman_nan_to_zero",
                "synthetic_file": synthetic_path.name,
            }
        )

    output = pd.DataFrame.from_records(records).sort_values("seed")
    if len(output) != len(seeds) or not np.isfinite(output["correlation_matrix_difference"]).all():
        raise ValueError("Patch output failed completeness validation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
