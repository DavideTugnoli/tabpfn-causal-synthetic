#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def cleaned_vanilla_original_seeds(csv_path: Path, train_size: int, limit: int = 100) -> list[int]:
    metric_cols = ["correlation_matrix_difference", "k_marginal_tvd", "nnaa"]
    seeds: list[int] = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("algorithm") != "vanilla" or row.get("column_order") != "original":
                continue
            if int(float(row["train_size"])) != train_size:
                continue
            ok = True
            for col in metric_cols:
                try:
                    ok = ok and float(row[col]) >= 0
                except Exception:
                    ok = False
            if ok:
                seeds.append(int(float(row["seed"])))
    seeds = sorted(dict.fromkeys(seeds))[:limit]
    if len(seeds) != limit:
        raise SystemExit(f"Expected {limit} cleaned seeds for {csv_path.name} N={train_size}, got {len(seeds)}")
    return seeds


def write_seed_chunks(seeds: list[int], chunk_size: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for idx in range(0, len(seeds), chunk_size):
            chunk = seeds[idx : idx + chunk_size]
            handle.write(f"chunk{idx // chunk_size:03d}\t{','.join(str(seed) for seed in chunk)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write paired cleaned-seed chunks for external baselines.")
    parser.add_argument("--cleaned-csv", required=True, type=Path)
    parser.add_argument("--train-size", required=True, type=int)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    seeds = cleaned_vanilla_original_seeds(args.cleaned_csv, args.train_size, args.limit)
    write_seed_chunks(seeds, args.chunk_size, args.output)


if __name__ == "__main__":
    main()
