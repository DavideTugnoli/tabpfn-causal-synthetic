#!/usr/bin/env python3
"""Validate and aggregate deterministic CI-preservation chunk outputs."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CI_SCRIPT = SCRIPT_DIR.parent / "conditional_independence_preservation.py"


def load_ci_module():
    spec = importlib.util.spec_from_file_location("ci_preservation_core", CI_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {CI_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-dir", type=Path, required=True)
    parser.add_argument("--expected-conditions", type=int, default=5)
    parser.add_argument("--expected-repetitions", type=int, default=100)
    args = parser.parse_args()

    result_paths = sorted((args.cell_dir / "chunks").glob("chunk_*/ci_preservation_results.csv"))
    if not result_paths:
        raise SystemExit(f"No chunk result CSVs found under {args.cell_dir}")

    results = pd.concat([pd.read_csv(path) for path in result_paths], ignore_index=True)
    key = ["source", "dataset", "condition", "sample_size", "repetition"]
    duplicates = results.duplicated(key, keep=False)
    if duplicates.any():
        raise SystemExit(f"Duplicate result keys found:\n{results.loc[duplicates, key].to_string(index=False)}")

    counts = (
        results.groupby(["source", "dataset", "condition", "sample_size"], dropna=False)
        .agg(n_rows=("repetition", "size"), n_unique_repetitions=("repetition", "nunique"))
        .reset_index()
    )
    invalid = counts[
        (counts["n_rows"] != args.expected_repetitions)
        | (counts["n_unique_repetitions"] != args.expected_repetitions)
    ]
    if len(counts) != args.expected_conditions or not invalid.empty:
        raise SystemExit(
            f"Expected {args.expected_conditions} conditions with {args.expected_repetitions} repetitions each.\n"
            f"Observed:\n{counts.to_string(index=False)}"
        )

    ci = load_ci_module()
    args.cell_dir.mkdir(parents=True, exist_ok=True)
    tables = args.cell_dir / "tables"
    tables.mkdir(exist_ok=True)
    results.to_csv(args.cell_dir / "ci_preservation_results.csv", index=False)
    counts.to_csv(tables / "chunk_aggregation_validation.csv", index=False)
    ci.aggregate_results(results).to_csv(tables / "ci_preservation_aggregate.csv", index=False)
    ci.compute_wilcoxon_table(results, alpha=ci.ALPHA).to_csv(
        tables / "ci_preservation_wilcoxon.csv", index=False
    )
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
