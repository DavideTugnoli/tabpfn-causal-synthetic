#!/usr/bin/env python3
"""Aggregate CI-preservation split outputs into paper-ready tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from conditional_independence_preservation import aggregate_results, compute_wilcoxon_table, write_markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--label", default="combined")
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root
    output_dir = args.output_dir or (run_root / "combined")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)

    result_files = sorted(run_root.glob("*/ci_preservation_results.csv"))
    if not result_files:
        raise FileNotFoundError(f"No ci_preservation_results.csv files found under {run_root}")

    frames = []
    skipped_files = []
    for path in result_files:
        try:
            df = pd.read_csv(path)
        except EmptyDataError:
            skipped_files.append({"path": str(path), "reason": "empty_csv"})
            continue
        if df.empty:
            skipped_files.append({"path": str(path), "reason": "empty_dataframe"})
            continue
        df.insert(0, "split_output_dir", path.parent.name)
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No readable non-empty ci_preservation_results.csv files found under {run_root}")
    combined = pd.concat(frames, ignore_index=True)
    combined_path = output_dir / f"{args.label}_ci_preservation_results.csv"
    combined.to_csv(combined_path, index=False)

    aggregate = aggregate_results(combined)
    aggregate.to_csv(output_dir / "tables" / f"{args.label}_ci_preservation_aggregate.csv", index=False)
    write_markdown_table(
        aggregate,
        output_dir / "tables" / f"{args.label}_ci_preservation_aggregate.md",
    )

    wilcoxon = compute_wilcoxon_table(combined, alpha=args.alpha)
    wilcoxon.to_csv(output_dir / "tables" / f"{args.label}_ci_preservation_wilcoxon.csv", index=False)
    write_markdown_table(
        wilcoxon,
        output_dir / "tables" / f"{args.label}_ci_preservation_wilcoxon.md",
    )

    status = (
        combined.groupby(["source", "dataset", "sample_size", "condition", "status"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    status.to_csv(output_dir / "tables" / f"{args.label}_ci_preservation_status.csv", index=False)
    if skipped_files:
        pd.DataFrame(skipped_files).to_csv(
            output_dir / "tables" / f"{args.label}_ci_preservation_skipped_files.csv",
            index=False,
        )

    summary_lines = [
        "# CI Preservation Aggregation",
        "",
        f"- Run root: `{run_root}`",
        f"- Result files: `{len(result_files)}`",
        f"- Skipped files: `{len(skipped_files)}`",
        f"- Rows: `{len(combined)}`",
        f"- Output: `{output_dir}`",
        "",
        "## Status Counts",
        "",
    ]
    try:
        summary_lines.append(
            combined.groupby("status", dropna=False).size().reset_index(name="n").to_markdown(index=False)
        )
    except Exception:
        summary_lines.append(combined.groupby("status", dropna=False).size().reset_index(name="n").to_csv(index=False))
    (output_dir / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Wrote combined results to {combined_path}")
    print(f"Rows: {len(combined)}")
    print(combined.groupby("status", dropna=False).size().reset_index(name="n").to_string(index=False))


if __name__ == "__main__":
    main()
