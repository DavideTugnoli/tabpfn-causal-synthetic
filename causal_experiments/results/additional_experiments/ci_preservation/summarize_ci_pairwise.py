#!/usr/bin/env python3
"""Summarize CI-preservation results for explicit paired condition contrasts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from conditional_independence_preservation import (
    ALPHA,
    hodges_lehmann_ci_from_diffs,
    holm_adjusted_pvalues,
    write_markdown_table,
)


def iqr_text(values: pd.Series) -> str:
    arr = values.dropna().to_numpy(float)
    if arr.size == 0:
        return ""
    return f"{np.nanpercentile(arr, 25):.4f}-{np.nanpercentile(arr, 75):.4f}"


def summarize_pairwise(
    df: pd.DataFrame,
    baseline: str,
    comparator: str,
    alpha: float,
) -> pd.DataFrame:
    valid = df[df["fraction_preserved"].notna()].copy()
    if "status" in valid:
        valid = valid[valid["status"].isin(["ok", "column_names_mismatch"])]
    records: list[dict[str, Any]] = []
    for (source, dataset, sample_size), group in valid.groupby(["source", "dataset", "sample_size"]):
        baseline_df = group[group["condition"] == baseline][
            ["repetition", "fraction_preserved", "n_reference_independent", "n_triples_total"]
        ].rename(columns={"fraction_preserved": "baseline_fraction"})
        comparator_df = group[group["condition"] == comparator][
            ["repetition", "fraction_preserved", "n_reference_independent", "n_triples_total"]
        ].rename(columns={"fraction_preserved": "comparator_fraction"})
        paired = pd.merge(baseline_df, comparator_df, on="repetition", how="inner").dropna()
        if paired.empty:
            continue
        diffs = paired["comparator_fraction"].to_numpy(float) - paired["baseline_fraction"].to_numpy(float)
        try:
            res = wilcoxon(diffs, zero_method="pratt", correction=False, alternative="two-sided", method="auto")
            statistic = float(res.statistic)
            p_value = float(res.pvalue)
        except ValueError:
            statistic = float("nan")
            p_value = 1.0
        hl, ci_lower, ci_upper = hodges_lehmann_ci_from_diffs(diffs, alpha=alpha)
        records.append(
            {
                "source": source,
                "dataset": dataset,
                "sample_size": int(sample_size),
                "baseline": baseline,
                "comparator": comparator,
                "n_pairs": int(len(paired)),
                "median_baseline": float(paired["baseline_fraction"].median()),
                "iqr_baseline": iqr_text(paired["baseline_fraction"]),
                "median_comparator": float(paired["comparator_fraction"].median()),
                "iqr_comparator": iqr_text(paired["comparator_fraction"]),
                "median_n_reference_independent": float(
                    paired["n_reference_independent_x"].replace(0, np.nan).median()
                ),
                "median_n_triples_total": float(paired["n_triples_total_x"].median()),
                "effect_hl": hl,
                "effect_ci_lower": ci_lower,
                "effect_ci_upper": ci_upper,
                "statistic": statistic,
                "p_value": p_value,
            }
        )
    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out
    out = out.sort_values(["source", "dataset", "sample_size"]).reset_index(drop=True)
    out["p_value_holm"] = np.nan
    out["holm_significant"] = False
    for _, idx in out.groupby(["source", "dataset"]).groups.items():
        adjusted = holm_adjusted_pvalues(out.loc[idx, "p_value"].to_numpy(float))
        out.loc[idx, "p_value_holm"] = adjusted
        out.loc[idx, "holm_significant"] = adjusted <= alpha
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="vanilla_original")
    parser.add_argument("--comparator", default="vanilla_topological")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args()

    paths = sorted(args.run_dir.glob("*/ci_preservation_results.csv"))
    if not paths and (args.run_dir / "ci_preservation_results.csv").exists():
        paths = [args.run_dir / "ci_preservation_results.csv"]
    if not paths:
        raise FileNotFoundError(f"No ci_preservation_results.csv files found under {args.run_dir}")
    df = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    out = summarize_pairwise(df, args.baseline, args.comparator, args.alpha)
    tables_dir = args.run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ci_preservation_pairwise_{args.comparator}_vs_{args.baseline}"
    out.to_csv(tables_dir / f"{stem}.csv", index=False)
    write_markdown_table(out, tables_dir / f"{stem}.md")
    print(f"Wrote {len(out)} rows to {tables_dir / f'{stem}.csv'}")


if __name__ == "__main__":
    main()
