#!/usr/bin/env python3
"""Compare the parser-fix spurious-correlation table with the historical table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
NEW = HERE / "outputs/spurious_correlations_summary.csv"
OLD = HERE.parent / "noise1e-2/spurious_correlations_summary_noise1e-2.csv"

KEYS = ["algorithm", "column_order", "train_size"]
METRICS = {
    "mean_corr_X0_X3": "mean_synthetic_corr_X0_X3",
    "std_corr_X0_X3": "std_synthetic_corr_X0_X3",
    "mean_corr_X0_X2": "mean_synthetic_corr_X0_X2",
    "std_corr_X0_X2": "std_synthetic_corr_X0_X2",
}


def main() -> None:
    new = pd.read_csv(NEW)
    old = pd.read_csv(OLD)
    old = old[KEYS + list(METRICS.values())].copy()
    old = old.rename(columns={old_name: f"{new_name}_old" for new_name, old_name in METRICS.items()})
    new = new.rename(columns={name: f"{name}_new" for name in METRICS})
    comparison = old.merge(new, on=KEYS, validate="one_to_one")

    for metric in METRICS:
        comparison[f"{metric}_delta"] = comparison[f"{metric}_new"] - comparison[f"{metric}_old"]
        comparison[f"{metric}_changed_3dp"] = (
            comparison[f"{metric}_new"].round(3) != comparison[f"{metric}_old"].round(3)
        )

    comparison.to_csv(HERE / "published_vs_parser_fix.csv", index=False)
    changed = comparison[
        comparison[[f"{metric}_changed_3dp" for metric in METRICS]].any(axis=1)
    ].copy()
    changed.to_csv(HERE / "changed_rows_at_paper_precision.csv", index=False)
    changed_above_tolerance = comparison[
        comparison[[f"{metric}_delta" for metric in METRICS]]
        .abs()
        .gt(1e-7)
        .any(axis=1)
    ].copy()
    changed_above_tolerance.to_csv(
        HERE / "changed_rows_above_numeric_tolerance.csv", index=False
    )

    lines = [
        "| N | Pair | Historical mean (std) | Parser-fix mean (std) | Mean delta |",
        "|---:|---|---:|---:|---:|",
    ]
    discovered = comparison[comparison["algorithm"] == "cpdag_discovered"].sort_values("train_size")
    for row in discovered.itertuples(index=False):
        for pair in ["X0_X3", "X0_X2"]:
            old_mean = getattr(row, f"mean_corr_{pair}_old")
            old_std = getattr(row, f"std_corr_{pair}_old")
            new_mean = getattr(row, f"mean_corr_{pair}_new")
            new_std = getattr(row, f"std_corr_{pair}_new")
            delta = getattr(row, f"mean_corr_{pair}_delta")
            lines.append(
                f"| {int(row.train_size)} | {pair.replace('_', '-')} | "
                f"{old_mean:.3f} ({old_std:.3f}) | {new_mean:.3f} ({new_std:.3f}) | "
                f"{delta:+.6f} |"
            )
    (HERE / "discovered_cpdag_old_vs_new.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Rows compared: {len(comparison)}")
    print(f"Rows changed at paper precision: {len(changed)}")
    print(f"Rows changed above 1e-7 numeric tolerance: {len(changed_above_tolerance)}")
    print(changed[KEYS].to_string(index=False))


if __name__ == "__main__":
    main()
