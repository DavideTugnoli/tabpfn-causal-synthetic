#!/usr/bin/env python3
"""Build the camera-ready CPDAG discovery table and published comparison."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
NEW_SUMMARY = HERE / "outputs/discovery_metrics_summary.csv"
PUBLISHED_SUMMARY = (
    HERE.parent
    / "scripts/outputs/cpdag_discovery_metrics/discovery_metrics_summary.csv"
)

LABELS = {
    "custom_scm": "CSM",
    "csuite_large_backdoor": "CLB",
    "csuite_mixed_confounding": "CMC",
    "csuite_mixed_simpson": "CMS",
    "csuite_nonlin_simpson": "CNS",
    "csuite_symprod_simpson": "CSS",
    "csuite_weak_arrows": "CWA",
}
ORDER = {label: index for index, label in enumerate(LABELS.values())}
METRICS = [
    "skeleton_recall",
    "direction_recall",
    "oriented_fraction",
    "direction_precision",
]


def prepare(path: Path, suffix: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data[data["dataset"].isin(LABELS)].copy()
    data["label"] = data["dataset"].map(LABELS)
    data["oriented_fraction"] = data["directed_edges_mean"] / data["skeleton_edges_mean"]
    data = data.rename(
        columns={
            "skeleton_recall_mean": "skeleton_recall",
            "direction_recall_mean": "direction_recall",
            "direction_precision_mean": "direction_precision",
        }
    )
    data["label_order"] = data["label"].map(ORDER)
    data = data.sort_values(["label_order", "train_size"])
    selected = data[
        ["dataset", "label", "train_size", *METRICS, "runs", "runs_with_directed"]
    ].copy()
    return selected.rename(
        columns={column: f"{column}_{suffix}" for column in [*METRICS, "runs", "runs_with_directed"]}
    )


def render(value: float) -> str:
    from decimal import Decimal, ROUND_HALF_UP
    if pd.isna(value):
        return "--"
    # round-half-up on the decimal value; avoids the binary-float artifact where
    # f"{0.295:.2f}" == "0.29" (0.295 -> 0.30, 0.145 -> 0.15, 0.125 -> 0.13)
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def main() -> None:
    new = prepare(NEW_SUMMARY, "new")
    published = prepare(PUBLISHED_SUMMARY, "published")
    comparison = published.merge(
        new,
        on=["dataset", "label", "train_size"],
        validate="one_to_one",
    )
    for metric in METRICS:
        comparison[f"{metric}_delta"] = (
            comparison[f"{metric}_new"] - comparison[f"{metric}_published"]
        )
        comparison[f"{metric}_published_2dp"] = comparison[f"{metric}_published"].round(2)
        comparison[f"{metric}_new_2dp"] = comparison[f"{metric}_new"].round(2)
        comparison[f"{metric}_changed_2dp"] = (
            comparison[f"{metric}_published_2dp"].fillna(-999)
            != comparison[f"{metric}_new_2dp"].fillna(-999)
        )
    comparison.to_csv(HERE / "published_vs_parser_fix.csv", index=False)

    table = new.rename(
        columns={
            "skeleton_recall_new": "skeleton_recall",
            "direction_recall_new": "direction_recall",
            "oriented_fraction_new": "oriented_fraction",
            "direction_precision_new": "direction_precision",
            "runs_new": "runs",
            "runs_with_directed_new": "runs_with_directed",
        }
    )
    table.to_csv(HERE / "pc_discovery_metrics_parser_fix.csv", index=False)

    md = [
        "| Dataset | N | Skel. rec. | Dir. rec. | Orient. frac. | Dir. prec. | Runs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    tex = []
    for row in table.itertuples(index=False):
        md.append(
            f"| {row.label} | {int(row.train_size)} | {render(row.skeleton_recall)} | "
            f"{render(row.direction_recall)} | {render(row.oriented_fraction)} | "
            f"{render(row.direction_precision)} | {int(row.runs)} |"
        )
        tex.append(
            f"  & {int(row.train_size):3d} & {render(row.skeleton_recall)} & "
            f"{render(row.direction_recall)} & {render(row.oriented_fraction)} & "
            f"{render(row.direction_precision)} \\\\"
        )
    (HERE / "pc_discovery_metrics_parser_fix.md").write_text("\n".join(md) + "\n")
    (HERE / "pc_discovery_metrics_parser_fix_rows.tex").write_text("\n".join(tex) + "\n")

    changed_exact = comparison[
        comparison[[f"{metric}_delta" for metric in METRICS]]
        .abs()
        .fillna(0)
        .gt(1e-12)
        .any(axis=1)
    ]
    changed_rounded = comparison[
        comparison[[f"{metric}_changed_2dp" for metric in METRICS]].any(axis=1)
    ]
    changed_exact.to_csv(HERE / "changed_cells_exact.csv", index=False)
    changed_rounded.to_csv(HERE / "changed_cells_at_paper_precision.csv", index=False)

    print(f"Table cells: {len(table)}; all have 100 runs: {table['runs'].eq(100).all()}")
    print(f"Cells changed exactly: {len(changed_exact)}")
    print(f"Cells changed at paper precision: {len(changed_rounded)}")
    print(
        changed_rounded[
            [
                "label",
                "train_size",
                *[
                    column
                    for metric in METRICS
                    for column in [f"{metric}_published_2dp", f"{metric}_new_2dp"]
                ],
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
