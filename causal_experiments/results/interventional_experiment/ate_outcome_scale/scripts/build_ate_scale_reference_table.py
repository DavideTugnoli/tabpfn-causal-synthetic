#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "tables" / "primary_scale_reference_main_all_train_sizes.csv"
STAT_TESTS_ROOT = ROOT.parent / "stat_tests"
OUTPUT_DIR = ROOT / "tables" / "full_reference"

DATASET_ACRONYMS = {
    "Custom SCM": "CSM",
    "CSuite mixed confounding": "CMC",
    "CSuite nonlinear Simpson": "CNS",
    "CSuite symprod Simpson": "CSS",
    "CSuite mixed Simpson": "CMS",
    "CSuite weak arrows": "CWA",
    "CSuite large backdoor": "CLB",
    "SimGlucose": "SGL",
}

DATASET_SLUGS = {
    "Custom SCM": "custom_scm",
    "CSuite mixed confounding": "csuite_mixed_confounding",
    "CSuite nonlinear Simpson": "csuite_nonlin_simpson",
    "CSuite symprod Simpson": "csuite_symprod_simpson",
    "CSuite mixed Simpson": "csuite_mixed_simpson",
    "CSuite weak arrows": "csuite_weak_arrows",
    "CSuite large backdoor": "csuite_large_backdoor",
    "SimGlucose": "simglucose",
}

COMPARATOR_CONDITIONS = {
    "DAG-aware": "dag_topological",
    "Vanilla topological": "vanilla_topological",
}

DATASET_ORDER = ["CSM", "CMC", "CNS", "CSS", "CMS", "CWA", "CLB", "SGL"]
TRAIN_SIZE_ORDER = [20, 50, 100, 200, 500, 1000]
COMPACT_DATASETS = ["CSM", "CMC", "SGL"]
COMPACT_TRAIN_SIZES = [20, 100, 500]


def median_from_ci_cell(value: object) -> float:
    return float(str(value).split("[", maxsplit=1)[0].strip())


def format_value(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3g}"


def format_percent(value: float) -> str:
    return f"{value:.1f}"


def load_paired_stats() -> pd.DataFrame:
    frames = []
    for dataset_slug in DATASET_SLUGS.values():
        path = STAT_TESTS_ROOT / dataset_slug / "posthoc_wilcoxon_summary.csv"
        stats = pd.read_csv(path)
        stats = stats[
            (stats["metric"] == "ate_difference")
            & (stats["condition_a"] == "vanilla_original")
        ].copy()
        frames.append(stats)
    return pd.concat(frames, ignore_index=True)


def build_table() -> pd.DataFrame:
    df = pd.read_csv(INPUT)
    stats = load_paired_stats()

    rows = []
    for _, row in df.iterrows():
        dataset_name = str(row["Dataset"])
        comparator_label = str(row["Comparator"])
        dataset_slug = DATASET_SLUGS[dataset_name]
        comparator_condition = COMPARATOR_CONDITIONS[comparator_label]
        train_size = int(row["N"])
        stat_match = stats[
            (stats["dataset"] == dataset_slug)
            & (stats["train_size"].astype(int) == train_size)
            & (stats["condition_b"] == comparator_condition)
        ]
        if len(stat_match) != 1:
            raise RuntimeError(
                f"Expected one paired stat row for {dataset_slug}, "
                f"N={train_size}, comparator={comparator_condition}; got {len(stat_match)}"
            )
        stat_row = stat_match.iloc[0]
        rows.append(
            {
                "Dataset": DATASET_ACRONYMS[dataset_name],
                "N": train_size,
                "Units": row["Outcome units"],
                "Abs. ATE_test": float(row["|ATE_test|"]),
                "Vanilla median abs. ATE error": median_from_ci_cell(
                    row["Vanilla |Delta_ATE| median [95% CI]"]
                ),
                "Comparator": comparator_label,
                "Comparator median abs. ATE error": median_from_ci_cell(
                    row["Comparator |Delta_ATE| median [95% CI]"]
                ),
                "HL diff (vanilla - comparator)": float(stat_row["effect_hl"]),
                "Significant": "Y" if bool(stat_row["holm_significant_stepdown"]) else "N",
            }
        )

    out = pd.DataFrame(
        rows
    )
    out["Dataset"] = pd.Categorical(out["Dataset"], DATASET_ORDER, ordered=True)
    out["N"] = pd.Categorical(out["N"], TRAIN_SIZE_ORDER, ordered=True)
    return out.sort_values(["Dataset", "N"]).reset_index(drop=True)


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for column in [
        "Abs. ATE_test",
        "Vanilla median abs. ATE error",
        "Comparator median abs. ATE error",
        "HL diff (vanilla - comparator)",
    ]:
        display[column] = display[column].astype(float).map(format_value)
    return display


def write_markdown(path: Path, df: pd.DataFrame) -> None:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, df: pd.DataFrame) -> None:
    def esc(value: object) -> str:
        return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")

    columns = list(df.columns)
    spec = "lrlrrlrrc"
    lines = [
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(esc(column) for column in columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(esc(row[column]) for column in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full = build_table()
    compact = full[
        full["Dataset"].astype(str).isin(COMPACT_DATASETS)
        & full["N"].astype(int).isin(COMPACT_TRAIN_SIZES)
    ].copy()

    full.to_csv(OUTPUT_DIR / "ate_scale_reference_full.csv", index=False)
    compact.to_csv(OUTPUT_DIR / "ate_scale_reference_compact.csv", index=False)

    full_display = display_table(full)
    compact_display = display_table(compact)
    write_markdown(OUTPUT_DIR / "ate_scale_reference_full.md", full_display)
    write_markdown(OUTPUT_DIR / "ate_scale_reference_compact.md", compact_display)
    write_latex(OUTPUT_DIR / "ate_scale_reference_full.tex", full_display)
    write_latex(OUTPUT_DIR / "ate_scale_reference_compact.tex", compact_display)


if __name__ == "__main__":
    main()
