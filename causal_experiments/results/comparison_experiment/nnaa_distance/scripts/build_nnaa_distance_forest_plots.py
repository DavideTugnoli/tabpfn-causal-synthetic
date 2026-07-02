#!/usr/bin/env python3
"""Build isolated forest plots for |NNAA - 0.5|.

The official CSVs store raw SynthEval NNAA. This script creates derived cleaned
CSV copies where the ``nnaa`` column is replaced by ``abs(nnaa - 0.5)`` and then
invokes the original comparison forest-plot script with
``--metrics k_marginal_tvd nnaa``. Requesting both metrics also produces the
combined two-panel ``forest_combined_*_2marginal_nnaa`` figures used in the
paper appendix; their kMTVD panels show the raw, unchanged kMTVD values. The
produced NNAA PDFs are then renamed with a ``_distance0p5`` suffix so a
transformed |NNAA - 0.5| figure can never be confused with a raw-NNAA plot if
copied into the paper by hand.

Nothing under the official forest-plot output folders is modified.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE.parent  # the nnaa_distance bundle root
COMPARISON_ROOT = OUTPUT_ROOT.parent
FOREST_SCRIPT = COMPARISON_ROOT / "forest_plots.py"
SOURCE_DATA_ROOT = COMPARISON_ROOT / "data"
REPO_ROOT = COMPARISON_ROOT.parents[2]


@dataclass(frozen=True)
class Bundle:
    name: str
    files: tuple[str, ...]


MAIN_FILES = (
    "result_csuite_large_backdoor_comparison_experiment_cleaned_reps_100.csv",
    "result_csuite_mixed_confounding_comparison_experiment_cleaned_reps_100.csv",
    "result_csuite_mixed_simpson_comparison_experiment_cleaned_reps_100.csv",
    "result_csuite_nonlin_simpson_comparison_experiment_cleaned_reps_100.csv",
    "result_csuite_symprod_simpson_comparison_experiment_cleaned_reps_100.csv",
    "result_csuite_weak_arrows_comparison_experiment_cleaned_reps_100.csv",
    "result_custom_scm_comparison_experiment_cleaned_reps_100.csv",
    "result_simglucose_comparison_experiment_cleaned_reps_100.csv",
)

NOISE_FILES = tuple(
    "result_custom_scm_noise1e-2_comparison_experiment_cleaned_reps_100.csv"
    if name == "result_custom_scm_comparison_experiment_cleaned_reps_100.csv"
    else name
    for name in MAIN_FILES
)

BUNDLES = (
    Bundle("main_cleaned", MAIN_FILES),
    Bundle("noise1e-2_cleaned", NOISE_FILES),
)

# Paper-figure pass: the published noise figures plot ONLY the noise Custom
# SCM, so the paper-ready noise distance plots must be generated from the
# noise file alone. This bundle is excluded from the raw-vs-distance summary
# (its dataset is already audited by the noise1e-2_cleaned bundle).
PAPER_NOISE_BUNDLE = Bundle(
    "noise1e-2_paper",
    ("result_custom_scm_noise1e-2_comparison_experiment_cleaned_reps_100.csv",),
)


def _require_existing(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required input files:\n{joined}")


def _derive_csvs(bundle: Bundle) -> list[Path]:
    input_dir = OUTPUT_ROOT / bundle.name / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    source_paths = [SOURCE_DATA_ROOT / name for name in bundle.files]
    _require_existing(source_paths)

    derived_paths: list[Path] = []
    manifest_rows: list[dict[str, object]] = []

    for source_path in source_paths:
        df = pd.read_csv(source_path)
        if "nnaa" not in df.columns:
            raise ValueError(f"Input CSV lacks 'nnaa': {source_path}")

        raw = pd.to_numeric(df["nnaa"], errors="coerce")
        transformed = (raw - 0.5).abs()
        out_df = df.copy()
        out_df["nnaa_raw_original"] = raw
        out_df["nnaa"] = transformed

        out_path = input_dir / source_path.name
        out_df.to_csv(out_path, index=False)
        derived_paths.append(out_path)

        manifest_rows.append(
            {
                "bundle": bundle.name,
                "file": source_path.name,
                "rows": int(len(out_df)),
                "raw_nnaa_min": float(raw.min(skipna=True)),
                "raw_nnaa_max": float(raw.max(skipna=True)),
                "distance_nnaa_min": float(transformed.min(skipna=True)),
                "distance_nnaa_max": float(transformed.max(skipna=True)),
            }
        )

    pd.DataFrame(manifest_rows).to_csv(OUTPUT_ROOT / bundle.name / "input_manifest.csv", index=False)
    return derived_paths


def _run_forest_script(bundle: Bundle, result_files: list[Path]) -> None:
    paper_root = OUTPUT_ROOT / bundle.name / "forest_plots" / "paper" / "comparison_experiment"
    stats_root = OUTPUT_ROOT / bundle.name / "comparison_results"
    log_dir = OUTPUT_ROOT / bundle.name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(FOREST_SCRIPT),
        "--metrics",
        "k_marginal_tvd",
        "nnaa",
        # usetex is active: bare "|" in text mode renders as an em dash, so
        # the bars and the minus sign are wrapped in math mode.
        "--nnaa-title",
        "$|$Nearest-Neighbor Adversarial Accuracy $-$ 0.5$|$",
        "--result-files",
        *[str(path) for path in result_files],
        "--paper-root",
        str(paper_root),
        "--comparison-results-root",
        str(stats_root),
    ]

    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    (log_dir / "forest_plots_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (log_dir / "forest_plots_command.json").write_text(
        json.dumps({"cmd": cmd, "returncode": completed.returncode}, indent=2),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Forest plot script failed for {bundle.name} with return code "
            f"{completed.returncode}. See {log_dir / 'forest_plots_stdout.log'}"
        )


def _append_distance_suffix(bundle: Bundle) -> int:
    """Insert the ``_distance0p5`` marker into the bundle's NNAA forest PDFs.

    The forest-plot script names these figures with the raw ``nnaa`` slug, so a
    transformed |NNAA - 0.5| figure is indistinguishable by filename from a raw
    NNAA figure. Renaming here makes the bundle output self-describing and keeps
    a transformed plot from being copied into the paper under a raw-looking name.
    Idempotent: files already carrying the marker are left untouched.
    """
    paper_root = OUTPUT_ROOT / bundle.name / "forest_plots" / "paper" / "comparison_experiment"
    renamed = 0
    for pdf in sorted(paper_root.glob("*/pdf/*nnaa*.pdf")):
        if "_distance0p5" in pdf.stem:
            continue
        pdf.rename(pdf.with_name(f"{pdf.stem}_distance0p5{pdf.suffix}"))
        renamed += 1
    return renamed


def _summarize_bundle(bundle: Bundle) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    original_stats_root = COMPARISON_ROOT / "comparison_results"
    distance_stats_root = OUTPUT_ROOT / bundle.name / "comparison_results"

    for distance_posthoc in sorted(distance_stats_root.glob("*/stat_tests/posthoc_wilcoxon_summary.csv")):
        dataset = distance_posthoc.parents[1].name
        original_posthoc = original_stats_root / dataset / "stat_tests" / "posthoc_wilcoxon_summary.csv"
        if not original_posthoc.exists():
            continue

        orig = pd.read_csv(original_posthoc)
        dist = pd.read_csv(distance_posthoc)
        orig = orig[orig["metric"].astype(str) == "nnaa"].copy()
        dist = dist[dist["metric"].astype(str) == "nnaa"].copy()

        key_cols = ["train_size", "condition_a", "condition_b"]
        merged = orig.merge(dist, on=key_cols, suffixes=("_raw", "_distance"))
        for _, rec in merged.iterrows():
            raw_sig = bool(rec.get("holm_significant_stepdown_raw", False)) or (
                pd.notna(rec.get("p_value_holm_raw")) and float(rec["p_value_holm_raw"]) <= 0.05
            )
            dist_sig = bool(rec.get("holm_significant_stepdown_distance", False)) or (
                pd.notna(rec.get("p_value_holm_distance")) and float(rec["p_value_holm_distance"]) <= 0.05
            )
            rows.append(
                {
                    "bundle": bundle.name,
                    "dataset": dataset,
                    "train_size": int(rec["train_size"]),
                    "condition_a": rec["condition_a"],
                    "condition_b": rec["condition_b"],
                    "raw_effect_hl": float(rec["effect_hl_raw"]),
                    "distance_effect_hl": float(rec["effect_hl_distance"]),
                    "hl_delta_distance_minus_raw": float(rec["effect_hl_distance"] - rec["effect_hl_raw"]),
                    "raw_p_holm": float(rec["p_value_holm_raw"]),
                    "distance_p_holm": float(rec["p_value_holm_distance"]),
                    "raw_significant": raw_sig,
                    "distance_significant": dist_sig,
                    "significance_changed": raw_sig != dist_sig,
                }
            )

    return pd.DataFrame(rows)


def _write_readme(summary: pd.DataFrame) -> None:
    readme = OUTPUT_ROOT / "README.md"
    if summary.empty:
        summary_text = "No comparison summary was produced."
    else:
        n_rows = len(summary)
        n_changed = int(summary["significance_changed"].sum())
        max_abs_delta = float(summary["hl_delta_distance_minus_raw"].abs().max())
        summary_text = (
            f"- Compared raw-NNAA vs |NNAA - 0.5| Wilcoxon summaries across {n_rows} "
            f"posthoc contrasts.\n"
            f"- Holm significance changes: {n_changed}.\n"
            f"- Max absolute HL delta (distance - raw, un-oriented posthoc scale): {max_abs_delta:.6g}.\n"
        )

    readme.write_text(
        "\n".join(
            [
                "# NNAA Distance-to-0.5 Forest Plots",
                "",
                "Date: 2026-06-03 (combined kMTVD+NNAA figures added 2026-06-11)",
                "",
                "This directory is an isolated robustness visualization. The official cleaned",
                "CSV files are not modified. Each input CSV here is a copy of a cleaned",
                "reps-100 CSV where `nnaa` has been replaced by `abs(nnaa - 0.5)` and the",
                "original value is retained as `nnaa_raw_original`.",
                "",
                "The forest plots are produced by invoking the original comparison",
                "`forest_plots.py` script with `--metrics k_marginal_tvd nnaa`,",
                "`--paper-root`, and `--comparison-results-root` pointing inside this",
                "analysis directory. Requesting both metrics also produces the combined",
                "two-panel `forest_combined_*_2marginal_nnaa_distance0p5.pdf` figures used",
                "in the paper appendix; their kMTVD panels show the raw, unchanged kMTVD",
                "values.",
                "",
                "NNAA panels are titled '|Nearest-Neighbor Adversarial Accuracy - 0.5|'",
                "via the forest-plot `--nnaa-title` option; the produced PDFs are then",
                "renamed with a `_distance0p5` suffix so a transformed figure is never",
                "mistaken for a raw-NNAA plot. Interpret every NNAA plot in this",
                "directory as distance from the ideal raw NNAA value 0.5, where lower is",
                "better.",
                "",
                "## Bundles",
                "",
                "- `main_cleaned`: original main cleaned comparison datasets.",
                "- `noise1e-2_cleaned`: same bundle but replacing `custom_scm` with",
                "  `custom_scm_noise1e-2`.",
                "",
                "## Raw-vs-distance summary",
                "",
                summary_text.rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only derive inputs and summaries; do not run forest_plots.py.",
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    derived_paper = _derive_csvs(PAPER_NOISE_BUNDLE)
    if not args.skip_plots:
        _run_forest_script(PAPER_NOISE_BUNDLE, derived_paper)
        _append_distance_suffix(PAPER_NOISE_BUNDLE)

    all_summaries: list[pd.DataFrame] = []
    for bundle in BUNDLES:
        result_files = _derive_csvs(bundle)
        if not args.skip_plots:
            _run_forest_script(bundle, result_files)
            _append_distance_suffix(bundle)
        bundle_summary = _summarize_bundle(bundle)
        if not bundle_summary.empty:
            all_summaries.append(bundle_summary)

    summary = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    summary_path = OUTPUT_ROOT / "raw_vs_distance_posthoc_comparison.csv"
    summary.to_csv(summary_path, index=False)
    _write_readme(summary)
    print(f"[OK] Wrote analysis bundle to {OUTPUT_ROOT}")
    print(f"[OK] Summary: {summary_path}")


if __name__ == "__main__":
    main()

