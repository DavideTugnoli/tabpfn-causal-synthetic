#!/usr/bin/env python3
"""Build the NNAA test-train (privacy-loss) forest plots.

The paper's proposed NNAA privacy-loss control reports the difference between
NNAA on the held-out test set and NNAA on the training set --- the "privacy
loss" of Yale et al. (2020) --- as Hodges-Lehmann estimates of vanilla minus
each method. The original one-off producer for these two figures was never
committed to the repo, so this script reproduces them from the public cleaned
CSVs, which carry both ``nnaa`` (test-set NNAA) and ``nnaa_train`` (train-set
NNAA) columns.

It mirrors ``build_nnaa_distance_forest_plots.py``: it writes derived CSV copies
where the ``nnaa`` column is replaced by ``nnaa - nnaa_train`` (the raw test-set
value is retained as ``nnaa_test_original``) and then invokes the comparison
``forest_plots.py`` with ``--metrics nnaa`` and a ``NNAA (test - train)`` title.
Requesting the DAG, oracle-PDAG and discovered-CPDAG comparisons produces the
combined ``forest_combined_dag_and_cpdag_minimal_nnaa`` figure and the discovered
single-panel figure. The produced NNAA PDFs are renamed with a
``_nnaa_test_train_diff`` slug to match the paper's figure filenames and so a
transformed privacy-loss plot can never be confused with a raw-NNAA plot.

Datasets that lack a ``nnaa_train`` column (e.g. simglucose) are skipped.
Nothing under the official forest-plot output folders is modified.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
COMPARISON_ROOT = HERE.parents[1]  # .../results/comparison_experiment
FOREST_SCRIPT = COMPARISON_ROOT / "forest_plots.py"
SOURCE_DATA_ROOT = COMPARISON_ROOT / "data"
REPO_ROOT = COMPARISON_ROOT.parents[2]
OUTPUT_ROOT = COMPARISON_ROOT / "nnaa_test_train"

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

# Comparisons needed for the two proposed figures: the combined DAG + oracle-PDAG
# panel and the discovered-CPDAG single panel. (usetex: the minus in the title is
# wrapped in math mode; a bare "-" renders as a hyphen and a bare "|" as an em dash.)
COMPARISONS = (
    "cross_dag_topological_vs_vanilla_original",
    "original_cpdag_minimal_vs_vanilla",
    "original_cpdag_discovered_vs_vanilla",
)
NNAA_TITLE = r"NNAA (test $-$ train)"


def _derive_csvs(input_dir: Path) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    derived: list[Path] = []
    for name in MAIN_FILES:
        df = pd.read_csv(SOURCE_DATA_ROOT / name)
        if "nnaa" not in df.columns or "nnaa_train" not in df.columns:
            print(f"[skip] {name}: missing 'nnaa'/'nnaa_train'")
            continue
        train = pd.to_numeric(df["nnaa_train"], errors="coerce")
        if train.notna().sum() == 0:
            print(f"[skip] {name}: 'nnaa_train' all-NaN")
            continue
        test = pd.to_numeric(df["nnaa"], errors="coerce")
        out = df.copy()
        out["nnaa_test_original"] = test
        out["nnaa"] = test - train
        out_path = input_dir / name
        out.to_csv(out_path, index=False)
        derived.append(out_path)
        print(f"[ok]   {name}: nnaa <- nnaa(test) - nnaa_train")
    if not derived:
        raise RuntimeError("No datasets with an 'nnaa_train' column were found.")
    return derived


def _run_forest_script(result_files: list[Path], paper_root: Path, stats_root: Path) -> None:
    cmd = [
        sys.executable,
        str(FOREST_SCRIPT),
        "--no-csv",
        "--metrics",
        "nnaa",
        "--nnaa-title",
        NNAA_TITLE,
        "--comparisons",
        *COMPARISONS,
        "--result-files",
        *[str(path) for path in result_files],
        "--paper-root",
        str(paper_root),
        "--comparison-results-root",
        str(stats_root),
    ]
    completed = subprocess.run(
        cmd, cwd=str(REPO_ROOT), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"forest_plots.py failed with return code {completed.returncode}.")


def _append_test_train_suffix(paper_root: Path) -> int:
    """Rename produced NNAA PDFs from the raw ``_nnaa`` slug to
    ``_nnaa_test_train_diff`` so a transformed privacy-loss figure is never
    mistaken for a raw-NNAA plot. Idempotent."""
    renamed = 0
    for pdf in sorted(paper_root.glob("*/*nnaa*.pdf")):
        if "_nnaa_test_train_diff" in pdf.stem:
            continue
        if pdf.stem.endswith("_nnaa"):
            new_stem = pdf.stem[: -len("_nnaa")] + "_nnaa_test_train_diff"
        else:
            new_stem = pdf.stem.replace("_nnaa", "_nnaa_test_train_diff")
        pdf.rename(pdf.with_name(f"{new_stem}{pdf.suffix}"))
        renamed += 1
    return renamed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only derive the transformed input CSVs; do not run forest_plots.py.",
    )
    args = parser.parse_args()

    input_dir = OUTPUT_ROOT / "inputs"
    paper_root = OUTPUT_ROOT / "forest_plots" / "paper" / "comparison_experiment"
    stats_root = OUTPUT_ROOT / "comparison_results"

    derived = _derive_csvs(input_dir)
    if not args.skip_plots:
        _run_forest_script(derived, paper_root, stats_root)
        renamed = _append_test_train_suffix(paper_root)
        print(f"[OK] Renamed {renamed} NNAA PDF(s) with the _nnaa_test_train_diff slug.")
    print(f"[OK] NNAA test-train bundle written to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
