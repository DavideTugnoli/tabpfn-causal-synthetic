#!/usr/bin/env python3
"""Prepare a symlinked NPZ input tree for CI-preservation analysis.

The tree mirrors the original experiment layout expected by
conditional_independence_preservation.py, but it contains only the seeds present
in the cleaned comparison CSVs. This keeps the CI analysis paired with the
published results while allowing cpdag_discovered synthetic data to come
from the parser-fix rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_WORK_ROOT = Path("/path/to/work_root")
DEFAULT_ORIGINAL_ROOT = DEFAULT_WORK_ROOT / "data/original_tabpfn_extensions"
DEFAULT_PARSER_FIX_ROOT = (
    DEFAULT_WORK_ROOT
    / "runs/cpdag_discovered_parser_fix_train_size_split/code/tabpfn-causal-synthetic"
)
DEFAULT_RECOVERY_ROOT = DEFAULT_WORK_ROOT / "runs/demetra_recovered_cpdag_synthetic_npz_20260428"
DEFAULT_MISSING_RECOVERY_ROOT = (
    DEFAULT_WORK_ROOT
    / "runs/ci_preservation_missing_npz_recovery_20260428/code/tabpfn-causal-synthetic"
)
DEFAULT_CLEANED_ROOT = (
    DEFAULT_WORK_ROOT
    / "code/tabpfn-causal-synthetic/causal_experiments/results/comparison_experiment/data"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_WORK_ROOT / "runs/ci_preservation_inputs_20260428/data"

DEFAULT_DATASETS = (
    "custom_scm",
    "csuite_large_backdoor",
    "csuite_mixed_confounding",
    "csuite_mixed_simpson",
    "csuite_nonlin_simpson",
    "csuite_symprod_simpson",
    "csuite_weak_arrows",
)

DEFAULT_CONDITIONS = (
    "vanilla_original",
    "vanilla_topological",
    "vanilla_reverse_topological",
    "dag_topological",
    "cpdag_minimal_original",
    "cpdag_discovered_original",
)

CONDITION_CANDIDATES = {
    "vanilla_original": ("vanilla_original",),
    "vanilla_topological": ("vanilla_topological",),
    "vanilla_reverse_topological": ("vanilla_reverse_topological", "vanilla_worst"),
    "dag_topological": ("dag_topological",),
    "cpdag_minimal_original": (
        "cpdag_minimal_original",
        "cpdag_v1_both_vanilla_minimal_first_original",
    ),
    "cpdag_discovered_original": (
        "cpdag_discovered_original",
        "cpdag_v1_both_vanilla_discovered_first_original",
    ),
}


def parse_csv_set(value: str | None, default: Iterable[str]) -> list[str]:
    if value is None or not value.strip():
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def dataset_result_relpath(dataset: str) -> Path:
    if dataset == "custom_scm":
        return Path("causal_experiments/custom_scm_experiment/comparison_experiment/results")
    if dataset.startswith("csuite_"):
        return Path("causal_experiments/csuite_experiment/comparison_experiment_csuite/results") / dataset
    raise ValueError(f"Unsupported dataset for CI input preparation: {dataset}")


def cleaned_csv_path(cleaned_root: Path, dataset: str) -> Path:
    return cleaned_root / f"result_{dataset}_comparison_experiment_cleaned_reps_100.csv"


def normalized_condition(row: pd.Series) -> str:
    algorithm = str(row["algorithm"])
    column_order = str(row["column_order"])
    return f"{algorithm}_{column_order}"


def synthetic_filename(condition: str, sample_size: int, seed: int) -> str:
    return f"synthetic_{condition}_ts{sample_size}_s{seed}.npz"


def train_filename(sample_size: int, seed: int) -> str:
    return f"train_ts{sample_size}_s{seed}.npz"


def is_valid_npz(path: Path, key: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path, allow_pickle=True) as archive:
            if key not in archive:
                return False
            data = np.asarray(archive[key])
            return data.ndim == 2 and data.shape[0] > 0 and data.shape[1] > 0
    except Exception:
        return False


def replace_symlink(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    os.symlink(source, target)


def candidate_dataset_roots(dataset: str, original_root: Path, parser_fix_root: Path) -> dict[str, Path]:
    rel = dataset_result_relpath(dataset)
    return {
        "original": original_root / rel,
        "parser_fix": parser_fix_root / rel,
    }


def synthetic_candidates(
    dataset: str,
    condition: str,
    sample_size: int,
    seed: int,
    roots: dict[str, Path],
    recovery_root: Path,
    missing_recovery_root: Path,
) -> list[tuple[str, Path]]:
    names = CONDITION_CANDIDATES[condition]
    candidates: list[tuple[str, Path]] = []
    missing_recovery_dataset_root = missing_recovery_root / dataset_result_relpath(dataset)
    for raw_condition in names:
        candidates.append(
            (
                "missing_npz_recovery",
                missing_recovery_dataset_root
                / "datasets/synthetic"
                / synthetic_filename(raw_condition, sample_size, seed),
            )
        )

    if condition == "cpdag_discovered_original" and dataset.startswith("csuite_"):
        recovery_dataset_root = recovery_root / dataset
        for raw_condition in names:
            candidates.append(
                (
                    "demetra_v100_recovery",
                    recovery_dataset_root
                    / "datasets/synthetic"
                    / synthetic_filename(raw_condition, sample_size, seed),
                )
            )

    root_order = ("parser_fix", "original") if condition == "cpdag_discovered_original" else ("original", "parser_fix")
    for source_name in root_order:
        root = roots[source_name]
        for raw_condition in names:
            candidates.append(
                (
                    source_name,
                    root / "datasets/synthetic" / synthetic_filename(raw_condition, sample_size, seed),
                )
            )
    return candidates


def prepare_inputs(args: argparse.Namespace) -> pd.DataFrame:
    datasets = parse_csv_set(args.datasets, DEFAULT_DATASETS)
    conditions = set(parse_csv_set(args.conditions, DEFAULT_CONDITIONS))
    records: list[dict[str, object]] = []

    if args.output_root.exists() and args.replace:
        shutil.rmtree(args.output_root)

    for dataset in datasets:
        csv_path = cleaned_csv_path(args.cleaned_csv_root, dataset)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing cleaned CSV for {dataset}: {csv_path}")

        cleaned = pd.read_csv(csv_path)
        cleaned["condition"] = cleaned.apply(normalized_condition, axis=1)
        cleaned = cleaned[cleaned["condition"].isin(conditions)].copy()
        roots = candidate_dataset_roots(dataset, args.original_root, args.parser_fix_root)

        output_dataset_root = args.output_root / "original_tabpfn_extensions" / dataset_result_relpath(dataset)
        global_candidates = [
            roots["original"] / "datasets/global_test_set.npz",
            roots["parser_fix"] / "datasets/global_test_set.npz",
        ]
        global_source = next((path for path in global_candidates if is_valid_npz(path, "X_test")), None)
        if global_source is None:
            raise FileNotFoundError(f"No valid global_test_set.npz for {dataset}")
        replace_symlink(output_dataset_root / "datasets/global_test_set.npz", global_source)

        for _, row in cleaned.iterrows():
            condition = str(row["condition"])
            sample_size = int(row["train_size"])
            seed = int(row["seed"])

            input_candidates = [
                roots["original"] / "datasets" / train_filename(sample_size, seed),
                roots["parser_fix"] / "datasets" / train_filename(sample_size, seed),
            ]
            input_source = next((path for path in input_candidates if is_valid_npz(path, "X_train")), None)
            if input_source is None:
                raise FileNotFoundError(f"No valid train NPZ for {dataset} ts={sample_size} seed={seed}")
            replace_symlink(output_dataset_root / "datasets" / input_source.name, input_source)

            selected_source = None
            selected_path = None
            for source_name, candidate in synthetic_candidates(
                dataset,
                condition,
                sample_size,
                seed,
                roots,
                args.recovery_root,
                args.missing_recovery_root,
            ):
                if is_valid_npz(candidate, "synthetic_data"):
                    selected_source = source_name
                    selected_path = candidate
                    break

            if selected_path is None or selected_source is None:
                attempted = [
                    str(path)
                    for _, path in synthetic_candidates(
                        dataset,
                        condition,
                        sample_size,
                        seed,
                        roots,
                        args.recovery_root,
                        args.missing_recovery_root,
                    )
                ]
                raise FileNotFoundError(
                    f"No valid synthetic NPZ for {dataset} {condition} ts={sample_size} seed={seed}; "
                    f"attempted={attempted}"
                )

            target_name = synthetic_filename(condition, sample_size, seed)
            replace_symlink(output_dataset_root / "datasets/synthetic" / target_name, selected_path)
            records.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "train_size": sample_size,
                    "seed": seed,
                    "synthetic_source": selected_source,
                    "synthetic_path": str(selected_path),
                    "target_path": str(output_dataset_root / "datasets/synthetic" / target_name),
                    "input_path": str(input_source),
                }
            )

    manifest = pd.DataFrame.from_records(records)
    manifest_path = args.output_root / "ci_preservation_input_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    summary = (
        manifest.groupby(["dataset", "condition", "synthetic_source"], dropna=False)
        .size()
        .reset_index(name="n_files")
    )
    summary.to_csv(args.output_root / "ci_preservation_input_summary.csv", index=False)

    metadata = {
        "cleaned_csv_root": str(args.cleaned_csv_root),
        "original_root": str(args.original_root),
        "parser_fix_root": str(args.parser_fix_root),
        "recovery_root": str(args.recovery_root),
        "missing_recovery_root": str(args.missing_recovery_root),
        "output_root": str(args.output_root),
        "datasets": datasets,
        "conditions": sorted(conditions),
    }
    (args.output_root / "ci_preservation_input_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-csv-root", type=Path, default=DEFAULT_CLEANED_ROOT)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--parser-fix-root", type=Path, default=DEFAULT_PARSER_FIX_ROOT)
    parser.add_argument("--recovery-root", type=Path, default=DEFAULT_RECOVERY_ROOT)
    parser.add_argument("--missing-recovery-root", type=Path, default=DEFAULT_MISSING_RECOVERY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--replace", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = prepare_inputs(args)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
