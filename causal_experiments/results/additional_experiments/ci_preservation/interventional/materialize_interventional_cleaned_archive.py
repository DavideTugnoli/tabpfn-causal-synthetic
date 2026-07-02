#!/usr/bin/env python3
"""Materialize an interventional cleaned-NPZ archive from cleaned CSV rows.

Only files referenced by the cleaned interventional CSV panels are materialized.
The script never deletes historical sources.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from validate_interventional_npz_metrics import (
    DEFAULT_CLEANED_DIR,
    DEFAULT_CODE_ROOT,
    DEFAULT_WORK_ROOT,
    DatasetPaths,
    build_dataset_paths,
    dataset_from_cleaned_path,
    synthetic_filenames,
)


DEFAULT_ARCHIVE_ROOT = (
    DEFAULT_WORK_ROOT / "data/cleaned_npz_archive/interventional/current_cleaned_20260429"
)


def copy_file(source: Path, destination: Path, mode: str) -> tuple[str, int]:
    if not source.exists():
        return "missing_source", 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = source.stat().st_size
    if destination.exists():
        if destination.stat().st_size == size:
            return "exists", size
        return "exists_size_mismatch", size
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "linked", size
        except OSError:
            shutil.copy2(source, destination)
            return "copied_after_link_failed", size
    shutil.copy2(source, destination)
    return "copied", size


def first_existing(candidates: list[Path]) -> Path | None:
    return next((candidate for candidate in candidates if candidate.exists()), None)


def synthetic_source(row: pd.Series, roots: tuple[Path, ...]) -> tuple[Path | None, str]:
    filenames = synthetic_filenames(row)
    candidates = [root / "datasets/synthetic" / filename for root in roots for filename in filenames]
    source = first_existing(candidates)
    return source, ",".join(filenames)


def train_source(row: pd.Series, roots: tuple[Path, ...]) -> Path | None:
    if isinstance(row.get("train_dataset_path"), str) and row["train_dataset_path"].strip():
        rel = Path(row["train_dataset_path"])
        candidates = [root / rel for root in roots]
    else:
        filename = f"train_ts{int(row['train_size'])}_s{int(row['seed'])}.npz"
        candidates = [root / "datasets" / filename for root in roots]
    return first_existing(candidates)


def global_test_source(roots: tuple[Path, ...]) -> Path | None:
    candidates = [
        root / "datasets/global_test_set.npz"
        for root in roots
    ]
    return first_existing(candidates)


def row_dataset(row: pd.Series, fallback: str) -> str:
    value = row.get("dataset_name", fallback)
    if pd.isna(value) or str(value).strip() == "":
        return fallback
    if str(value) == "simglucose_static_scm":
        return "simglucose"
    return str(value)


def build_manifest(dataset_paths: list[DatasetPaths], archive_root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    seen_files: set[tuple[str, str]] = set()

    for spec in dataset_paths:
        df = pd.read_csv(spec.cleaned_csv)
        df = df.copy()
        if "dataset_name" not in df.columns:
            df["dataset_name"] = spec.dataset
        cleaned_destination = f"cleaned_csv/{spec.cleaned_csv.name}"
        records.append(
            {
                "dataset": spec.dataset,
                "file_role": "cleaned_csv",
                "row_index": "",
                "condition": "",
                "train_size": "",
                "seed": "",
                "source_path": str(spec.cleaned_csv),
                "destination_relpath": cleaned_destination,
                "candidate_filenames": "",
                "source_found": spec.cleaned_csv.exists(),
            }
        )

        test_source = global_test_source(spec.roots)
        records.append(
            {
                "dataset": spec.dataset,
                "file_role": "global_test",
                "row_index": "",
                "condition": "",
                "train_size": "",
                "seed": "",
                "source_path": str(test_source) if test_source else "",
                "destination_relpath": f"datasets/{spec.dataset}/global_test_set.npz",
                "candidate_filenames": "global_test_set.npz",
                "source_found": test_source is not None,
            }
        )

        for row_index, row in df.iterrows():
            dataset = row_dataset(row, spec.dataset)
            condition = f"{row['algorithm']}_{row['column_order']}"
            train_size = int(row["train_size"])
            seed = int(row["seed"])

            synthetic, candidate_filenames = synthetic_source(row, spec.roots)
            synthetic_dest_name = synthetic.name if synthetic is not None else synthetic_filenames(row)[0]
            synthetic_rel = f"datasets/{dataset}/synthetic/{synthetic_dest_name}"
            synthetic_key = ("synthetic", synthetic_rel)
            if synthetic_key not in seen_files:
                seen_files.add(synthetic_key)
                records.append(
                    {
                        "dataset": dataset,
                        "file_role": "synthetic",
                        "row_index": row_index,
                        "condition": condition,
                        "train_size": train_size,
                        "seed": seed,
                        "source_path": str(synthetic) if synthetic else "",
                        "destination_relpath": synthetic_rel,
                        "candidate_filenames": candidate_filenames,
                        "source_found": synthetic is not None,
                    }
                )

            train = train_source(row, spec.roots)
            train_rel = f"datasets/{dataset}/train/train_ts{train_size}_s{seed}.npz"
            train_key = ("train", train_rel)
            if train_key not in seen_files:
                seen_files.add(train_key)
                records.append(
                    {
                        "dataset": dataset,
                        "file_role": "train",
                        "row_index": row_index,
                        "condition": condition,
                        "train_size": train_size,
                        "seed": seed,
                        "source_path": str(train) if train else "",
                        "destination_relpath": train_rel,
                        "candidate_filenames": f"train_ts{train_size}_s{seed}.npz",
                        "source_found": train is not None,
                    }
                )

    manifest = pd.DataFrame.from_records(records)
    manifest["destination_path"] = manifest["destination_relpath"].map(lambda rel: str(archive_root / str(rel)))
    return manifest


def write_summary(manifest: pd.DataFrame, report: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        manifest.groupby(["dataset", "file_role"], dropna=False)
        .agg(
            n=("source_found", "size"),
            source_found=("source_found", "sum"),
        )
        .reset_index()
    )
    copy_summary = report.groupby(["status"], dropna=False).size().reset_index(name="n")
    summary.to_csv(output_dir / "interventional_archive_manifest_summary.csv", index=False)
    copy_summary.to_csv(output_dir / "interventional_archive_copy_summary.csv", index=False)
    lines = [
        "# Interventional Cleaned NPZ Archive",
        "",
        f"Archive root: `{manifest.attrs.get('archive_root', '')}`",
        "",
        "## Manifest Summary",
        "",
    ]
    try:
        lines.append(summary.to_markdown(index=False))
    except Exception:
        lines.append(summary.to_csv(index=False))
    lines.extend(["", "## Copy Summary", ""])
    try:
        lines.append(copy_summary.to_markdown(index=False))
    except Exception:
        lines.append(copy_summary.to_csv(index=False))
    missing = manifest[~manifest["source_found"]]
    lines.extend(["", "## Missing Sources", ""])
    if missing.empty:
        lines.append("_None._")
    else:
        lines.append(missing[["dataset", "file_role", "condition", "train_size", "seed", "destination_relpath"]].to_csv(index=False))
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument("--cleaned-dir", type=Path, default=DEFAULT_CLEANED_DIR)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_paths = build_dataset_paths(args.work_root, args.code_root, args.cleaned_dir)
    manifest = build_manifest(dataset_paths, args.archive_root)
    manifest.attrs["archive_root"] = str(args.archive_root)

    output_dir = args.archive_root / "manifest"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "interventional_archive_file_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    report_rows = []
    for row in manifest.itertuples(index=False):
        source = Path(str(row.source_path)) if str(row.source_path) else Path("")
        destination = args.archive_root / str(row.destination_relpath)
        if not bool(row.source_found):
            status, size = "missing_source", 0
        elif args.execute:
            status, size = copy_file(source, destination, args.mode)
        else:
            status, size = "dry_run_ok", source.stat().st_size
        report_rows.append(
            {
                "dataset": row.dataset,
                "file_role": row.file_role,
                "source_path": str(source) if source else "",
                "destination_relpath": row.destination_relpath,
                "status": status,
                "size": size,
            }
        )

    report = pd.DataFrame.from_records(report_rows)
    report.to_csv(output_dir / "interventional_archive_materialization_report.csv", index=False)
    write_summary(manifest, report, output_dir)

    print("Manifest:", manifest_path)
    print(report.groupby("status").size().reset_index(name="n").to_string(index=False))
    if (manifest["source_found"] == False).any():  # noqa: E712
        raise SystemExit(2)


if __name__ == "__main__":
    main()
