#!/usr/bin/env python3
"""Prepare REX (boot20, no-HPO) task files for ANM CSuite datasets.

This utility supports two task sources:
1) cleaned-csv: use (train_size, seed) pairs from cleaned comparison CSVs.
2) all-npz: scan every available NPZ and deduplicate duplicate path variants.

Outputs:
- Task TSV: dataset, seed, train_size, npz_path, out_dir
- Manifest TSV: task TSV + expected_json path
- Count report (per dataset/train_size)
- Optional chunked task files for array submission
"""

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import re


PROJECT_ROOT = Path(os.environ.get("CAUSALEXPLAIN_ROOT", "/path/to/causalexplain"))
CSV_ROOT = PROJECT_ROOT / "folder_ex_eperiments" / "csuite_experiment" / "comparison_experiment_csuite"
RESULTS_ROOT = CSV_ROOT / "results"
SLURM_DIR = PROJECT_ROOT / "slurm"
NOTES_DIR = PROJECT_ROOT / "notes"
OUT_ROOT_BASE = PROJECT_ROOT / "results" / "rex_boot20_nohpo_json"

# ANM selection follows CSuite README table ("Additive noise model? = Y")
ANM_DATASETS = {
    "csuite_nonlin_simpson",
    "csuite_symprod_simpson",
    "csuite_large_backdoor",
    "csuite_weak_arrows",
}

TRAIN_FILE_RE = re.compile(r"train_ts(\d+)_s(\d+)\.npz$")
CSV_PREFIX = "result_"
CSV_SUFFIX = "_comparison_experiment_cleaned_reps_100.csv"


TaskKey = Tuple[str, int, int]  # (dataset, train_size, seed)
TaskRow = Tuple[str, int, int, str, str]  # dataset, seed, train_size, npz, out_dir


def _dataset_from_csv_name(csv_name: str) -> str:
    if not csv_name.startswith(CSV_PREFIX) or not csv_name.endswith(CSV_SUFFIX):
        raise ValueError(f"Unexpected cleaned CSV name: {csv_name}")
    return csv_name[len(CSV_PREFIX) : -len(CSV_SUFFIX)]


def _extract_dataset_from_npz(npz_path: Path) -> Optional[str]:
    parts = npz_path.parts
    if "datasets" not in parts:
        return None
    idx = parts.index("datasets")
    if idx < 1:
        return None
    return parts[idx - 1]


def _npz_priority(npz_path: Path) -> Tuple[int, int, str]:
    """Prefer nested results/results paths when both variants exist."""
    path_str = npz_path.as_posix()
    nested_bonus = 0 if "/results/results/" in path_str else 1
    return (nested_bonus, len(path_str), path_str)


def _scan_anm_npz_candidates() -> Tuple[Dict[TaskKey, Path], Dict[TaskKey, List[Path]]]:
    grouped: Dict[TaskKey, List[Path]] = defaultdict(list)

    for npz in RESULTS_ROOT.rglob("train_ts*_s*.npz"):
        match = TRAIN_FILE_RE.fullmatch(npz.name)
        if match is None:
            continue

        dataset = _extract_dataset_from_npz(npz)
        if dataset is None or dataset not in ANM_DATASETS:
            continue

        train_size = int(match.group(1))
        seed = int(match.group(2))
        key: TaskKey = (dataset, train_size, seed)
        grouped[key].append(npz.resolve())

    selected: Dict[TaskKey, Path] = {}
    for key, candidates in grouped.items():
        selected[key] = min(candidates, key=_npz_priority)

    return selected, grouped


def _load_keys_from_cleaned_csv() -> Set[TaskKey]:
    keys: Set[TaskKey] = set()
    csv_candidates = sorted(CSV_ROOT.glob("result_*_comparison_experiment_cleaned_reps_100.csv"))
    csv_candidates += sorted((CSV_ROOT / "cleaned_csv").glob("result_*_comparison_experiment_cleaned_reps_100.csv"))
    # Keep deterministic ordering without duplicates.
    seen_paths = set()
    ordered_candidates = []
    for csv_path in csv_candidates:
        if csv_path in seen_paths:
            continue
        seen_paths.add(csv_path)
        ordered_candidates.append(csv_path)

    for csv_path in ordered_candidates:
        dataset = _dataset_from_csv_name(csv_path.name)
        if dataset not in ANM_DATASETS:
            continue

        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                train_size_raw = row.get("train_size")
                seed_raw = row.get("seed")
                if train_size_raw in (None, "") or seed_raw in (None, ""):
                    continue
                train_size = int(float(train_size_raw))
                seed = int(float(seed_raw))
                keys.add((dataset, train_size, seed))

    return keys


def _build_rows(
    keys: Iterable[TaskKey],
    npz_by_key: Dict[TaskKey, Path],
    train_sizes: Set[int],
    scope: str,
) -> Tuple[List[TaskRow], List[TaskKey]]:
    out_root = OUT_ROOT_BASE / scope
    rows: List[TaskRow] = []
    missing: List[TaskKey] = []

    for dataset, train_size, seed in sorted(keys):
        if train_size not in train_sizes:
            continue

        npz = npz_by_key.get((dataset, train_size, seed))
        if npz is None:
            missing.append((dataset, train_size, seed))
            continue

        out_dir = out_root / "comparison" / dataset / f"ts{train_size}_s{seed}"
        rows.append((dataset, seed, train_size, str(npz), str(out_dir)))

    return rows, missing


def _write_rows_tsv(path: Path, rows: List[TaskRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join("\t".join(map(str, row)) for row in rows)
    if content:
        content += "\n"
    path.write_text(content)


def _write_manifest(path: Path, rows: List[TaskRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["dataset\tseed\ttrain_size\tnpz\tout_dir\texpected_json"]
    for dataset, seed, train_size, npz, out_dir in rows:
        expected_json = f"{out_dir}/rex_union.json"
        lines.append(f"{dataset}\t{seed}\t{train_size}\t{npz}\t{out_dir}\t{expected_json}")
    path.write_text("\n".join(lines) + "\n")


def _write_counts_report(
    path: Path,
    rows: List[TaskRow],
    mode: str,
    scope: str,
    selected_train_sizes: List[int],
    grouped_candidates: Dict[TaskKey, List[Path]],
    missing: List[TaskKey],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    by_dataset_ts: Dict[str, Counter] = defaultdict(Counter)
    for dataset, _, train_size, _, _ in rows:
        by_dataset_ts[dataset][train_size] += 1

    duplicated_keys = [key for key, candidates in grouped_candidates.items() if len(candidates) > 1]

    lines: List[str] = []
    lines.append(f"mode: {mode}")
    lines.append(f"scope: {scope}")
    lines.append(f"train_sizes: {selected_train_sizes}")
    lines.append(f"tasks: {len(rows)}")
    lines.append(f"npz_keys_with_multiple_paths: {len(duplicated_keys)}")
    lines.append(f"missing_keys: {len(missing)}")
    lines.append("")

    for dataset in sorted(by_dataset_ts):
        lines.append(dataset)
        for train_size in sorted(by_dataset_ts[dataset]):
            lines.append(f"  ts{train_size}: n={by_dataset_ts[dataset][train_size]}")
        lines.append("")

    if missing:
        lines.append("Missing keys (dataset,train_size,seed):")
        for dataset, train_size, seed in missing[:200]:
            lines.append(f"  {dataset}\tts{train_size}\ts{seed}")
        if len(missing) > 200:
            lines.append(f"  ... truncated ({len(missing) - 200} more)")
        lines.append("")

    if duplicated_keys:
        lines.append("Example duplicate path variants (first 20 keys):")
        for key in sorted(duplicated_keys)[:20]:
            lines.append(f"  {key}")
            candidates = sorted(grouped_candidates[key], key=_npz_priority)
            for candidate in candidates:
                lines.append(f"    - {candidate}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def _write_chunks(chunk_dir: Path, rows: List[TaskRow], chunk_size: int) -> int:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for old in chunk_dir.glob("part_*.tsv"):
        old.unlink()

    if chunk_size <= 0:
        return 0

    chunk_count = 0
    for start in range(0, len(rows), chunk_size):
        chunk_rows = rows[start : start + chunk_size]
        chunk_path = chunk_dir / f"part_{chunk_count:02d}.tsv"
        _write_rows_tsv(chunk_path, chunk_rows)
        chunk_count += 1

    return chunk_count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare REX boot20 no-HPO task files for ANM CSuite")
    parser.add_argument(
        "--mode",
        choices=["cleaned-csv", "all-npz"],
        default="all-npz",
        help="Task source: cleaned CSV keys or all NPZ scan",
    )
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        default=[20, 50, 100, 200, 500],
        help="Train sizes to keep in output task list",
    )
    parser.add_argument(
        "--scope",
        default=None,
        help="Scope name used for output folders/files (default: auto-generated)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Rows per chunk file (0 disables chunk generation)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_sizes = sorted(set(args.train_sizes))
    train_size_set = set(train_sizes)

    auto_scope = f"anm_{args.mode.replace('-', '_')}_ts{'-'.join(map(str, train_sizes))}"
    scope = args.scope or auto_scope

    selected_npz, grouped_candidates = _scan_anm_npz_candidates()

    if args.mode == "cleaned-csv":
        source_keys = _load_keys_from_cleaned_csv()
    else:
        source_keys = set(selected_npz.keys())

    rows, missing = _build_rows(
        keys=source_keys,
        npz_by_key=selected_npz,
        train_sizes=train_size_set,
        scope=scope,
    )

    task_file = SLURM_DIR / f"rex_tasks_{scope}.tsv"
    manifest_file = NOTES_DIR / f"rex_task_manifest_{scope}.tsv"
    report_file = NOTES_DIR / f"rex_task_counts_{scope}.txt"
    chunk_dir = SLURM_DIR / "chunks" / scope

    _write_rows_tsv(task_file, rows)
    _write_manifest(manifest_file, rows)
    _write_counts_report(
        path=report_file,
        rows=rows,
        mode=args.mode,
        scope=scope,
        selected_train_sizes=train_sizes,
        grouped_candidates=grouped_candidates,
        missing=missing,
    )
    chunk_count = _write_chunks(chunk_dir, rows, args.chunk_size)

    print(f"scope: {scope}")
    print(f"mode: {args.mode}")
    print(f"train_sizes: {train_sizes}")
    print(f"tasks: {len(rows)} -> {task_file}")
    print(f"manifest: {manifest_file}")
    print(f"report: {report_file}")
    print(f"chunks: {chunk_count} -> {chunk_dir}")
    if missing:
        print(f"warning: missing keys: {len(missing)} (see report)")


if __name__ == "__main__":
    main()
