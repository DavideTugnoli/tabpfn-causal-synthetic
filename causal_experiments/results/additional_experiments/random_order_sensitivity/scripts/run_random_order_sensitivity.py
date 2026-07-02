#!/usr/bin/env python3
"""Run vanilla TabPFN random column-order sensitivity on cached NPZ splits."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_experiments.utils.metrics import FaithfulDataEvaluator
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised


METRIC_COLUMNS = ["correlation_matrix_difference", "k_marginal_tvd", "nnaa"]


@dataclass(frozen=True)
class CachedSplit:
    train_size: int
    seed: int
    repetition_index: int
    path: Path
    X_train: np.ndarray
    column_names: list[str]


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def set_run_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz_array(path: Path, preferred_keys: list[str]) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=True) as data:
        array_key = next((key for key in preferred_keys if key in data), None)
        if array_key is None:
            raise KeyError(f"None of {preferred_keys} found in {path}; keys={list(data.keys())}")
        X = np.asarray(data[array_key])
        if "column_names" in data:
            column_names = [str(x) for x in data["column_names"].tolist()]
        else:
            column_names = [f"X{i}" for i in range(X.shape[1])]
    return X, column_names


def infer_seed_from_train_path(path: Path) -> int:
    match = re.search(r"_s(\d+)\.npz$", path.name)
    if not match:
        raise ValueError(f"Cannot infer seed from {path.name}")
    return int(match.group(1))


def load_cached_split(
    dataset_dir: Path,
    train_size: int,
    seed: int,
    repetition_index: int,
) -> CachedSplit:
    path = dataset_dir / f"train_ts{train_size}_s{seed}.npz"
    X_train, column_names = load_npz_array(path, ["X_train", "X"])
    return CachedSplit(
        train_size=train_size,
        seed=seed,
        repetition_index=repetition_index,
        path=path,
        X_train=X_train,
        column_names=column_names,
    )


def valid_metric_row(row: pd.Series) -> bool:
    return all(float(row.get(metric, -1.0)) >= 0 for metric in METRIC_COLUMNS)


def result_key(row: pd.Series | dict[str, Any]) -> tuple[int, int, str, int]:
    return (
        int(row["train_size"]),
        int(row["row_seed"]),
        str(row["ordering_label"]),
        int(row["ordering_id"]),
    )


def choose_seeds_from_cleaned_csv(
    cleaned_csv: Path,
    train_sizes: list[int],
    n_row_repetitions: int,
    algorithm: str = "vanilla",
    column_order: str = "original",
) -> dict[int, list[int]]:
    df = pd.read_csv(cleaned_csv)
    required = {"train_size", "seed", "algorithm", "column_order", *METRIC_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {cleaned_csv}: {sorted(missing)}")

    filtered = df[
        (df["algorithm"] == algorithm)
        & (df["column_order"] == column_order)
        & (df["train_size"].isin(train_sizes))
    ].copy()
    filtered = filtered[filtered.apply(valid_metric_row, axis=1)]

    seeds_by_train_size: dict[int, list[int]] = {}
    for train_size in train_sizes:
        seeds = (
            filtered.loc[filtered["train_size"] == train_size, "seed"]
            .astype(int)
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        if len(seeds) < n_row_repetitions:
            raise ValueError(
                f"Need {n_row_repetitions} cleaned seeds for train_size={train_size}, "
                f"found {len(seeds)} in {cleaned_csv}"
            )
        seeds_by_train_size[train_size] = seeds[:n_row_repetitions]
    return seeds_by_train_size


def choose_cached_splits(
    dataset_dir: Path,
    train_sizes: list[int],
    n_row_repetitions: int,
    explicit_row_seeds: list[int] | None,
    seeds_by_train_size: dict[int, list[int]] | None = None,
) -> list[CachedSplit]:
    splits: list[CachedSplit] = []
    for train_size in train_sizes:
        if seeds_by_train_size is not None:
            seeds = seeds_by_train_size[train_size]
        elif explicit_row_seeds is not None:
            seeds = explicit_row_seeds
        else:
            candidates = sorted(dataset_dir.glob(f"train_ts{train_size}_s*.npz"))
            if len(candidates) < n_row_repetitions:
                raise FileNotFoundError(
                    f"Need {n_row_repetitions} cached splits for train_size={train_size}, "
                    f"found {len(candidates)} in {dataset_dir}"
                )
            seeds = [infer_seed_from_train_path(path) for path in candidates[:n_row_repetitions]]
        for repetition_index, seed in enumerate(seeds):
            splits.append(load_cached_split(dataset_dir, train_size, seed, repetition_index))
    return splits


def load_test_dataframe(dataset_dir: Path) -> pd.DataFrame:
    path = dataset_dir / "global_test_set.npz"
    X_test, column_names = load_npz_array(path, ["X_test", "X"])
    return pd.DataFrame(X_test, columns=column_names)


def infer_categorical_columns(dataset_kind: str, dataset_name: str, column_names: list[str]) -> list[str]:
    if dataset_kind == "simglucose":
        return [name for name in ["patient_id", "action_CHO_g"] if name in column_names]
    if dataset_kind == "csuite":
        try:
            from causal_experiments.utils.csuite_loader import load_csuite_dataset

            csuite_data = load_csuite_dataset(dataset_name)
            return [name for name in csuite_data.get("categorical_columns", []) if name in column_names]
        except Exception as exc:
            print(f"[WARN] Could not infer CSuite categorical columns for {dataset_name}: {exc}")
            return []
    return []


def sample_unique_orderings(
    n_features: int,
    n_orderings: int,
    seed: int,
) -> list[tuple[int, ...]]:
    all_indices = tuple(range(n_features))
    total_possible = math.factorial(n_features)
    if total_possible <= n_orderings:
        return list(itertools.permutations(all_indices))

    rng = np.random.default_rng(seed)
    orderings: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    max_attempts = max(100, n_orderings * 50)
    attempts = 0
    while len(orderings) < n_orderings and attempts < max_attempts:
        perm = tuple(int(x) for x in rng.permutation(n_features))
        if perm not in seen:
            seen.add(perm)
            orderings.append(perm)
        attempts += 1
    if len(orderings) < n_orderings:
        raise RuntimeError(f"Could only sample {len(orderings)} unique orderings out of {n_orderings}")
    return orderings


def run_vanilla_generation(
    X_train: np.ndarray,
    test_df: pd.DataFrame,
    column_names: list[str],
    categorical_cols: list[str],
    ordering: tuple[int, ...],
    seed: int,
    n_estimators: int,
    n_permutations: int,
    temperature: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    set_run_seed(seed)

    ordered_names = [column_names[i] for i in ordering]
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(ordering)}
    categorical_original = [column_names.index(col) for col in categorical_cols if col in column_names]
    categorical_reordered = [old_to_new[idx] for idx in categorical_original if idx in old_to_new]

    X_train_reordered = X_train[:, ordering]

    clf = TabPFNClassifier(n_estimators=n_estimators)
    reg = TabPFNRegressor(n_estimators=n_estimators)
    try:
        if hasattr(clf, "max_num_classes_") and clf.max_num_classes_ < 512:
            clf.max_num_classes_ = 512
        elif not hasattr(clf, "max_num_classes_"):
            setattr(clf, "max_num_classes_", 512)
    except Exception:
        pass

    model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    if categorical_reordered:
        model.set_categorical_features(categorical_reordered)
    model.set_feature_names(ordered_names)

    experiment = unsupervised.experiments.GenerateSyntheticDataExperiment(task_type="unsupervised")
    experiment.run(
        tabpfn=model,
        X=torch.tensor(X_train_reordered, dtype=torch.float32),
        y=None,
        attribute_names=ordered_names,
        temp=temperature,
        n_samples=len(test_df),
        n_permutations=n_permutations,
        indices=list(range(X_train_reordered.shape[1])),
        categorical_features=categorical_reordered,
    )

    if torch.is_tensor(experiment.synthetic_X):
        synthetic_reordered = experiment.synthetic_X.detach().cpu().numpy()
    else:
        synthetic_reordered = np.asarray(experiment.synthetic_X)
    reverse_ordering = [ordering.index(i) for i in range(len(ordering))]
    synthetic_original = synthetic_reordered[:, reverse_ordering]
    synthetic_df = pd.DataFrame(synthetic_original, columns=column_names)

    evaluator = FaithfulDataEvaluator()
    metrics = evaluator.evaluate(
        real_data=test_df,
        synthetic_data=synthetic_df,
        categorical_columns=categorical_cols,
        k_for_kmarginal=2,
        random_seed=seed,
    )
    return {key: float(metrics.get(key, np.nan)) for key in METRIC_COLUMNS}, synthetic_df


def save_synthetic_npz(
    output_dir: Path,
    synthetic_df: pd.DataFrame,
    row: dict[str, Any],
    column_names: list[str],
) -> str:
    synthetic_dir = output_dir / "synthetic_data"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"synthetic_ts{row['train_size']}_s{row['row_seed']}_"
        f"{row['ordering_label']}_p{row['ordering_id']}.npz"
    )
    path = synthetic_dir / filename
    np.savez_compressed(
        path,
        X_synthetic=synthetic_df.to_numpy(),
        column_names=np.asarray(column_names),
        metadata=np.asarray(row, dtype=object),
    )
    return str(path)


def copy_input_file_once(source: Path, input_dir: Path) -> str:
    input_dir.mkdir(parents=True, exist_ok=True)
    destination = input_dir / source.name
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise FileExistsError(
                f"Refusing to overwrite existing input copy with different size: {destination}"
            )
    else:
        shutil.copy2(source, destination)
    return str(destination)


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["dataset", "train_size", "row_seed"]
    for group_values, group in results.groupby(keys, dropna=False):
        group_dict = dict(zip(keys, group_values if isinstance(group_values, tuple) else (group_values,)))
        original = group[group["ordering_label"] == "original_reference"]
        random_df = group[group["ordering_label"] == "random_order"]
        if original.empty or random_df.empty:
            continue
        for metric in METRIC_COLUMNS:
            original_value = float(original.iloc[0][metric])
            random_values = random_df[metric].astype(float)
            rows.append(
                {
                    **group_dict,
                    "metric": metric,
                    "original_reference": original_value,
                    "random_mean": float(random_values.mean()),
                    "random_median": float(random_values.median()),
                    "random_min": float(random_values.min()),
                    "random_max": float(random_values.max()),
                    "mean_minus_original": float(random_values.mean() - original_value),
                    "median_minus_original": float(random_values.median() - original_value),
                    "n_random_orderings": int(random_values.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Comparison result root containing datasets/*.npz")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-kind", choices=["custom", "csuite", "simglucose"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--design",
        choices=["grid", "paired"],
        default="grid",
        help=(
            "grid: run original_reference plus all random orderings for every split. "
            "paired: run one random ordering per split, cycling through a fixed dataset-level pool."
        ),
    )
    parser.add_argument("--train-sizes", default="20,50,100,200,500")
    parser.add_argument("--n-row-repetitions", type=int, default=1)
    parser.add_argument("--row-seeds", default=None, help="Optional comma-separated seed list reused for every train size")
    parser.add_argument(
        "--cleaned-reference-csv",
        type=Path,
        default=None,
        help="Optional cleaned reps CSV used to select the exact vanilla_original valid seeds per train size.",
    )
    parser.add_argument("--n-orderings", type=int, default=10)
    parser.add_argument("--order-seed", type=int, default=20260427)
    parser.add_argument(
        "--include-original-reference",
        action="store_true",
        help="Also rerun vanilla original on each selected split. Useful for small grid checks; expensive for paired all-dataset runs.",
    )
    parser.add_argument("--n-estimators", type=int, default=3)
    parser.add_argument("--n-permutations", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save-synthetic", action="store_true")
    parser.add_argument(
        "--copy-input-datasets",
        action="store_true",
        help="Copy the exact cached train/test NPZ files used by this run into output-root/input_datasets.",
    )
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()

    datasets_dir = args.dataset_root / "datasets"
    if not datasets_dir.is_dir():
        raise FileNotFoundError(f"Missing datasets directory: {datasets_dir}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    train_sizes = parse_int_list(args.train_sizes)
    row_seeds = parse_int_list(args.row_seeds) if args.row_seeds else None
    seeds_by_train_size = None
    if args.cleaned_reference_csv is not None:
        seeds_by_train_size = choose_seeds_from_cleaned_csv(
            cleaned_csv=args.cleaned_reference_csv,
            train_sizes=train_sizes,
            n_row_repetitions=args.n_row_repetitions,
        )

    test_df = load_test_dataframe(datasets_dir)
    splits = choose_cached_splits(
        datasets_dir,
        train_sizes=train_sizes,
        n_row_repetitions=args.n_row_repetitions,
        explicit_row_seeds=row_seeds,
        seeds_by_train_size=seeds_by_train_size,
    )
    copied_input_paths: dict[str, str] = {}
    if args.copy_input_datasets:
        input_copy_dir = args.output_root / "input_datasets"
        copied_input_paths[str(datasets_dir / "global_test_set.npz")] = copy_input_file_once(
            datasets_dir / "global_test_set.npz",
            input_copy_dir,
        )
        if args.cleaned_reference_csv is not None:
            copied_input_paths[str(args.cleaned_reference_csv)] = copy_input_file_once(
                args.cleaned_reference_csv,
                input_copy_dir,
            )
        for split in splits:
            copied_input_paths[str(split.path)] = copy_input_file_once(split.path, input_copy_dir)
    categorical_cols = infer_categorical_columns(args.dataset_kind, args.dataset_name, list(test_df.columns))
    print(f"[INFO] Dataset={args.dataset_name} kind={args.dataset_kind}")
    print(f"[INFO] Design={args.design}")
    print(f"[INFO] Train sizes={train_sizes}")
    print(f"[INFO] Splits={[(s.train_size, s.seed, s.repetition_index) for s in splits]}")
    print(f"[INFO] Categorical columns={categorical_cols}")

    results_path = args.output_root / "random_order_sensitivity_results.csv"
    all_rows: list[dict[str, Any]] = []
    completed_keys: set[tuple[int, int, str, int]] = set()
    if results_path.exists():
        existing_results = pd.read_csv(results_path)
        required_resume_columns = {"train_size", "row_seed", "ordering_label", "ordering_id", *METRIC_COLUMNS}
        missing_resume_columns = required_resume_columns - set(existing_results.columns)
        if missing_resume_columns:
            print(f"[WARN] Existing results cannot be resumed, missing columns: {sorted(missing_resume_columns)}")
        else:
            valid_existing = existing_results[existing_results.apply(valid_metric_row, axis=1)].copy()
            for row in valid_existing.to_dict("records"):
                key = result_key(row)
                if key in completed_keys:
                    continue
                completed_keys.add(key)
                all_rows.append(row)
            print(
                f"[INFO] Resume: loaded {len(all_rows)} valid completed rows from "
                f"{results_path}; invalid or incomplete rows will be recomputed."
            )

    manifest: dict[str, Any] = {
        "dataset": args.dataset_name,
        "dataset_kind": args.dataset_kind,
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "design": args.design,
        "train_sizes": train_sizes,
        "n_row_repetitions": args.n_row_repetitions,
        "row_seeds": row_seeds,
        "cleaned_reference_csv": str(args.cleaned_reference_csv) if args.cleaned_reference_csv else None,
        "n_orderings": args.n_orderings,
        "order_seed": args.order_seed,
        "include_original_reference": bool(args.include_original_reference),
        "save_synthetic": bool(args.save_synthetic),
        "copy_input_datasets": bool(args.copy_input_datasets),
        "copied_input_paths": copied_input_paths,
        "orderings": [],
    }

    first_split = splits[0]
    fixed_random_orderings = sample_unique_orderings(
        n_features=first_split.X_train.shape[1],
        n_orderings=args.n_orderings,
        seed=args.order_seed,
    )
    manifest["fixed_random_ordering_pool"] = [list(ordering) for ordering in fixed_random_orderings]
    manifest["column_names"] = first_split.column_names

    for split in splits:
        n_features = split.X_train.shape[1]
        if n_features != len(first_split.column_names):
            raise ValueError("All splits for one dataset must have the same number of columns.")
        original_order = tuple(range(n_features))
        if args.design == "grid":
            random_orderings = fixed_random_orderings
            run_specs = [
                ("original_reference", -1, original_order)
            ] + [("random_order", idx, ordering) for idx, ordering in enumerate(random_orderings)]
        else:
            ordering_id = split.repetition_index % len(fixed_random_orderings)
            random_orderings = [fixed_random_orderings[ordering_id]]
            run_specs = [("random_order", ordering_id, random_orderings[0])]
            if args.include_original_reference:
                run_specs = [("original_reference", -1, original_order)] + run_specs
        manifest["orderings"].append(
            {
                "train_size": split.train_size,
                "row_seed": split.seed,
                "repetition_index": split.repetition_index,
                "original_order": list(original_order),
                "random_orderings": [list(ordering) for ordering in random_orderings],
                "column_names": split.column_names,
            }
        )

        for ordering_label, ordering_id, ordering in run_specs:
            run_seed = args.order_seed + split.train_size * 100_000 + split.seed * 1_000 + max(ordering_id, 0)
            key = (
                int(split.train_size),
                int(split.seed),
                str(ordering_label),
                int(ordering_id),
            )
            if key in completed_keys:
                print(
                    f"[INFO] Skipping completed {args.dataset_name} train_size={split.train_size} "
                    f"row_seed={split.seed} {ordering_label} ordering_id={ordering_id}"
                )
                continue
            print(
                f"[INFO] Running {args.dataset_name} train_size={split.train_size} "
                f"row_seed={split.seed} {ordering_label} ordering_id={ordering_id} ordering={ordering}"
            )
            try:
                metrics, synthetic_df = run_vanilla_generation(
                    X_train=split.X_train,
                    test_df=test_df,
                    column_names=split.column_names,
                    categorical_cols=categorical_cols,
                    ordering=ordering,
                    seed=run_seed,
                    n_estimators=args.n_estimators,
                    n_permutations=args.n_permutations,
                    temperature=args.temperature,
                )
                row = {
                    "dataset": args.dataset_name,
                    "dataset_kind": args.dataset_kind,
                    "train_size": split.train_size,
                    "row_seed": split.seed,
                    "repetition_index": split.repetition_index,
                    "ordering_label": ordering_label,
                    "ordering_id": ordering_id,
                    "ordering": " ".join(map(str, ordering)),
                    "run_seed": run_seed,
                    "train_dataset_path": str(split.path),
                    "test_dataset_path": str(datasets_dir / "global_test_set.npz"),
                    "copied_train_dataset_path": copied_input_paths.get(str(split.path)),
                    "copied_test_dataset_path": copied_input_paths.get(str(datasets_dir / "global_test_set.npz")),
                    **metrics,
                }
                if args.save_synthetic:
                    row["synthetic_data_path"] = save_synthetic_npz(
                        args.output_root,
                        synthetic_df,
                        row,
                        split.column_names,
                    )
            except Exception as exc:
                row = {
                    "dataset": args.dataset_name,
                    "dataset_kind": args.dataset_kind,
                    "train_size": split.train_size,
                    "row_seed": split.seed,
                    "repetition_index": split.repetition_index,
                    "ordering_label": ordering_label,
                    "ordering_id": ordering_id,
                    "ordering": " ".join(map(str, ordering)),
                    "run_seed": run_seed,
                    "train_dataset_path": str(split.path),
                    "test_dataset_path": str(datasets_dir / "global_test_set.npz"),
                    "copied_train_dataset_path": copied_input_paths.get(str(split.path)),
                    "copied_test_dataset_path": copied_input_paths.get(str(datasets_dir / "global_test_set.npz")),
                    "error": repr(exc),
                    **{metric: -1.0 for metric in METRIC_COLUMNS},
                }
                print(f"[ERROR] Run failed: {exc!r}")
            all_rows.append(row)
            if valid_metric_row(pd.Series(row)):
                completed_keys.add(key)
            if args.save_every > 0 and len(all_rows) % args.save_every == 0:
                pd.DataFrame(all_rows).to_csv(results_path, index=False)

    results = pd.DataFrame(all_rows)
    results.to_csv(results_path, index=False)
    summary = build_summary(results[results[METRIC_COLUMNS].ge(0).all(axis=1)])
    summary.to_csv(args.output_root / "random_order_sensitivity_summary.csv", index=False)
    with (args.output_root / "orderings_manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[INFO] Wrote {results_path}")
    print(f"[INFO] Wrote {args.output_root / 'random_order_sensitivity_summary.csv'}")
    print(f"[INFO] Wrote {args.output_root / 'orderings_manifest.json'}")


if __name__ == "__main__":
    main()
