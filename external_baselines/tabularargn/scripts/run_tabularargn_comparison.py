#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
import tempfile
import time
import traceback
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-tabularargn"))

import numpy as np
import pandas as pd


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def add_tabpfn_repo_to_path(tabpfn_repo: Path) -> None:
    repo = tabpfn_repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def load_npz_array(path: Path, keys: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=True) as data:
        for key in keys:
            if key in data:
                array = data[key]
                break
        else:
            raise KeyError(f"None of {keys} found in {path}")
        if "column_names" not in data:
            raise KeyError(f"column_names missing in {path}")
        column_names = data["column_names"].tolist()
    return array, list(column_names)


def load_train_split(dataset_dir: Path, train_size: int, seed: int) -> tuple[pd.DataFrame, str]:
    path = dataset_dir / "datasets" / f"train_ts{train_size}_s{seed}.npz"
    data, column_names = load_npz_array(path, ("X_train", "train_data", "data"))
    return pd.DataFrame(data, columns=column_names), str(path)


def load_global_test(dataset_dir: Path) -> tuple[pd.DataFrame, str]:
    path = dataset_dir / "datasets" / "global_test_set.npz"
    data, column_names = load_npz_array(path, ("X_test", "test_data", "data"))
    return pd.DataFrame(data, columns=column_names), str(path)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_model(
    model_id: str | None,
    max_training_time: float | None,
    seed: int,
    device: str | None,
    verbose: int,
    workspace_dir: Path | None,
):
    from mostlyai.engine import TabularARGN

    signature = inspect.signature(TabularARGN)
    kwargs: dict[str, Any] = {}
    if model_id and "model" in signature.parameters:
        kwargs["model"] = model_id
    if max_training_time is not None and "max_training_time" in signature.parameters:
        kwargs["max_training_time"] = max_training_time
    if "verbose" in signature.parameters:
        kwargs["verbose"] = verbose
    if device and "device" in signature.parameters:
        kwargs["device"] = device
    if workspace_dir is not None and "workspace_dir" in signature.parameters:
        kwargs["workspace_dir"] = workspace_dir
    for seed_name in ("random_state", "seed", "random_seed"):
        if seed_name in signature.parameters:
            kwargs[seed_name] = seed
            break
    return TabularARGN(**kwargs)


def sample_synthetic(model: Any, n_samples: int, seed: int, device: str | None) -> pd.DataFrame:
    set_all_seeds(seed)
    signature = inspect.signature(model.sample)
    kwargs: dict[str, Any] = {}
    if "n_samples" in signature.parameters:
        kwargs["n_samples"] = n_samples
    elif "size" in signature.parameters:
        kwargs["size"] = n_samples
    else:
        kwargs["n_samples"] = n_samples
    for seed_name in ("random_state", "seed", "random_seed"):
        if seed_name in signature.parameters:
            kwargs[seed_name] = seed
            break
    if device and "device" in signature.parameters:
        kwargs["device"] = device
    synthetic = model.sample(**kwargs)
    if not isinstance(synthetic, pd.DataFrame):
        synthetic = pd.DataFrame(synthetic)
    return synthetic


def coerce_like_reference(synthetic: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    synthetic = synthetic.copy()
    missing = [column for column in reference.columns if column not in synthetic.columns]
    if missing:
        raise ValueError(f"Synthetic data is missing columns: {missing}")
    synthetic = synthetic[reference.columns]
    for column in reference.columns:
        if pd.api.types.is_numeric_dtype(reference[column]):
            synthetic[column] = pd.to_numeric(synthetic[column], errors="coerce").astype(reference[column].dtype)
    return synthetic


def save_synthetic_npz(
    synthetic: pd.DataFrame,
    output_dir: Path,
    train_size: int,
    seed: int,
    metrics: dict[str, float],
    train_path: str,
    test_path: str,
    max_training_time: float | None,
) -> str:
    path = output_dir / "datasets" / "synthetic" / f"synthetic_tabularargn_unconditional_ts{train_size}_s{seed}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "algorithm": "tabularargn",
        "column_order": "unconditional",
        "train_size": train_size,
        "seed": seed,
        "n_samples": int(len(synthetic)),
        "train_dataset_path": train_path,
        "test_dataset_path": test_path,
        "max_training_time": max_training_time,
        "metrics": metrics,
    }
    np.savez_compressed(
        path,
        synthetic_data=synthetic.to_numpy(),
        column_names=list(synthetic.columns),
        metadata=np.array(metadata, dtype=object),
    )
    return str(path)


def evaluate_paper_metrics(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    categorical_columns: list[str],
    random_seed: int,
) -> dict[str, float]:
    from causal_experiments.utils.metrics import (
        calculate_kmarginal_tvd,
        frobenius_corr_mixed_spearman,
    )
    from syntheval import SynthEval

    results: dict[str, float] = {}
    try:
        corr_diff, _ = frobenius_corr_mixed_spearman(real_data, synthetic_data, categorical_columns)
        if not np.isfinite(corr_diff):
            raise ValueError("non-finite correlation_matrix_difference")
        results["correlation_matrix_difference"] = float(corr_diff)
    except Exception:
        try:
            numeric_columns = [column for column in real_data.columns if column not in categorical_columns]
            real_corr = real_data[numeric_columns].corr(method="spearman").to_numpy(copy=True)
            synthetic_corr = synthetic_data[numeric_columns].corr(method="spearman").to_numpy(copy=True)
            np.fill_diagonal(real_corr, 1.0)
            np.fill_diagonal(synthetic_corr, 1.0)
            # Constant generated columns have undefined Spearman correlations.
            # Treat undefined off-diagonal correlations as absent signal so CMD
            # remains finite instead of becoming an empty CSV cell.
            real_corr = np.nan_to_num(real_corr, nan=0.0, posinf=0.0, neginf=0.0)
            synthetic_corr = np.nan_to_num(synthetic_corr, nan=0.0, posinf=0.0, neginf=0.0)
            corr_diff = float(np.linalg.norm(real_corr - synthetic_corr, ord="fro"))
            results["correlation_matrix_difference"] = corr_diff if np.isfinite(corr_diff) else -1.0
        except Exception:
            results["correlation_matrix_difference"] = -1.0

    try:
        evaluator = SynthEval(real_data, cat_cols=categorical_columns, verbose=False)
        with warnings.catch_warnings(), open(os.devnull, "w") as devnull, redirect_stdout(devnull):
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            evaluator.evaluate(
                synthetic_data,
                nnaa={"n_resample": 30},
            )
        results["nnaa"] = float(evaluator._raw_results["nnaa"]["avg"])
    except Exception:
        results["nnaa"] = -1.0

    try:
        results["k_marginal_tvd"] = float(
            calculate_kmarginal_tvd(
                real_data,
                synthetic_data,
                categorical_columns,
                k=2,
                random_seed=random_seed,
            )
        )
    except Exception:
        results["k_marginal_tvd"] = -1.0

    return results


def run_one(
    dataset: str,
    dataset_dir: Path,
    output_dir: Path,
    train_size: int,
    seed: int,
    model_id: str | None,
    max_training_time: float | None,
    device: str | None,
    save_synthetic: bool,
    verbose: int,
) -> dict[str, Any]:
    started = time.time()
    set_all_seeds(seed)
    train_df, train_path = load_train_split(dataset_dir, train_size, seed)
    test_df, test_path = load_global_test(dataset_dir)

    workspace_root = output_dir / "tmp_workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"tabularargn_ts{train_size}_s{seed}_",
        dir=workspace_root,
    ) as workspace:
        model = build_model(
            model_id=model_id,
            max_training_time=max_training_time,
            seed=seed,
            device=device,
            verbose=verbose,
            workspace_dir=Path(workspace),
        )
        fit_started = time.time()
        model.fit(X=train_df)
        fit_seconds = time.time() - fit_started

        sample_started = time.time()
        synthetic_df = sample_synthetic(model, n_samples=len(test_df), seed=seed, device=device)
        sample_seconds = time.time() - sample_started
    synthetic_df = coerce_like_reference(synthetic_df, test_df)

    metrics = evaluate_paper_metrics(
        real_data=test_df,
        synthetic_data=synthetic_df,
        categorical_columns=[],
        random_seed=seed,
    )

    synthetic_path = ""
    if save_synthetic:
        synthetic_path = save_synthetic_npz(
            synthetic=synthetic_df,
            output_dir=output_dir,
            train_size=train_size,
            seed=seed,
            metrics=metrics,
            train_path=train_path,
            test_path=test_path,
            max_training_time=max_training_time,
        )

    return {
        "dataset": dataset,
        "algorithm": "tabularargn",
        "column_order": "unconditional",
        "train_size": train_size,
        "seed": seed,
        "repetition": None,
        "train_dataset_path": train_path,
        "test_dataset_path": test_path,
        "synthetic_dataset_path": synthetic_path,
        "model_id": model_id,
        "max_training_time": max_training_time,
        "fit_seconds": fit_seconds,
        "sample_seconds": sample_seconds,
        "total_seconds": time.time() - started,
        "correlation_matrix_difference": metrics.get("correlation_matrix_difference", -1.0),
        "k_marginal_tvd": metrics.get("k_marginal_tvd", -1.0),
        "nnaa": metrics.get("nnaa", -1.0),
    }


def append_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(rows)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def load_completed(csv_path: Path) -> set[tuple[int, int]]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    if "train_size" not in df.columns or "seed" not in df.columns:
        return set()
    return {(int(row.train_size), int(row.seed)) for row in df.itertuples(index=False)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TabularARGN baseline on cached comparison NPZ splits.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--tabpfn-repo", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-sizes", default="20,50,100,200,500")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--seed-list",
        default="",
        help=(
            "Optional comma-separated explicit seeds. When provided, these exact "
            "seeds are used for each requested train size instead of the contiguous "
            "seed_start/repetitions range."
        ),
    )
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--max-training-time", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--save-synthetic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    add_tabpfn_repo_to_path(args.tabpfn_repo)
    try:
        from causal_experiments.utils.metrics import FaithfulDataEvaluator  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Unable to import the paper metric implementation. Fix the environment before launching "
            "the baseline, otherwise the job would only emit failed rows."
        ) from exc

    train_sizes = parse_csv_ints(args.train_sizes)
    explicit_seeds = parse_csv_ints(args.seed_list) if args.seed_list else None
    output_dir = args.output_dir.resolve()
    csv_path = output_dir / f"result_{args.dataset}_tabularargn_baseline{args.output_suffix}.csv"
    completed = load_completed(csv_path) if args.resume else set()

    rows: list[dict[str, Any]] = []
    for train_size in train_sizes:
        if explicit_seeds is not None:
            seeds = explicit_seeds
        else:
            seeds = list(range(args.seed_start, args.seed_start + args.repetitions))
        for seed in seeds:
            if (train_size, seed) in completed:
                continue
            try:
                row = run_one(
                    dataset=args.dataset,
                    dataset_dir=args.dataset_dir.resolve(),
                    output_dir=output_dir,
                    train_size=train_size,
                    seed=seed,
                    model_id=args.model_id,
                    max_training_time=args.max_training_time,
                    device=args.device,
                    save_synthetic=args.save_synthetic,
                    verbose=args.verbose,
                )
            except Exception as exc:
                if args.fail_fast:
                    raise
                row = {
                    "dataset": args.dataset,
                    "algorithm": "tabularargn",
                    "column_order": "unconditional",
                    "train_size": train_size,
                    "seed": seed,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "correlation_matrix_difference": -1.0,
                    "k_marginal_tvd": -1.0,
                    "nnaa": -1.0,
                }
            rows.append(row)
            if len(rows) >= args.save_every:
                append_rows(csv_path, rows)
                rows.clear()
    if rows:
        append_rows(csv_path, rows)

    metadata = {
        "dataset": args.dataset,
        "dataset_dir": str(args.dataset_dir),
        "train_sizes": train_sizes,
        "seed_start": args.seed_start,
        "seed_list": explicit_seeds,
        "repetitions": args.repetitions,
        "model_id": args.model_id,
        "max_training_time": args.max_training_time,
        "device": args.device,
        "save_synthetic": args.save_synthetic,
    }
    (output_dir / f"metadata_{args.dataset}_tabularargn_baseline{args.output_suffix}.json").write_text(
        json.dumps(metadata, indent=2)
    )


if __name__ == "__main__":
    main()
