#!/usr/bin/env python3
from __future__ import annotations

import abc
import json
import os
import random
import time
import traceback
import warnings
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProtocolConfig:
    """Fixed comparison protocol shared by external generator adapters."""

    dataset: str
    dataset_dir: Path
    output_dir: Path
    train_sizes: tuple[int, ...]
    seeds: tuple[int, ...]
    tabpfn_repo: Path
    save_synthetic: bool = True
    resume: bool = True
    save_every: int = 1
    output_suffix: str = ""


@dataclass(frozen=True)
class ProtocolResult:
    dataset: str
    algorithm: str
    column_order: str
    train_size: int
    seed: int
    train_dataset_path: str
    test_dataset_path: str
    synthetic_dataset_path: str
    fit_sample_seconds: float
    total_seconds: float
    correlation_matrix_difference: float
    k_marginal_tvd: float
    nnaa: float
    error: str = ""
    traceback: str = ""


class ExternalGeneratorAdapter(abc.ABC):
    """Adapter contract for models evaluated under the paired-seed protocol."""

    name: str
    column_order: str = "unconditional"

    @abc.abstractmethod
    def fit_sample(
        self,
        train_df: pd.DataFrame,
        n_samples: int,
        seed: int,
        workspace_dir: Path,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fit on the training split and return a synthetic table plus metadata."""


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


def add_tabpfn_repo_to_path(tabpfn_repo: Path) -> None:
    import sys

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
    candidates = (
        dataset_dir / "datasets" / f"train_ts{train_size}_s{seed}.npz",
        dataset_dir / "train" / f"train_ts{train_size}_s{seed}.npz",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    data, column_names = load_npz_array(path, ("X_train", "train_data", "data"))
    return pd.DataFrame(data, columns=column_names), str(path)


def load_global_test(dataset_dir: Path) -> tuple[pd.DataFrame, str]:
    candidates = (
        dataset_dir / "datasets" / "global_test_set.npz",
        dataset_dir / "global_test_set.npz",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    data, column_names = load_npz_array(path, ("X_test", "test_data", "data"))
    return pd.DataFrame(data, columns=column_names), str(path)


def coerce_like_reference(synthetic: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    synthetic = synthetic.copy()
    missing = [column for column in reference.columns if column not in synthetic.columns]
    if missing:
        raise ValueError(f"Synthetic data is missing columns: {missing}")
    synthetic = synthetic[reference.columns]
    for column in reference.columns:
        synthetic[column] = pd.to_numeric(synthetic[column], errors="coerce")
    if synthetic.isna().any().any():
        raise ValueError("Synthetic data contains NaN after numeric coercion.")
    return synthetic.astype(reference.dtypes.to_dict())


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
            evaluator.evaluate(synthetic_data, nnaa={"n_resample": 30})
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


def save_synthetic_npz(
    synthetic: pd.DataFrame,
    output_dir: Path,
    adapter: ExternalGeneratorAdapter,
    train_size: int,
    seed: int,
    metrics: dict[str, float],
    train_path: str,
    test_path: str,
    run_metadata: dict[str, Any],
) -> str:
    path = (
        output_dir
        / "datasets"
        / "synthetic"
        / f"synthetic_{adapter.name}_{adapter.column_order}_ts{train_size}_s{seed}.npz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "algorithm": adapter.name,
        "column_order": adapter.column_order,
        "train_size": train_size,
        "seed": seed,
        "n_samples": int(len(synthetic)),
        "train_dataset_path": train_path,
        "test_dataset_path": test_path,
        "metrics": metrics,
        "run_metadata": run_metadata,
    }
    np.savez_compressed(
        path,
        synthetic_data=synthetic.to_numpy(),
        column_names=list(synthetic.columns),
        metadata=np.array(metadata, dtype=object),
    )
    return str(path)


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
    valid = df[
        (pd.to_numeric(df.get("correlation_matrix_difference"), errors="coerce") >= 0)
        & (pd.to_numeric(df.get("k_marginal_tvd"), errors="coerce") >= 0)
        & (pd.to_numeric(df.get("nnaa"), errors="coerce") >= 0)
    ]
    return {(int(row.train_size), int(row.seed)) for row in valid.itertuples(index=False)}


def run_external_baseline_protocol(
    adapter: ExternalGeneratorAdapter,
    config: ProtocolConfig,
    categorical_columns: list[str] | None = None,
) -> Path:
    """Run an external generator while fixing seeds, splits, metrics and outputs."""

    add_tabpfn_repo_to_path(config.tabpfn_repo)
    categorical_columns = categorical_columns or []
    output_dir = config.output_dir.resolve()
    csv_path = output_dir / f"result_{config.dataset}_{adapter.name}_baseline{config.output_suffix}.csv"
    completed = load_completed(csv_path) if config.resume else set()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for train_size in config.train_sizes:
        for seed in config.seeds:
            if (train_size, seed) in completed:
                continue
            started = time.time()
            set_all_seeds(seed)
            try:
                train_df, train_path = load_train_split(config.dataset_dir, train_size, seed)
                test_df, test_path = load_global_test(config.dataset_dir)
                workspace_dir = output_dir / "tmp_workspaces" / f"{adapter.name}_ts{train_size}_s{seed}"
                workspace_dir.mkdir(parents=True, exist_ok=True)
                fit_started = time.time()
                synthetic_df, run_metadata = adapter.fit_sample(
                    train_df=train_df,
                    n_samples=len(test_df),
                    seed=seed,
                    workspace_dir=workspace_dir,
                )
                fit_sample_seconds = time.time() - fit_started
                synthetic_df = coerce_like_reference(synthetic_df, test_df)
                metrics = evaluate_paper_metrics(test_df, synthetic_df, categorical_columns, seed)
                synthetic_path = ""
                if config.save_synthetic:
                    synthetic_path = save_synthetic_npz(
                        synthetic=synthetic_df,
                        output_dir=output_dir,
                        adapter=adapter,
                        train_size=train_size,
                        seed=seed,
                        metrics=metrics,
                        train_path=train_path,
                        test_path=test_path,
                        run_metadata=run_metadata,
                    )
                result = ProtocolResult(
                    dataset=config.dataset,
                    algorithm=adapter.name,
                    column_order=adapter.column_order,
                    train_size=train_size,
                    seed=seed,
                    train_dataset_path=train_path,
                    test_dataset_path=test_path,
                    synthetic_dataset_path=synthetic_path,
                    fit_sample_seconds=fit_sample_seconds,
                    total_seconds=time.time() - started,
                    correlation_matrix_difference=metrics.get("correlation_matrix_difference", -1.0),
                    k_marginal_tvd=metrics.get("k_marginal_tvd", -1.0),
                    nnaa=metrics.get("nnaa", -1.0),
                )
            except Exception as exc:
                result = ProtocolResult(
                    dataset=config.dataset,
                    algorithm=adapter.name,
                    column_order=adapter.column_order,
                    train_size=train_size,
                    seed=seed,
                    train_dataset_path="",
                    test_dataset_path="",
                    synthetic_dataset_path="",
                    fit_sample_seconds=0.0,
                    total_seconds=time.time() - started,
                    correlation_matrix_difference=-1.0,
                    k_marginal_tvd=-1.0,
                    nnaa=-1.0,
                    error=repr(exc),
                    traceback=traceback.format_exc(),
                )
            rows.append(asdict(result))
            if len(rows) >= config.save_every:
                append_rows(csv_path, rows)
                rows.clear()
    if rows:
        append_rows(csv_path, rows)

    metadata = {
        "protocol": asdict(config),
        "adapter": {
            "name": adapter.name,
            "column_order": adapter.column_order,
            "class": type(adapter).__name__,
        },
    }
    (output_dir / f"metadata_{config.dataset}_{adapter.name}_baseline{config.output_suffix}.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    return csv_path


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_seed_iterable(values: Iterable[int] | str) -> tuple[int, ...]:
    if isinstance(values, str):
        return parse_csv_ints(values)
    return tuple(int(value) for value in values)
