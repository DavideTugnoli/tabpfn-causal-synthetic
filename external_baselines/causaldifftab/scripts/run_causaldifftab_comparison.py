#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-causaldifftab"))

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


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def copy_upstream_code(source: Path, destination: Path) -> Path:
    ignore = shutil.ignore_patterns(
        ".git",
        "__pycache__",
        "*.pyc",
        "eval/report_runs",
        "causal_masks",
        "tabdiff/ckpt",
        "tabdiff/result",
    )
    shutil.copytree(source, destination, ignore=ignore)
    patch_numeric_only_support(destination)
    return destination


def patch_numeric_only_support(code_dir: Path) -> None:
    trainer_path = code_dir / "tabdiff" / "trainer.py"
    diffusion_path = code_dir / "tabdiff" / "models" / "unified_ctime_diffusion.py"

    trainer_text = trainer_path.read_text()
    trainer_text = trainer_text.replace(
        "        x_cat_one_hot = self.diffusion.to_one_hot(x_cat)\n",
        "        if x_cat.shape[1] == 0:\n"
        "            x_cat_one_hot = torch.empty((x_cat.shape[0], 0), dtype=x_num.dtype, device=x_num.device)\n"
        "        else:\n"
        "            x_cat_one_hot = self.diffusion.to_one_hot(x_cat)\n",
    )
    trainer_text = trainer_text.replace(
        "        if os.path.exists(cat_causal_path):\n"
        "            print(\"检测到类别型因果图缓存文件，正在加载...\")\n"
        "            cat_causal_mask = np.load(cat_causal_path)\n"
        "        else:\n"
        "            cat_column_names = [f\"cat_{i}\" for i in range(x_cat_one_hot.shape[1])]\n",
        "        if x_cat_one_hot.shape[1] == 0:\n"
        "            cat_causal_mask = np.zeros((0, 0), dtype=int)\n"
        "        elif os.path.exists(cat_causal_path):\n"
        "            print(\"检测到类别型因果图缓存文件，正在加载...\")\n"
        "            cat_causal_mask = np.load(cat_causal_path)\n"
        "        else:\n"
        "            cat_column_names = [f\"cat_{i}\" for i in range(x_cat_one_hot.shape[1])]\n",
    )
    trainer_text = trainer_text.replace(
        "        end_time = time.time()\n"
        "        print_with_bar(f\"Ending Trainnig Loop, totoal training time = {end_time - start_time}\")\n",
        "        if self.model_save_path is not None:\n"
        "            state_dicts = {\n"
        "                'denoise_fn': self.ema_model.state_dict(),\n"
        "                'num_schedule': self.ema_num_schedule.state_dict(),\n"
        "                'cat_schedule': self.ema_cat_schedule.state_dict(),\n"
        "            }\n"
        "            torch.save(state_dicts, os.path.join(self.model_save_path, f'best_ema_model_final_{self.curr_epoch}.pt'))\n"
        "        end_time = time.time()\n"
        "        print_with_bar(f\"Ending Trainnig Loop, totoal training time = {end_time - start_time}\")\n",
    )
    trainer_path.write_text(trainer_text)

    diffusion_text = diffusion_path.read_text()
    diffusion_text = diffusion_text.replace(
        "    def to_one_hot(self, x_cat):\n"
        "        x_cat_oh = torch.cat(\n",
        "    def to_one_hot(self, x_cat):\n"
        "        if len(self.num_classes) == 0:\n"
        "            return torch.empty((x_cat.shape[0], 0), dtype=torch.float32, device=x_cat.device)\n"
        "        x_cat_oh = torch.cat(\n",
    )
    diffusion_path.write_text(diffusion_text)


def prepare_official_dataset(
    workspace: Path,
    dataname: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    data_dir = workspace / "data" / dataname
    synthetic_dir = workspace / "synthetic" / dataname
    data_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    train_numeric = train_df.apply(pd.to_numeric, errors="coerce")
    test_numeric = test_df.apply(pd.to_numeric, errors="coerce")
    if train_numeric.isna().any().any() or test_numeric.isna().any().any():
        raise ValueError("CausalDiffTab wrapper currently supports numeric comparison datasets only.")

    column_names = list(train_df.columns)
    if len(column_names) < 2:
        raise ValueError("At least two columns are required for the CausalDiffTab regression layout.")

    target_index = len(column_names) - 1
    feature_indices = [index for index in range(len(column_names)) if index != target_index]

    train_array = train_numeric.to_numpy(dtype=np.float32, copy=True)
    test_array = test_numeric.to_numpy(dtype=np.float32, copy=True)

    np.save(data_dir / "X_num_train.npy", train_array[:, feature_indices])
    np.save(data_dir / "X_num_test.npy", test_array[:, feature_indices])
    np.save(data_dir / "X_cat_train.npy", np.empty((len(train_array), 0), dtype=np.int64))
    np.save(data_dir / "X_cat_test.npy", np.empty((len(test_array), 0), dtype=np.int64))
    np.save(data_dir / "y_train.npy", train_array[:, target_index])
    np.save(data_dir / "y_test.npy", test_array[:, target_index])

    train_numeric.to_csv(synthetic_dir / "real.csv", index=False)
    test_numeric.to_csv(synthetic_dir / "test.csv", index=False)

    idx_mapping = {str(original_index): mapped_index for mapped_index, original_index in enumerate(feature_indices)}
    idx_mapping[str(target_index)] = len(feature_indices)
    inverse_idx_mapping = {str(value): int(key) for key, value in idx_mapping.items()}
    idx_name_mapping = {str(index): name for index, name in enumerate(column_names)}

    info: dict[str, Any] = {
        "name": dataname,
        "task_type": "regression",
        "header": "infer",
        "column_names": column_names,
        "num_col_idx": feature_indices,
        "cat_col_idx": [],
        "target_col_idx": [target_index],
        "file_type": "npy",
        "data_path": str(data_dir / "real.csv"),
        "test_path": str(synthetic_dir / "test.csv"),
        "val_path": None,
        "train_num": int(len(train_df)),
        "test_num": int(len(test_df)),
        "int_col_idx_wrt_num": [],
        "idx_mapping": idx_mapping,
        "inverse_idx_mapping": inverse_idx_mapping,
        "idx_name_mapping": idx_name_mapping,
    }
    (data_dir / "info.json").write_text(json.dumps(info, indent=2))
    return info


def find_checkpoint(code_dir: Path, dataname: str, exp_name: str) -> Path:
    ckpt_root = code_dir / "tabdiff" / "ckpt" / dataname / exp_name
    candidates = sorted(
        ckpt_root.glob("best_ema_model_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            ckpt_root.glob("model_*.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        raise FileNotFoundError(f"No CausalDiffTab checkpoint found in {ckpt_root}")
    return candidates[0]


def find_sample_csv(code_dir: Path, dataname: str, exp_name: str) -> Path:
    result_root = code_dir / "tabdiff" / "result" / dataname / exp_name
    candidates = sorted(
        result_root.glob("**/samples.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No generated samples.csv found in {result_root}")
    return candidates[0]


def run_official_train_and_sample(
    causaldifftab_repo: Path,
    workspace_root: Path,
    dataname: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    device: str,
    debug: bool,
    keep_workspace: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    workspace = workspace_root / dataname
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    code_dir = copy_upstream_code(causaldifftab_repo.resolve(), workspace / "code")
    config_path = code_dir / "tabdiff" / "configs" / "tabdiff_configs.toml"
    config_text = config_path.read_text()
    config_text = config_text.replace("check_val_every = 2000", "check_val_every = 100000000")
    if debug:
        config_text = config_text.replace("steps = 8000", "steps = 4")
    config_path.write_text(config_text)
    prepare_official_dataset(workspace, dataname, train_df, test_df)

    exp_name = "learnable_schedule"
    gpu_arg = "-1" if device == "cpu" else "0"
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["PYTHONPATH"] = str(code_dir)
    env.setdefault("WANDB_MODE", "disabled")

    set_all_seeds(seed)
    train_cmd = [
        sys.executable,
        str(code_dir / "main.py"),
        "--dataname",
        dataname,
        "--mode",
        "train",
        "--gpu",
        gpu_arg,
        "--no_wandb",
    ]
    subprocess.run(train_cmd, cwd=workspace, env=env, check=True)

    ckpt_path = find_checkpoint(code_dir, dataname, exp_name)
    test_cmd = [
        sys.executable,
        str(code_dir / "main.py"),
        "--dataname",
        dataname,
        "--mode",
        "test",
        "--gpu",
        gpu_arg,
        "--no_wandb",
        "--ckpt_path",
        str(ckpt_path),
        "--num_samples_to_generate",
        str(len(test_df)),
    ]
    test_result = subprocess.run(test_cmd, cwd=workspace, env=env, check=False)
    sample_path = find_sample_csv(code_dir, dataname, exp_name)
    if test_result.returncode != 0 and not sample_path.exists():
        raise subprocess.CalledProcessError(test_result.returncode, test_cmd)
    synthetic = pd.read_csv(sample_path)
    metadata = {
        "workspace": str(workspace),
        "checkpoint_path": str(ckpt_path),
        "sample_path": str(sample_path),
        "debug": debug,
    }
    if not keep_workspace:
        shutil.rmtree(workspace, ignore_errors=True)
    return synthetic, metadata


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
        results["correlation_matrix_difference"] = float(corr_diff)
    except Exception:
        numeric_columns = [column for column in real_data.columns if column not in categorical_columns]
        real_corr = real_data[numeric_columns].corr(method="spearman").to_numpy(copy=True)
        synthetic_corr = synthetic_data[numeric_columns].corr(method="spearman").to_numpy(copy=True)
        np.fill_diagonal(real_corr, 1.0)
        np.fill_diagonal(synthetic_corr, 1.0)
        results["correlation_matrix_difference"] = float(np.linalg.norm(real_corr - synthetic_corr, ord="fro"))

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
    train_size: int,
    seed: int,
    metrics: dict[str, float],
    train_path: str,
    test_path: str,
    run_metadata: dict[str, Any],
) -> str:
    path = output_dir / "datasets" / "synthetic" / f"synthetic_causaldifftab_unconditional_ts{train_size}_s{seed}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "algorithm": "causaldifftab",
        "column_order": "unconditional",
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


def run_one(
    dataset: str,
    dataset_dir: Path,
    causaldifftab_repo: Path,
    output_dir: Path,
    train_size: int,
    seed: int,
    device: str,
    save_synthetic: bool,
    debug: bool,
    keep_workspace: bool,
) -> dict[str, Any]:
    started = time.time()
    set_all_seeds(seed)
    train_df, train_path = load_train_split(dataset_dir, train_size, seed)
    test_df, test_path = load_global_test(dataset_dir)

    workspace_root = output_dir / "tmp_workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)
    run_name = sanitize_name(f"{dataset}_ts{train_size}_s{seed}")

    train_started = time.time()
    synthetic_df, run_metadata = run_official_train_and_sample(
        causaldifftab_repo=causaldifftab_repo,
        workspace_root=workspace_root,
        dataname=run_name,
        train_df=train_df,
        test_df=test_df,
        seed=seed,
        device=device,
        debug=debug,
        keep_workspace=keep_workspace,
    )
    fit_sample_seconds = time.time() - train_started

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
            run_metadata=run_metadata,
        )

    return {
        "dataset": dataset,
        "algorithm": "causaldifftab",
        "column_order": "unconditional",
        "train_size": train_size,
        "seed": seed,
        "repetition": None,
        "train_dataset_path": train_path,
        "test_dataset_path": test_path,
        "synthetic_dataset_path": synthetic_path,
        "fit_sample_seconds": fit_sample_seconds,
        "total_seconds": time.time() - started,
        "debug": debug,
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
    parser = argparse.ArgumentParser(description="Run CausalDiffTab baseline on cached comparison NPZ splits.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--causaldifftab-repo", required=True, type=Path)
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--save-synthetic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--keep-workspace", action="store_true")
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
    csv_path = output_dir / f"result_{args.dataset}_causaldifftab_baseline{args.output_suffix}.csv"
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
                    causaldifftab_repo=args.causaldifftab_repo.resolve(),
                    output_dir=output_dir,
                    train_size=train_size,
                    seed=seed,
                    device=args.device,
                    save_synthetic=args.save_synthetic,
                    debug=args.debug,
                    keep_workspace=args.keep_workspace,
                )
            except Exception as exc:
                if args.fail_fast:
                    raise
                row = {
                    "dataset": args.dataset,
                    "algorithm": "causaldifftab",
                    "column_order": "unconditional",
                    "train_size": train_size,
                    "seed": seed,
                    "debug": args.debug,
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
        "causaldifftab_repo": str(args.causaldifftab_repo),
        "train_sizes": train_sizes,
        "seed_start": args.seed_start,
        "seed_list": explicit_seeds,
        "repetitions": args.repetitions,
        "device": args.device,
        "save_synthetic": args.save_synthetic,
        "debug": args.debug,
    }
    (output_dir / f"metadata_{args.dataset}_causaldifftab_baseline{args.output_suffix}.json").write_text(
        json.dumps(metadata, indent=2)
    )


if __name__ == "__main__":
    main()
