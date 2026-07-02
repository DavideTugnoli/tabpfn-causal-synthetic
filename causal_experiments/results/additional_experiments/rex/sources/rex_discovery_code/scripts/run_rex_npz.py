#!/usr/bin/env python3
"""Run REX on a single NPZ train set and save DAGs + metrics.

This script loads an NPZ produced by the CSuite experiments (train_ts*_s*.npz),
fits REX (NN + GBT), combines DAGs (union or intersection), and saves outputs.

Discovery engine: ReX from the causalexplain library
(https://github.com/renero/causalexplain), v0.9.1, with a one-line pandas dtype
fix in causalexplain/explainability/shapley.py (see
../causalexplain_shapley_dtype.patch). Camera-ready config: bootstrap_trials=20,
combine=union, no HPO. Output edges_idx encodes [parent_idx, child_idx].
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import pickle
from sklearn.preprocessing import StandardScaler

from causalexplain.estimators.rex import Rex
from causalexplain.common import utils
from causalexplain.metrics.compare_graphs import evaluate_graph


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def _resolve_bootstrap_sampling_split(
    value: str | float,
    n_samples: int,
    bootstrap_trials: int,
    min_bootstrap_rows: int,
) -> str | float:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            return "auto"
        if normalized == "auto-safe":
            auto_split = 1.0 - np.exp(-4.605170185988091 / max(1, bootstrap_trials))
            min_split = min(1.0, max(0.05, float(min_bootstrap_rows) / max(1, n_samples)))
            return float(max(auto_split, min_split))
        try:
            value = float(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Invalid bootstrap sampling split '{value}'. "
                "Use auto, auto-safe, or a float in (0, 1]."
            ) from exc

    split = float(value)
    if split <= 0.0 or split > 1.0:
        raise ValueError(
            f"bootstrap sampling split must be in (0, 1], got {split}"
        )
    return split


def _load_npz(npz_path: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    with np.load(str(npz_path), allow_pickle=True) as data:
        if "X_train" not in data:
            raise ValueError(f"Missing X_train in {npz_path}")
        X_train = data["X_train"]
        if "column_names" in data:
            column_names = data["column_names"].tolist()
        else:
            column_names = [f"x{i}" for i in range(X_train.shape[1])]
        metadata = {}
        for key in ("train_size", "seed", "seed_base"):
            if key in data:
                val = data[key]
                metadata[key] = int(val) if np.ndim(val) == 0 else val.tolist()
        if "metadata" in data:
            meta_val = data["metadata"]
            if isinstance(meta_val, np.ndarray) and meta_val.shape == ():
                meta_val = meta_val.item()
            metadata["metadata"] = meta_val
    df = pd.DataFrame(X_train, columns=column_names)
    return df, metadata


def _load_true_dag(dataset_dir: Path) -> Optional[nx.DiGraph]:
    adj_path = dataset_dir / "adj_matrix.csv"
    vars_path = dataset_dir / "variables.json"
    if not adj_path.exists() or not vars_path.exists():
        return None
    adj = np.loadtxt(str(adj_path), delimiter=",")
    with vars_path.open() as f:
        vars_json = json.load(f)
    names = [v["name"] for v in vars_json.get("variables", [])]
    if not names:
        return None
    if adj.shape[0] != len(names) or adj.shape[1] != len(names):
        raise ValueError("Adjacency matrix shape does not match variables.json")
    g = nx.DiGraph()
    g.add_nodes_from(names)
    for i, src in enumerate(names):
        for j, dst in enumerate(names):
            if adj[i, j] != 0:
                g.add_edge(src, dst)
    return g


def _run_rex_model(
    model_type: str,
    data: pd.DataFrame,
    run_name: str,
    bootstrap_trials: int,
    bootstrap_sampling_split: str,
    bootstrap_tolerance: str,
    parallel_jobs: int,
    bootstrap_parallel_jobs: int,
    random_state: int,
    use_default_pipeline: bool,
    tune_model: bool,
    hpo_trials: int,
    device: Optional[str],
) -> Rex:
    rex_kwargs = {
        "name": run_name,
        "model_type": model_type,
        "tune_model": tune_model,
        "bootstrap_trials": bootstrap_trials,
        "bootstrap_sampling_split": bootstrap_sampling_split,
        "bootstrap_tolerance": bootstrap_tolerance,
        "parallel_jobs": parallel_jobs,
        "bootstrap_parallel_jobs": bootstrap_parallel_jobs,
        "random_state": random_state,
    }
    if hpo_trials is not None:
        rex_kwargs["hpo_n_trials"] = hpo_trials
    if device:
        rex_kwargs["device"] = device
    rex = Rex(**rex_kwargs)
    if use_default_pipeline:
        rex.fit(data)
    else:
        fit_pipeline = [
            ("models", rex.model_type),
            ("models.fit", {}),
            ("models.score", {}),
        ]
        rex.fit(data, pipeline=fit_pipeline)
    rex.predict(data)
    return rex


def _save_graph(graph: nx.DiGraph, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dot_path = out_dir / f"{name}.dot"
    utils.graph_to_dot_file(graph, dot_path)
    with (out_dir / f"{name}.gpickle").open("wb") as f:
        pickle.dump(graph, f)


def _export_dag_json(
    graph: nx.DiGraph,
    out_dir: Path,
    name: str,
    column_names: list[str],
    meta: Dict[str, object],
    args: argparse.Namespace,
    bootstrap_sampling_split_resolved: str | float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    name_to_idx = {col: idx for idx, col in enumerate(column_names)}

    edges = [[str(u), str(v)] for u, v in graph.edges()]
    edges_idx = []
    missing_nodes = set()
    for u, v in graph.edges():
        if u in name_to_idx and v in name_to_idx:
            edges_idx.append([name_to_idx[u], name_to_idx[v]])
        else:
            missing_nodes.add(str(u))
            missing_nodes.add(str(v))

    payload = {
        "algorithm": {
            "name": "rex",
            "graph_type": "dag",
        },
        "column_names": column_names,
        "edges": edges,
        "edges_idx": edges_idx,
        "n_features": len(column_names),
        "train_size": meta.get("train_size"),
        "seed": meta.get("seed"),
        "seed_base": meta.get("seed_base"),
        "method": "rex",
        "combine_op": args.combine_op,
        "bootstrap_trials": args.bootstrap_trials,
        "bootstrap_sampling_split": args.bootstrap_sampling_split,
        "bootstrap_sampling_split_resolved": bootstrap_sampling_split_resolved,
        "bootstrap_tolerance": args.bootstrap_tolerance,
        "parallel_jobs": args.parallel_jobs,
        "bootstrap_parallel_jobs": args.bootstrap_parallel_jobs,
        "tune_model": args.tune_model,
        "hpo_trials": args.hpo_trials,
        "scale": args.scale,
        "device": args.device,
    }
    if missing_nodes:
        payload["missing_nodes"] = sorted(missing_nodes)

    json_path = out_dir / f"{name}.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run REX on NPZ train set")
    parser.add_argument("--npz", required=True, help="Path to train_ts*.npz")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset-dir", required=False, help="CSuite dataset dir for true DAG")
    parser.add_argument("--combine-op", choices=["union", "intersection"], default="union")
    parser.add_argument("--bootstrap-trials", type=int, default=20)
    parser.add_argument(
        "--bootstrap-sampling-split",
        default="auto",
        help="Bootstrap split: auto | auto-safe | float in (0,1]",
    )
    parser.add_argument("--bootstrap-tolerance", default="auto")
    parser.add_argument(
        "--min-bootstrap-rows",
        type=int,
        default=10,
        help="Minimum sampled rows when --bootstrap-sampling-split=auto-safe",
    )
    parser.add_argument("--parallel-jobs", type=int, default=4)
    parser.add_argument("--bootstrap-parallel-jobs", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=1234)
    parser.add_argument("--use-default-pipeline", action="store_true")
    parser.add_argument("--tune-model", action="store_true")
    parser.add_argument("--hpo-trials", type=int, default=20)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--scale", action="store_true", help="Standardize features (mean=0, std=1)")
    parser.add_argument("--export-dag-json", action="store_true", help="Export final DAG as JSON (names and indices)")
    args = parser.parse_args()

    npz_path = Path(args.npz).resolve()
    out_dir = Path(args.output_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else None

    out_dir.mkdir(parents=True, exist_ok=True)

    data, meta = _load_npz(npz_path)
    if args.scale:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data.values)
        data = pd.DataFrame(scaled, columns=data.columns, index=data.index)

    bootstrap_sampling_split = _resolve_bootstrap_sampling_split(
        args.bootstrap_sampling_split,
        int(data.shape[0]),
        args.bootstrap_trials,
        args.min_bootstrap_rows,
    )
    run_id = npz_path.stem

    seed = int(meta.get("seed", args.random_state))
    _set_seeds(seed)

    start = time.time()
    rex_nn = _run_rex_model(
        "nn",
        data,
        run_id + "_nn",
        args.bootstrap_trials,
        bootstrap_sampling_split,
        args.bootstrap_tolerance,
        args.parallel_jobs,
        args.bootstrap_parallel_jobs,
        seed,
        args.use_default_pipeline,
        args.tune_model,
        args.hpo_trials,
        args.device,
    )
    nn_time = time.time() - start

    start = time.time()
    rex_gbt = _run_rex_model(
        "gbt",
        data,
        run_id + "_gbt",
        args.bootstrap_trials,
        bootstrap_sampling_split,
        args.bootstrap_tolerance,
        args.parallel_jobs,
        args.bootstrap_parallel_jobs,
        seed,
        args.use_default_pipeline,
        args.tune_model,
        args.hpo_trials,
        args.device,
    )
    gbt_time = time.time() - start

    union, inter, union_fixed, inter_fixed = utils.combine_dags(
        rex_nn.dag,
        rex_gbt.dag,
        discrepancies=rex_nn.shaps.shap_discrepancies,
        prior=None,
    )

    if args.combine_op == "union":
        final_dag = union_fixed
        final_name = "rex_union"
    else:
        final_dag = inter_fixed
        final_name = "rex_intersection"

    _save_graph(rex_nn.dag, out_dir, "rex_nn")
    _save_graph(rex_gbt.dag, out_dir, "rex_gbt")
    _save_graph(union, out_dir, "rex_union_raw")
    _save_graph(inter, out_dir, "rex_intersection_raw")
    _save_graph(union_fixed, out_dir, "rex_union")
    _save_graph(inter_fixed, out_dir, "rex_intersection")
    if args.export_dag_json:
        _export_dag_json(
            final_dag,
            out_dir,
            final_name,
            list(data.columns),
            meta,
            args,
            bootstrap_sampling_split,
        )

    metrics = None
    true_dag = None
    if dataset_dir is not None:
        true_dag = _load_true_dag(dataset_dir)
    if true_dag is not None:
        metrics_obj = evaluate_graph(true_dag, final_dag, feature_names=list(data.columns))
        metrics = metrics_obj.to_dict() if metrics_obj is not None else None

    summary = {
        "algorithm": {
            "name": "rex",
            "graph_type": "dag",
        },
        "npz": str(npz_path),
        "output_dir": str(out_dir),
        "run_id": run_id,
        "train_size": meta.get("train_size"),
        "seed": meta.get("seed"),
        "seed_base": meta.get("seed_base"),
        "n_samples": int(data.shape[0]),
        "n_features": int(data.shape[1]),
        "bootstrap_trials": args.bootstrap_trials,
        "bootstrap_sampling_split": args.bootstrap_sampling_split,
        "bootstrap_sampling_split_resolved": bootstrap_sampling_split,
        "bootstrap_tolerance": args.bootstrap_tolerance,
        "parallel_jobs": args.parallel_jobs,
        "bootstrap_parallel_jobs": args.bootstrap_parallel_jobs,
        "use_default_pipeline": args.use_default_pipeline,
        "tune_model": args.tune_model,
        "hpo_trials": args.hpo_trials,
        "device": args.device,
        "scale": args.scale,
        "combine_op": args.combine_op,
        "time_nn_sec": nn_time,
        "time_gbt_sec": gbt_time,
        "metrics": metrics,
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    print("[INFO] REX run completed")
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
