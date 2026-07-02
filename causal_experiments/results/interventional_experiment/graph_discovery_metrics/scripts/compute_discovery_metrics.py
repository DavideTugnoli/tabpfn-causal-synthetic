#!/usr/bin/env python3
"""Compute CPDAG discovery quality metrics for interventional experiment results.

This script scans interventional experiment CSVs, extracts CPDAGs used in
`cpdag_discovered` runs, and computes discovery metrics against per-run
interventional DAG ground truths recovered from `dag` + `topological` rows.

Outputs:
- discovery_metrics_per_run.csv: per-run metrics
- discovery_metrics_summary.csv: aggregated metrics per dataset and train size
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (parent for parent in SCRIPT_DIR.parents if (parent / "pyproject.toml").exists()),
    SCRIPT_DIR.parents[4],
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_COLUMNS = {"algorithm", "graph_structure", "train_size", "seed"}


def _find_result_csvs(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []

    # Canonical source for the cleaned interventional data.
    data_dir = repo_root / "causal_experiments" / "results" / "interventional_experiment" / "data"
    if data_dir.exists():
        candidates.extend(sorted(data_dir.glob("*.csv")))
        if candidates:
            return candidates

    # Fallback to broader discovery only if canonical cleaned CSVs are missing.
    custom_root = repo_root / "causal_experiments" / "custom_scm_experiment" / "interventional_experiment"
    if custom_root.exists():
        candidates.extend(sorted(custom_root.glob("results*/**/*.csv")))

    other_results = repo_root / "causal_experiments" / "results"
    if other_results.exists():
        candidates.extend(sorted(other_results.glob("**/*intervention*_experiment*.csv")))
        candidates.extend(sorted(other_results.glob("**/*interventional_experiment*.csv")))

    # De-duplicate while preserving order
    seen = set()
    unique = []
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _parse_dataset_id(path: Path) -> tuple[str, str]:
    """Return (dataset_id, source) from a results CSV path."""
    stem = path.stem
    if stem.startswith("result_csuite_"):
        # result_csuite_mixed_confounding_intervention_experiment_cleaned_reps_100
        match = re.search(
            r"result_(csuite_[^_]+(?:_[^_]+)*)_(?:intervention|interventional)_experiment",
            stem,
        )
        if match:
            return match.group(1), "csuite"
    path_str = str(path)
    if "custom_scm_noise1e-2" in stem or "results_noise1e-2" in path_str:
        return "custom_scm_noise1e-2", "custom"
    if "custom_scm_experiment" in path_str or "custom_scm" in stem:
        return "custom_scm", "custom"
    # Fallback to stem
    return stem, "unknown"


def _dag_dict_to_adj(dag_dict: dict[int, list[int]]) -> np.ndarray:
    n_nodes = max(dag_dict.keys()) + 1 if dag_dict else 0
    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    for child, parents in dag_dict.items():
        for parent in parents:
            adj[parent, child] = 1
    return adj


def _load_dag(row_graph: str) -> dict[int, list[int]]:
    parsed = ast.literal_eval(row_graph)
    out: dict[int, list[int]] = {}
    for child, parents in parsed.items():
        c = int(child)
        out[c] = [int(parent) for parent in parents]
    return out


def _parse_actual_column_order(order_value: Any, n_nodes: int) -> list[str]:
    if isinstance(order_value, str) and order_value.strip():
        names = [part.strip() for part in order_value.split(",") if part.strip()]
        if len(names) == n_nodes:
            return names
    return [str(i) for i in range(n_nodes)]


def _remap_adj_between_orders(
    adj: np.ndarray,
    source_order: list[str],
    target_order: list[str],
) -> np.ndarray | None:
    n_nodes = adj.shape[0]
    if len(source_order) != n_nodes or len(target_order) != n_nodes:
        return None
    if len(set(source_order)) != n_nodes or len(set(target_order)) != n_nodes:
        return None
    if set(source_order) != set(target_order):
        return None

    target_idx = {name: idx for idx, name in enumerate(target_order)}
    remapped = np.zeros_like(adj)
    for src_parent in range(n_nodes):
        src_parent_name = source_order[src_parent]
        tgt_parent = target_idx[src_parent_name]
        children = np.where(adj[src_parent] == 1)[0]
        for src_child in children:
            src_child_name = source_order[src_child]
            tgt_child = target_idx[src_child_name]
            remapped[tgt_parent, tgt_child] = 1
    return remapped


def _true_dag_for_dataset(dataset_id: str, source: str, repo_root: Path, n_nodes_hint: int | None) -> np.ndarray:
    if source == "csuite":
        from causal_experiments.utils.csuite_loader import load_csuite_dataset

        csuite = load_csuite_dataset(
            dataset_id,
            base_path=repo_root / "causal_experiments" / "csuite_experiment" / "csuite_datasets",
        )
        return csuite["dag"]

    if source == "custom":
        # Resolve custom SCM DAGs without importing heavy experiment utilities.
        # This keeps metrics computation independent from optional torch installs.
        from causal_experiments.utils.scm_data import get_mixed_dag_and_config, get_numeric_dag_and_config

        if n_nodes_hint == 5:
            dag_dict, _, _ = get_mixed_dag_and_config()
        else:
            dag_dict, _, _ = get_numeric_dag_and_config()
        return _dag_dict_to_adj(dag_dict)

    raise ValueError(f"Unknown dataset source: {source} for {dataset_id}")


def _cpdag_to_sets(cpdag_dict: dict[int, dict[str, list[int]]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    directed: set[tuple[int, int]] = set()
    undirected: set[tuple[int, int]] = set()

    for child, info in cpdag_dict.items():
        for parent in info.get("parents", []):
            directed.add((parent, child))
        for other in info.get("undirected", []):
            a, b = sorted((child, other))
            undirected.add((a, b))

    # If an edge appears directed, drop its undirected representation
    for parent, child in list(directed):
        a, b = sorted((parent, child))
        undirected.discard((a, b))

    return directed, undirected


def _skeleton_from_directed(directed: Iterable[tuple[int, int]], undirected: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    skeleton = set(undirected)
    skeleton.update(tuple(sorted((i, j))) for (i, j) in directed)
    return skeleton


def _v_structures_from_adj(adj: np.ndarray) -> set[tuple[int, int, int]]:
    n = adj.shape[0]
    skeleton = {(min(i, j), max(i, j)) for i in range(n) for j in range(n) if adj[i, j] == 1}
    v_structs = set()
    for child in range(n):
        parents = [p for p in range(n) if adj[p, child] == 1]
        if len(parents) < 2:
            continue
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                a, b = parents[i], parents[j]
                pair = (min(a, b), max(a, b))
                if pair not in skeleton:
                    v_structs.add((pair[0], pair[1], child))
    return v_structs


def _v_structures_from_cpdag(cpdag_dict: dict[int, dict[str, list[int]]]) -> set[tuple[int, int, int]]:
    directed, undirected = _cpdag_to_sets(cpdag_dict)
    skeleton = _skeleton_from_directed(directed, undirected)

    parents_map: dict[int, list[int]] = {node: info.get("parents", []) for node, info in cpdag_dict.items()}

    v_structs = set()
    for child, parents in parents_map.items():
        if len(parents) < 2:
            continue
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                a, b = parents[i], parents[j]
                pair = (min(a, b), max(a, b))
                if pair not in skeleton:
                    v_structs.add((pair[0], pair[1], child))
    return v_structs


def _compute_metrics(cpdag_dict: dict[int, dict[str, list[int]]], true_adj: np.ndarray) -> dict[str, float]:
    n = true_adj.shape[0]

    true_directed = {(i, j) for i in range(n) for j in range(n) if true_adj[i, j] == 1}
    true_skeleton = {tuple(sorted((i, j))) for (i, j) in true_directed}
    true_v_structs = _v_structures_from_adj(true_adj)

    directed, undirected = _cpdag_to_sets(cpdag_dict)
    pred_skeleton = _skeleton_from_directed(directed, undirected)
    pred_v_structs = _v_structures_from_cpdag(cpdag_dict)

    sk_tp = len([e for e in pred_skeleton if e in true_skeleton])
    sk_recall = sk_tp / len(true_skeleton) if true_skeleton else float("nan")
    sk_precision = sk_tp / len(pred_skeleton) if pred_skeleton else float("nan")
    sk_shd = len(true_skeleton.symmetric_difference(pred_skeleton))

    if directed:
        dir_tp = len([e for e in directed if e in true_directed])
        dir_precision = dir_tp / len(directed)
    else:
        dir_precision = float("nan")

    dir_recall = len([e for e in directed if e in true_directed]) / len(true_directed) if true_directed else float("nan")

    v_tp = len([v for v in pred_v_structs if v in true_v_structs])
    v_precision = v_tp / len(pred_v_structs) if pred_v_structs else float("nan")
    v_recall = v_tp / len(true_v_structs) if true_v_structs else float("nan")
    v_f1 = float("nan")
    if v_precision == v_precision and v_recall == v_recall and (v_precision + v_recall) > 0:
        v_f1 = 2 * v_precision * v_recall / (v_precision + v_recall)

    return {
        "skeleton_recall": sk_recall,
        "skeleton_precision": sk_precision,
        "skeleton_shd": float(sk_shd),
        "direction_precision": dir_precision,
        "direction_recall": dir_recall,
        "v_structure_precision": v_precision,
        "v_structure_recall": v_recall,
        "v_structure_f1": v_f1,
        "directed_edges": float(len(directed)),
        "undirected_edges": float(len(undirected)),
        "skeleton_edges": float(len(pred_skeleton)),
        "v_structures": float(len(pred_v_structs)),
    }


def _load_cpdag(row_graph: str) -> dict[int, dict[str, list[int]]]:
    parsed = ast.literal_eval(row_graph)
    return {int(k): v for k, v in parsed.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CPDAG discovery metrics from results CSVs.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root (default: project root).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for CSVs (default: scripts/outputs/cpdag_discovery_metrics).",
    )
    parser.add_argument(
        "--algorithm",
        default="cpdag_discovered",
        help="Algorithm to analyze (default: cpdag_discovered).",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = SCRIPT_DIR / "outputs" / "cpdag_discovery_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = _find_result_csvs(repo_root)
    if not csv_paths:
        raise SystemExit("No result CSVs found.")

    per_run_rows: list[dict[str, Any]] = []

    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if not REQUIRED_COLUMNS.issubset(df.columns):
            continue

        algo_df = df[df["algorithm"] == args.algorithm]
        if algo_df.empty:
            continue

        dataset_id, source = _parse_dataset_id(path)

        key_cols = ["train_size", "seed"]
        if "repetition" in df.columns:
            key_cols.append("repetition")

        dag_lookup: dict[tuple[Any, ...], tuple[np.ndarray, list[str]]] = {}
        if {"column_order", "actual_column_order"}.issubset(df.columns):
            dag_df = df[(df["algorithm"] == "dag") & (df["column_order"] == "topological")]
            for _, dag_row in dag_df.iterrows():
                dag_graph_value = dag_row.get("graph_structure", "")
                if dag_graph_value in ("", "no_graph") or pd.isna(dag_graph_value):
                    continue
                try:
                    dag_dict = _load_dag(dag_graph_value)
                except Exception:
                    continue

                dag_adj = _dag_dict_to_adj(dag_dict)
                dag_order = _parse_actual_column_order(
                    dag_row.get("actual_column_order", ""),
                    dag_adj.shape[0],
                )
                key = tuple(dag_row.get(col) for col in key_cols)
                dag_lookup[key] = (dag_adj, dag_order)

        # Fallback to static dataset DAG only if per-run interventional DAGs are unavailable.
        fallback_true_adj = None
        if not dag_lookup:
            n_nodes_hint = None
            for graph_value in algo_df["graph_structure"].dropna().tolist():
                if graph_value == "no_graph":
                    continue
                try:
                    cpdag = _load_cpdag(graph_value)
                    n_nodes_hint = len(cpdag)
                    break
                except Exception:
                    continue
            try:
                fallback_true_adj = _true_dag_for_dataset(dataset_id, source, repo_root, n_nodes_hint)
            except Exception:
                fallback_true_adj = None

        for _, row in algo_df.iterrows():
            graph_value = row.get("graph_structure", "")
            if graph_value in ("", "no_graph") or pd.isna(graph_value):
                continue

            try:
                cpdag = _load_cpdag(graph_value)
            except Exception:
                continue

            cp_order = _parse_actual_column_order(
                row.get("actual_column_order", ""),
                len(cpdag),
            )

            true_adj = None
            key = tuple(row.get(col) for col in key_cols)
            dag_entry = dag_lookup.get(key)
            if dag_entry is not None:
                dag_adj, dag_order = dag_entry
                if dag_adj.shape[0] == len(cpdag):
                    true_adj = _remap_adj_between_orders(dag_adj, dag_order, cp_order)

            if true_adj is None and fallback_true_adj is not None and fallback_true_adj.shape[0] == len(cpdag):
                true_adj = fallback_true_adj
            if true_adj is None:
                continue

            metrics = _compute_metrics(cpdag, true_adj)
            per_run_rows.append({
                "dataset": dataset_id,
                "source": source,
                "result_file": str(path.relative_to(repo_root)),
                "train_size": row.get("train_size"),
                "seed": row.get("seed"),
                "repetition": row.get("repetition"),
                "noise_level": row.get("noise_level", ""),
                **metrics,
            })

    if not per_run_rows:
        raise SystemExit("No CPDAG discovery rows found.")

    per_run_df = pd.DataFrame(per_run_rows)
    per_run_path = output_dir / "discovery_metrics_per_run.csv"
    per_run_df.to_csv(per_run_path, index=False)

    # Summary per dataset and train_size
    summary = (
        per_run_df
        .groupby(["dataset", "source", "train_size", "noise_level"], dropna=False)
        .agg(
            skeleton_recall_mean=("skeleton_recall", "mean"),
            skeleton_recall_std=("skeleton_recall", "std"),
            skeleton_precision_mean=("skeleton_precision", "mean"),
            skeleton_precision_std=("skeleton_precision", "std"),
            skeleton_shd_mean=("skeleton_shd", "mean"),
            skeleton_shd_std=("skeleton_shd", "std"),
            direction_precision_mean=("direction_precision", "mean"),
            direction_precision_std=("direction_precision", "std"),
            direction_recall_mean=("direction_recall", "mean"),
            direction_recall_std=("direction_recall", "std"),
            v_structure_precision_mean=("v_structure_precision", "mean"),
            v_structure_precision_std=("v_structure_precision", "std"),
            v_structure_recall_mean=("v_structure_recall", "mean"),
            v_structure_recall_std=("v_structure_recall", "std"),
            v_structure_f1_mean=("v_structure_f1", "mean"),
            v_structure_f1_std=("v_structure_f1", "std"),
            directed_edges_mean=("directed_edges", "mean"),
            directed_edges_std=("directed_edges", "std"),
            skeleton_edges_mean=("skeleton_edges", "mean"),
            skeleton_edges_std=("skeleton_edges", "std"),
            v_structures_mean=("v_structures", "mean"),
            v_structures_std=("v_structures", "std"),
            runs=("seed", "count"),
            runs_with_directed=("directed_edges", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )

    summary_path = output_dir / "discovery_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved per-run metrics to: {per_run_path}")
    print(f"Saved summary metrics to: {summary_path}")


if __name__ == "__main__":
    main()
