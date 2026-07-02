#!/usr/bin/env python3
"""Screen custom SCM parameterizations for PC discovery quality.

This script is intentionally limited to causal discovery. It does not run
TabPFN or generate synthetic data. The goal is to identify SCM parameters for
which PC-stable reliably recovers the Markov-equivalent CPDAG before launching
expensive generation experiments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_experiments.utils.run_pc_discovery import run_pc_discovery_on_dataset

COLUMN_NAMES = ["X0", "X1", "X2", "X3"]
TRUE_DAG = {
    0: [],
    1: [0, 2],
    2: [3],
    3: [],
}
TRUE_DIRECTED_EDGES = {(0, 1), (2, 1), (3, 2)}
TRUE_SKELETON = {tuple(sorted(edge)) for edge in TRUE_DIRECTED_EDGES}
TRUE_V_STRUCTURES = {(0, 2, 1)}


def format_noise_tag(noise_level: float) -> str:
    """Format a compact filesystem-friendly noise tag."""
    if noise_level == 0:
        return "noise0"
    sci = f"{noise_level:.0e}"
    sci = sci.replace("e-0", "e-").replace("e+0", "e+")
    return f"noise{sci}"


def generate_numeric_scm_data(
    n_samples: int,
    random_state: int,
    *,
    x2_from_x3: float,
    x0_to_x1: float,
    x2_to_x1: float,
    noise_x2: float,
    noise_x1: float,
) -> np.ndarray:
    """Generate X3 -> X2 -> X1 <- X0 with configurable linear coefficients."""
    rng = np.random.default_rng(random_state)
    x3 = rng.normal(0, 1, n_samples)
    x0 = rng.normal(0, 1, n_samples)
    x2 = x2_from_x3 * x3 + rng.normal(0, noise_x2, n_samples)
    x1 = x0_to_x1 * x0 + x2_to_x1 * x2 + rng.normal(0, noise_x1, n_samples)
    return np.column_stack([x0, x1, x2, x3]).astype(np.float64)


def causal_learn_cpdag_to_dict(cpdag_adj: np.ndarray) -> dict[int, dict[str, list[int]]]:
    """Convert a causal-learn CPDAG matrix to the project CPDAG dictionary."""
    n_nodes = cpdag_adj.shape[0]
    cpdag: dict[int, dict[str, list[int]]] = {}
    for child in range(n_nodes):
        parents = [parent for parent in range(n_nodes) if cpdag_adj[child, parent] == 1]
        undirected = [
            other
            for other in range(n_nodes)
            if other != child
            and cpdag_adj[child, other] == -1
            and cpdag_adj[other, child] == -1
        ]
        cpdag[child] = {"parents": parents, "undirected": undirected}
    return cpdag


def cpdag_to_sets(
    cpdag: dict[int, dict[str, list[int]]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    """Return directed, undirected, and skeleton edge sets from a CPDAG dict."""
    directed: set[tuple[int, int]] = set()
    undirected: set[tuple[int, int]] = set()

    for child, info in cpdag.items():
        for parent in info.get("parents", []):
            directed.add((parent, child))
        for other in info.get("undirected", []):
            undirected.add(tuple(sorted((child, other))))

    for edge in list(directed):
        undirected.discard(tuple(sorted(edge)))

    skeleton = set(undirected)
    skeleton.update(tuple(sorted(edge)) for edge in directed)
    return directed, undirected, skeleton


def v_structures_from_cpdag(cpdag: dict[int, dict[str, list[int]]]) -> set[tuple[int, int, int]]:
    """Extract oriented unshielded colliders from a CPDAG dict."""
    _, _, skeleton = cpdag_to_sets(cpdag)
    v_structures: set[tuple[int, int, int]] = set()
    for child, info in cpdag.items():
        parents = info.get("parents", [])
        for left_idx in range(len(parents)):
            for right_idx in range(left_idx + 1, len(parents)):
                left, right = sorted((parents[left_idx], parents[right_idx]))
                if (left, right) not in skeleton:
                    v_structures.add((left, right, child))
    return v_structures


def compute_discovery_metrics(cpdag: dict[int, dict[str, list[int]]]) -> dict[str, float | int | bool]:
    """Compute graph recovery metrics against the true custom SCM DAG."""
    directed, undirected, skeleton = cpdag_to_sets(cpdag)
    v_structures = v_structures_from_cpdag(cpdag)

    skeleton_tp = len(skeleton & TRUE_SKELETON)
    directed_tp = len(directed & TRUE_DIRECTED_EDGES)
    v_structure_tp = len(v_structures & TRUE_V_STRUCTURES)

    skeleton_precision = skeleton_tp / len(skeleton) if skeleton else 0.0
    direction_precision = directed_tp / len(directed) if directed else 0.0
    v_structure_precision = v_structure_tp / len(v_structures) if v_structures else 0.0

    exact_markov_cpdag = skeleton == TRUE_SKELETON and v_structures == TRUE_V_STRUCTURES
    exact_directed_dag = directed == TRUE_DIRECTED_EDGES and not undirected

    return {
        "skeleton_recall": skeleton_tp / len(TRUE_SKELETON),
        "skeleton_precision": skeleton_precision,
        "skeleton_shd": len(skeleton.symmetric_difference(TRUE_SKELETON)),
        "direction_recall": directed_tp / len(TRUE_DIRECTED_EDGES),
        "direction_precision": direction_precision,
        "v_structure_recall": v_structure_tp / len(TRUE_V_STRUCTURES),
        "v_structure_precision": v_structure_precision,
        "exact_markov_cpdag": exact_markov_cpdag,
        "exact_directed_dag": exact_directed_dag,
        "directed_edges": len(directed),
        "undirected_edges": len(undirected),
        "skeleton_edges": len(skeleton),
        "extra_skeleton_edges": len(skeleton - TRUE_SKELETON),
        "missing_skeleton_edges": len(TRUE_SKELETON - skeleton),
        "x1_parent_set_exact": set(cpdag[1]["parents"]) == {0, 2},
        "x2_x3_undirected": 3 in cpdag[2]["undirected"] and 2 in cpdag[3]["undirected"],
    }


def build_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build candidate SCM parameterizations from CLI arguments."""
    candidates: list[dict[str, Any]] = []
    for noise_level in args.noise_levels:
        candidates.append(
            {
                "candidate": format_noise_tag(noise_level),
                "x2_from_x3": args.x2_from_x3,
                "x0_to_x1": args.x0_to_x1,
                "x2_to_x1": args.x2_to_x1,
                "noise_x2": noise_level,
                "noise_x1": noise_level,
            }
        )

    if args.include_balanced_candidate:
        candidates.append(
            {
                "candidate": "balanced_linear",
                "x2_from_x3": 1.0,
                "x0_to_x1": 1.5,
                "x2_to_x1": 1.5,
                "noise_x2": 0.7,
                "noise_x1": 0.5,
            }
        )

    return candidates


def sample_train_sets(
    x_all: np.ndarray,
    *,
    train_sizes: list[int],
    repetitions: int,
    test_size: int,
    test_seed: int,
) -> dict[tuple[int, int], np.ndarray]:
    """Mimic the comparison experiment's global-test then train-pool sampling."""
    if test_size >= len(x_all):
        raise ValueError("test_size must be smaller than the generated dataset.")

    rng = np.random.default_rng(test_seed)
    test_indices = rng.choice(len(x_all), size=test_size, replace=False)
    mask = np.ones(len(x_all), dtype=bool)
    mask[test_indices] = False
    train_pool = x_all[mask]

    train_sets: dict[tuple[int, int], np.ndarray] = {}
    linear_seed = 0
    for train_size in train_sizes:
        for _ in range(repetitions):
            split_rng = np.random.default_rng(linear_seed)
            replace = train_size > len(train_pool)
            sampled_indices = split_rng.choice(len(train_pool), size=train_size, replace=replace)
            train_sets[(train_size, linear_seed)] = train_pool[sampled_indices]
            linear_seed += 1
    return train_sets


def run_screen(args: argparse.Namespace) -> None:
    """Run the discovery screen and write CSV outputs."""
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "causal_experiments"
            / "results"
            / "custom_scm_discovery_search"
            / "runs"
            / args.run_name
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates(args)
    per_run_rows: list[dict[str, Any]] = []

    for candidate in candidates:
        x_all = generate_numeric_scm_data(
            args.samples,
            args.dataset_seed,
            x2_from_x3=candidate["x2_from_x3"],
            x0_to_x1=candidate["x0_to_x1"],
            x2_to_x1=candidate["x2_to_x1"],
            noise_x2=candidate["noise_x2"],
            noise_x1=candidate["noise_x1"],
        )
        train_sets = sample_train_sets(
            x_all,
            train_sizes=args.train_sizes,
            repetitions=args.repetitions,
            test_size=args.test_size,
            test_seed=args.test_seed,
        )

        for (train_size, split_seed), x_train in train_sets.items():
            cpdag_object = run_pc_discovery_on_dataset(
                dataset_name="continuous",
                data=x_train,
                col_names=COLUMN_NAMES,
                categorical_cols=[],
                true_dag=TRUE_DAG,
                alpha=args.alpha,
                indep_test=args.indep_test,
            )
            cpdag_matrix = cpdag_object if isinstance(cpdag_object, np.ndarray) else cpdag_object.G.graph
            cpdag = causal_learn_cpdag_to_dict(cpdag_matrix)
            metrics = compute_discovery_metrics(cpdag)

            per_run_rows.append(
                {
                    "candidate": candidate["candidate"],
                    "train_size": train_size,
                    "split_seed": split_seed,
                    "alpha": args.alpha,
                    "indep_test": args.indep_test,
                    "x2_from_x3": candidate["x2_from_x3"],
                    "x0_to_x1": candidate["x0_to_x1"],
                    "x2_to_x1": candidate["x2_to_x1"],
                    "noise_x2": candidate["noise_x2"],
                    "noise_x1": candidate["noise_x1"],
                    "cpdag": json.dumps(cpdag, sort_keys=True),
                    **metrics,
                }
            )

    per_run = pd.DataFrame(per_run_rows)
    per_run_path = output_dir / "discovery_screen_per_run.csv"
    per_run.to_csv(per_run_path, index=False)

    summary = (
        per_run.groupby(
            [
                "candidate",
                "train_size",
                "alpha",
                "indep_test",
                "x2_from_x3",
                "x0_to_x1",
                "x2_to_x1",
                "noise_x2",
                "noise_x1",
            ],
            dropna=False,
        )
        .agg(
            skeleton_recall_mean=("skeleton_recall", "mean"),
            skeleton_precision_mean=("skeleton_precision", "mean"),
            skeleton_shd_mean=("skeleton_shd", "mean"),
            direction_recall_mean=("direction_recall", "mean"),
            direction_precision_mean=("direction_precision", "mean"),
            v_structure_recall_mean=("v_structure_recall", "mean"),
            v_structure_precision_mean=("v_structure_precision", "mean"),
            exact_markov_cpdag_rate=("exact_markov_cpdag", "mean"),
            exact_directed_dag_rate=("exact_directed_dag", "mean"),
            x1_parent_set_exact_rate=("x1_parent_set_exact", "mean"),
            x2_x3_undirected_rate=("x2_x3_undirected", "mean"),
            extra_skeleton_edges_mean=("extra_skeleton_edges", "mean"),
            missing_skeleton_edges_mean=("missing_skeleton_edges", "mean"),
            runs=("split_seed", "count"),
        )
        .reset_index()
    )
    summary_path = output_dir / "discovery_screen_summary.csv"
    summary.to_csv(summary_path, index=False)

    candidate_summary = (
        summary.groupby(
            [
                "candidate",
                "alpha",
                "indep_test",
                "x2_from_x3",
                "x0_to_x1",
                "x2_to_x1",
                "noise_x2",
                "noise_x1",
            ],
            dropna=False,
        )
        .agg(
            exact_markov_cpdag_rate_mean=("exact_markov_cpdag_rate", "mean"),
            v_structure_recall_mean=("v_structure_recall_mean", "mean"),
            skeleton_recall_mean=("skeleton_recall_mean", "mean"),
            skeleton_precision_mean=("skeleton_precision_mean", "mean"),
            extra_skeleton_edges_mean=("extra_skeleton_edges_mean", "mean"),
            runs=("runs", "sum"),
        )
        .reset_index()
        .sort_values(
            [
                "exact_markov_cpdag_rate_mean",
                "v_structure_recall_mean",
                "skeleton_recall_mean",
                "skeleton_precision_mean",
            ],
            ascending=[False, False, False, False],
        )
    )
    candidate_summary_path = output_dir / "discovery_screen_candidate_summary.csv"
    candidate_summary.to_csv(candidate_summary_path, index=False)

    config = {
        "samples": args.samples,
        "dataset_seed": args.dataset_seed,
        "test_size": args.test_size,
        "test_seed": args.test_seed,
        "train_sizes": args.train_sizes,
        "repetitions": args.repetitions,
        "alpha": args.alpha,
        "indep_test": args.indep_test,
        "candidates": candidates,
    }
    with (output_dir / "screen_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"Saved per-run metrics to {per_run_path}")
    print(f"Saved summary metrics to {summary_path}")
    print(f"Saved candidate summary to {candidate_summary_path}")
    print(candidate_summary.head(10).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="noise_sweep_default")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--test-seed", type=int, default=2000)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[20, 50, 100, 200, 500])
    parser.add_argument("--repetitions", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--indep-test", default="fisherz")
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[1e-5, 1e-2, 1e-1, 2e-1, 3e-1, 5e-1, 7e-1, 1.0],
    )
    parser.add_argument("--x2-from-x3", type=float, default=0.5)
    parser.add_argument("--x0-to-x1", type=float, default=5.0)
    parser.add_argument("--x2-to-x1", type=float, default=10.0)
    parser.add_argument("--include-balanced-candidate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_screen(parse_args())
