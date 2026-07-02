#!/usr/bin/env python3
"""Run small TabPFN generation smokes for custom SCM candidates.

This script is for candidate selection only. It keeps outputs under the
discovery-search results tree and should not be treated as a final experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_experiments.custom_scm_experiment.comparison_experiment.comparison_experiment import (
    run_single_experiment,
)
from causal_experiments.utils import (
    create_result_row,
    discover_cpdag_from_data,
    prepare_cpdag_data,
    prepare_dag_data,
    prepare_vanilla_data,
    save_results_to_csv,
    set_experiment_seeds,
    setup_determinism,
)
from causal_experiments.utils.dag_utils import dag_to_ideal_cpdag

COLUMN_NAMES = ["X0", "X1", "X2", "X3"]


def generate_single_collider_noise3e1(n_samples: int, random_state: int) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Generate the current single-collider SCM with stronger SEM noise."""
    rng = np.random.default_rng(random_state)
    x3 = rng.normal(0, 1, n_samples)
    x0 = rng.normal(0, 1, n_samples)
    x2 = 0.5 * x3 + rng.normal(0, 0.3, n_samples)
    x1 = 5.0 * x0 + 10.0 * x2 + rng.normal(0, 0.3, n_samples)
    dag = {
        0: [],
        1: [0, 2],
        2: [3],
        3: [],
    }
    return np.column_stack([x0, x1, x2, x3]).astype(np.float64), dag


def generate_double_collider(n_samples: int, random_state: int) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Generate a two-collider SCM with identifiable PC directions.

    Structure:
        X2 -> X0 <- X3
        X2 -> X1 <- X3

    The original column order [X0, X1, X2, X3] puts both collider children
    before their parents, making vanilla original intentionally unfavorable.
    """
    rng = np.random.default_rng(random_state)
    x2 = rng.normal(0, 1, n_samples)
    x3 = rng.normal(0, 1, n_samples)
    x0 = 2.0 * x2 + 2.0 * x3 + rng.normal(0, 0.3, n_samples)
    x1 = 2.0 * x2 + 3.0 * x3 + rng.normal(0, 0.3, n_samples)
    dag = {
        0: [2, 3],
        1: [2, 3],
        2: [],
        3: [],
    }
    return np.column_stack([x0, x1, x2, x3]).astype(np.float64), dag


CANDIDATES: dict[str, Callable[[int, int], tuple[np.ndarray, dict[int, list[int]]]]] = {
    "single_collider_noise3e-1": generate_single_collider_noise3e1,
    "double_collider": generate_double_collider,
}


def sample_train_sets(
    x_all: np.ndarray,
    *,
    train_sizes: list[int],
    repetitions: int,
    test_size: int,
    test_seed: int,
) -> tuple[pd.DataFrame, dict[tuple[int, int, int], np.ndarray]]:
    """Create one global test set and deterministic train subsets."""
    rng = np.random.default_rng(test_seed)
    test_indices = rng.choice(len(x_all), size=test_size, replace=False)
    mask = np.ones(len(x_all), dtype=bool)
    mask[test_indices] = False

    train_pool = x_all[mask]
    test_df = pd.DataFrame(x_all[test_indices], columns=COLUMN_NAMES)

    train_sets: dict[tuple[int, int, int], np.ndarray] = {}
    linear_seed = 0
    for train_size in train_sizes:
        for repetition in range(1, repetitions + 1):
            split_rng = np.random.default_rng(linear_seed)
            sampled_indices = split_rng.choice(
                len(train_pool),
                size=train_size,
                replace=train_size > len(train_pool),
            )
            train_sets[(train_size, repetition, linear_seed)] = train_pool[sampled_indices]
            linear_seed += 1
    return test_df, train_sets


def run_algorithm(
    *,
    algorithm: str,
    x_train_original: np.ndarray,
    test_df: pd.DataFrame,
    dag: dict[int, list[int]],
    seed: int,
    train_size: int,
    repetition: int,
    n_estimators: int,
    n_permutations: int,
    temperature: float,
) -> dict[str, Any]:
    """Run one algorithm on one train split and return a result row."""
    column_order = "original"
    categorical_cols: list[str] = []
    graph_dict: dict[str, Any] | dict[int, Any] | None = None

    if algorithm == "vanilla":
        x_train_prepared, column_ordering_used, updated_categorical_features = prepare_vanilla_data(
            x_train_original,
            column_order,
            dag,
            COLUMN_NAMES,
            None,
        )
        dag_prepared = None
        cpdag_prepared = None
    elif algorithm == "dag":
        x_train_prepared, dag_prepared = prepare_dag_data(x_train_original, dag, COLUMN_NAMES)
        cpdag_prepared = None
        column_ordering_used = None
        updated_categorical_features = None
        graph_dict = dag
    elif algorithm == "cpdag_minimal":
        cpdag_to_use = dag_to_ideal_cpdag(dag)
        x_train_prepared, cpdag_prepared = prepare_cpdag_data(x_train_original, cpdag_to_use, COLUMN_NAMES)
        dag_prepared = None
        column_ordering_used = None
        updated_categorical_features = None
        graph_dict = cpdag_to_use
    elif algorithm == "cpdag_discovered":
        cpdag_to_use = discover_cpdag_from_data(
            x_train_original,
            COLUMN_NAMES,
            categorical_cols,
            use_categorical=False,
            true_dag=dag,
            alpha=0.05,
            indep_test="fisherz",
        )
        x_train_prepared, cpdag_prepared = prepare_cpdag_data(x_train_original, cpdag_to_use, COLUMN_NAMES)
        dag_prepared = None
        column_ordering_used = None
        updated_categorical_features = None
        graph_dict = cpdag_to_use
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    metrics, _synthetic_df = run_single_experiment(
        X_train=x_train_prepared,
        test_df=test_df,
        algorithm=algorithm,
        column_order=column_order,
        dag=dag_prepared,
        cpdag=cpdag_prepared,
        column_names=COLUMN_NAMES,
        categorical_cols=categorical_cols,
        column_ordering_used=column_ordering_used,
        updated_categorical_features=updated_categorical_features,
        n_permutations=n_permutations,
        temp=temperature,
        n_estimators=n_estimators,
        seed=seed,
        causal_structures_last=False,
    )

    result_row = create_result_row(
        algorithm=algorithm,
        column_order=column_order,
        graph_dict=graph_dict,
        train_size=train_size,
        seed=seed,
        repetition=repetition,
        categorical_cols=categorical_cols,
        column_names=COLUMN_NAMES,
        metrics_results=metrics,
        reordered_graph_dict=None,
        column_ordering_used=column_ordering_used,
    )
    result_row["actual_column_order"] = ",".join(COLUMN_NAMES)
    return result_row


def run_smoke(args: argparse.Namespace) -> None:
    """Run the candidate smoke."""
    setup_determinism(
        enable_cuda_determinism=True,
        cublas_workspace_config=":4096:8",
        set_num_threads=1,
        verbose=True,
    )

    if args.candidate not in CANDIDATES:
        raise ValueError(f"Unknown candidate {args.candidate}. Available: {sorted(CANDIDATES)}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "causal_experiments"
            / "results"
            / "custom_scm_discovery_search"
            / "runs"
            / f"generation_smoke_{args.candidate}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    x_all, dag = CANDIDATES[args.candidate](args.samples, args.dataset_seed)
    test_df, train_sets = sample_train_sets(
        x_all,
        train_sizes=args.train_sizes,
        repetitions=args.repetitions,
        test_size=args.test_size,
        test_seed=args.test_seed,
    )

    config = {
        "candidate": args.candidate,
        "samples": args.samples,
        "dataset_seed": args.dataset_seed,
        "test_size": args.test_size,
        "test_seed": args.test_seed,
        "train_sizes": args.train_sizes,
        "repetitions": args.repetitions,
        "algorithms": args.algorithms,
        "dag": dag,
        "n_estimators": args.n_estimators,
        "n_permutations": args.n_permutations,
        "temperature": args.temperature,
    }
    with (output_dir / "smoke_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    results: list[dict[str, Any]] = []
    start = time.time()
    total = len(train_sets) * len(args.algorithms)
    counter = 0

    for algorithm in args.algorithms:
        for (train_size, repetition, seed), x_train in train_sets.items():
            counter += 1
            print(
                f"[{counter}/{total}] candidate={args.candidate} "
                f"algorithm={algorithm} train_size={train_size} repetition={repetition} seed={seed}"
            )
            set_experiment_seeds(seed, include_cuda=True, verbose=False)
            result = run_algorithm(
                algorithm=algorithm,
                x_train_original=x_train,
                test_df=test_df,
                dag=dag,
                seed=seed,
                train_size=train_size,
                repetition=repetition,
                n_estimators=args.n_estimators,
                n_permutations=args.n_permutations,
                temperature=args.temperature,
            )
            result["candidate"] = args.candidate
            results.append(result)

            if len(results) % args.save_every == 0:
                save_results_to_csv(results, str(output_dir / "generation_smoke_results.csv"))

    save_results_to_csv(results, str(output_dir / "generation_smoke_results.csv"))
    elapsed = time.time() - start
    print(f"Completed smoke in {elapsed / 60:.2f} minutes")

    df = pd.DataFrame(results)
    summary = (
        df.groupby(["candidate", "train_size", "algorithm", "column_order"], dropna=False)[
            ["correlation_matrix_difference", "k_marginal_tvd", "nnaa"]
        ]
        .median(numeric_only=True)
        .reset_index()
    )
    summary.to_csv(output_dir / "generation_smoke_summary.csv", index=False)
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="double_collider", choices=sorted(CANDIDATES))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--test-seed", type=int, default=2000)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--algorithms", nargs="+", default=["vanilla", "cpdag_discovered"])
    parser.add_argument("--n-estimators", type=int, default=3)
    parser.add_argument("--n-permutations", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run_smoke(parse_args())
