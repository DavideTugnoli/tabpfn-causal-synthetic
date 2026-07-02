#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


UPSTREAM_COMMIT = "103929336703b91794f85c6160a8338b30eb158a"
CUSTOM_SCM_EDGES = [("X3", "X2"), ("X2", "X1"), ("X0", "X1")]


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DATGAN under the paired cleaned-seed protocol.")
    parser.add_argument("--dataset", default="custom_scm")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--tabpfn-repo", required=True, type=Path)
    parser.add_argument("--protocol-dir", required=True, type=Path)
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-sizes", required=True)
    parser.add_argument("--seed-list", required=True)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()

    if args.dataset != "custom_scm":
        raise ValueError("The DATGAN adapter currently defines a verified DAG only for custom_scm.")
    sys.path.insert(0, str(args.protocol_dir.parent))
    sys.path.insert(0, str(args.upstream_dir))

    from protocol.core import ExternalGeneratorAdapter, ProtocolConfig, run_external_baseline_protocol

    class DATGANAdapter(ExternalGeneratorAdapter):
        name = "datgan"
        column_order = "known_dag"

        def fit_sample(
            self,
            train_df: pd.DataFrame,
            n_samples: int,
            seed: int,
            workspace_dir: Path,
        ) -> tuple[pd.DataFrame, dict[str, object]]:
            import tensorflow as tf
            from datgan import DATGAN

            np.random.seed(seed)
            tf.random.set_seed(seed)
            graph = nx.DiGraph()
            graph.add_nodes_from(train_df.columns)
            graph.add_edges_from(CUSTOM_SCM_EDGES)
            metadata = {
                column: {"type": "continuous", "discrete": False}
                for column in train_df.columns
            }
            batch_size = min(500, len(train_df))
            model_dir = workspace_dir / "datgan_model"
            model = DATGAN(
                output=str(model_dir),
                num_epochs=args.epochs,
                batch_size=batch_size,
                verbose=1,
            )
            model.fit(train_df, metadata=metadata, dag=graph)
            synthetic = model.sample(n_samples)
            return synthetic, {
                "upstream_commit": UPSTREAM_COMMIT,
                "known_dag_edges": CUSTOM_SCM_EDGES,
                "paper_epochs": args.epochs,
                "paper_batch_size": 500,
                "effective_batch_size": batch_size,
                "low_sample_adaptation": "min(500, N)",
                "loss_function_selected_by_upstream": model.loss_function,
                "learning_rate_selected_by_upstream": model.learning_rate,
                "g_period_selected_by_upstream": model.g_period,
                "model_dir": str(model_dir),
            }

    config = ProtocolConfig(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        train_sizes=parse_ints(args.train_sizes),
        seeds=parse_ints(args.seed_list),
        tabpfn_repo=args.tabpfn_repo,
        save_synthetic=True,
        resume=True,
        save_every=1,
        output_suffix=args.output_suffix,
    )
    print(run_external_baseline_protocol(DATGANAdapter(), config))


if __name__ == "__main__":
    main()

