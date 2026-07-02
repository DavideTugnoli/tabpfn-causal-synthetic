#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


UPSTREAM_COMMIT = "3fbf35369641f87afde7999b932291a5d30ccd7b"
CUSTOM_SCM_DAG = [[3, 2], [2, 1], [0, 1]]


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DECAF under the paired cleaned-seed protocol.")
    parser.add_argument("--dataset", default="custom_scm")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--tabpfn-repo", required=True, type=Path)
    parser.add_argument("--protocol-dir", required=True, type=Path)
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-sizes", required=True)
    parser.add_argument("--seed-list", required=True)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    if args.dataset != "custom_scm":
        raise ValueError("The DECAF adapter currently defines a verified DAG only for custom_scm.")

    sys.path.insert(0, str(args.protocol_dir.parent))
    sys.path.insert(0, str(args.upstream_dir))

    from protocol.core import ExternalGeneratorAdapter, ProtocolConfig, run_external_baseline_protocol

    class DECAFAdapter(ExternalGeneratorAdapter):
        name = "decaf"
        column_order = "known_dag"

        def fit_sample(
            self,
            train_df: pd.DataFrame,
            n_samples: int,
            seed: int,
            workspace_dir: Path,
        ) -> tuple[pd.DataFrame, dict[str, object]]:
            import pytorch_lightning as pl
            import torch
            from decaf import DECAF
            from decaf.data import DataModule

            pl.seed_everything(seed, workers=True)
            means = train_df.mean(axis=0)
            stds = train_df.std(axis=0, ddof=0).replace(0, 1.0)
            standardized = (train_df - means) / stds

            batch_size = 32
            dm = DataModule(standardized.to_numpy(), batch_size=batch_size)
            model = DECAF(
                input_dim=train_df.shape[1],
                dag_seed=CUSTOM_SCM_DAG,
                h_dim=2 * train_df.shape[1],
                lr=0.001,
                batch_size=batch_size,
            )
            trainer = pl.Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                max_epochs=args.epochs,
                logger=False,
                enable_checkpointing=True,
                default_root_dir=str(workspace_dir),
                enable_progress_bar=False,
                deterministic=True,
            )
            trainer.fit(model, dm)

            rng = np.random.default_rng(seed)
            template_idx = rng.choice(len(standardized), size=n_samples, replace=True)
            template = torch.tensor(
                standardized.iloc[template_idx].to_numpy(dtype=np.float32),
                device=model.device,
            )
            with torch.no_grad():
                generated = model.gen_synthetic(template).detach().cpu().numpy()
            synthetic = pd.DataFrame(generated, columns=train_df.columns)
            synthetic = synthetic * stds + means
            return synthetic, {
                "upstream_commit": UPSTREAM_COMMIT,
                "known_dag_edges_by_column_index": CUSTOM_SCM_DAG,
                "paper_epochs": args.epochs,
                "paper_learning_rate": 0.001,
                "paper_hidden_width": 2 * train_df.shape[1],
                "standardized_continuous_inputs": True,
                "batch_size": batch_size,
                "generator_updates": int(model.iterations_g),
                "discriminator_updates": int(model.iterations_d),
                "upstream_training_schedule_note": (
                    "Current official repository updates G and D once per batch; "
                    "paper text describes one G update per ten D updates."
                ),
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
    print(run_external_baseline_protocol(DECAFAdapter(), config))


if __name__ == "__main__":
    main()

