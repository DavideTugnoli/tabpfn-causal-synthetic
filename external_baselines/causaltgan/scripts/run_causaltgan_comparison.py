#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


UPSTREAM_COMMIT = "5d5b35bc3ae96e9ce6b7a4fc7cd90ca7679ff859"
CAUSAL_GRAPH = [
    ["X0", []],
    ["X1", ["X0", "X2"]],
    ["X2", ["X3"]],
    ["X3", []],
]


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Causal-TGAN under the paired cleaned-seed protocol.")
    parser.add_argument("--dataset", default="custom_scm")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--tabpfn-repo", required=True, type=Path)
    parser.add_argument("--protocol-dir", required=True, type=Path)
    parser.add_argument("--upstream-parent", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-sizes", required=True)
    parser.add_argument("--seed-list", required=True)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    sys.path.insert(0, str(args.protocol_dir.parent))
    sys.path.insert(0, str(args.upstream_parent))

    from protocol.core import ExternalGeneratorAdapter, ProtocolConfig, run_external_baseline_protocol

    class CausalTGANAdapter(ExternalGeneratorAdapter):
        name = "causaltgan"
        column_order = "dag"

        def fit_sample(
            self,
            train_df: pd.DataFrame,
            n_samples: int,
            seed: int,
            workspace_dir: Path,
        ) -> tuple[pd.DataFrame, dict[str, object]]:
            import rdt.transformers
            import torch
            from torch.utils.data import DataLoader

            # Upstream targets the legacy RDT class name. The custom SCM is
            # continuous-only, so this compatibility alias is not exercised.
            if not hasattr(rdt.transformers, "OneHotEncodingTransformer"):
                from rdt.transformers.categorical import OneHotEncoder

                rdt.transformers.OneHotEncodingTransformer = OneHotEncoder

            from CausalTGAN.configuration import CausalTGANConfig, TrainingOptions
            from CausalTGAN.dataset import DataTransformer, NumpyDataset
            from CausalTGAN.helper.feature_info import FeatureINFO
            from CausalTGAN.model.causalTGAN import CausalTGAN

            transformer = DataTransformer()
            transformer.fit(train_df, discrete_columns=())
            transformed, dimensions = transformer.transform(train_df)
            feature_info = FeatureINFO(list(train_df.columns), [], dimensions)
            train_options = TrainingOptions(
                batch_size=500,
                number_of_epochs=400,
                runs_folder=str(workspace_dir),
                experiment_name=f"causaltgan_ts{len(train_df)}_s{seed}",
            )
            config = CausalTGANConfig(
                causal_graph=CAUSAL_GRAPH,
                z_dim=2,
                pac_num=1,
                D_iter=3,
            )
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            loader = DataLoader(NumpyDataset(transformed), batch_size=500, shuffle=True)
            model = CausalTGAN(device, config, feature_info, transformer)
            model.fit(loader, train_options, full_knowledge=True, verbose=False)
            with torch.no_grad():
                sampled = model.sample(n_samples).detach().cpu().numpy()
            synthetic = transformer.inverse_transform(sampled)
            return synthetic, {
                "upstream_commit": UPSTREAM_COMMIT,
                "official_defaults": {
                    "batch_size": 500,
                    "epochs": 400,
                    "pac_num": 1,
                    "z_dim": 2,
                    "D_iter": 3,
                    "transformer_type": "ctgan",
                },
                "causal_graph": CAUSAL_GRAPH,
                "checkpoint_dir": str(workspace_dir / "checkpoints"),
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
    csv_path = run_external_baseline_protocol(CausalTGANAdapter(), config)
    print(csv_path)


if __name__ == "__main__":
    main()
