#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CTGAN under the paired cleaned-seed protocol.")
    parser.add_argument("--dataset", default="custom_scm")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--tabpfn-repo", required=True, type=Path)
    parser.add_argument("--protocol-dir", required=True, type=Path)
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-sizes", required=True)
    parser.add_argument("--seed-list", required=True)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    sys.path.insert(0, str(args.protocol_dir.parent))
    sys.path.insert(0, str(args.upstream_dir))

    from protocol.core import ExternalGeneratorAdapter, ProtocolConfig, run_external_baseline_protocol

    class CTGANAdapter(ExternalGeneratorAdapter):
        name = "ctgan"
        column_order = "unconditional"

        def fit_sample(
            self,
            train_df: pd.DataFrame,
            n_samples: int,
            seed: int,
            workspace_dir: Path,
        ) -> tuple[pd.DataFrame, dict[str, object]]:
            from ctgan import CTGAN

            model = CTGAN()
            model.set_random_state(seed)
            model.fit(train_df, discrete_columns=())
            synthetic = model.sample(n_samples)
            model_path = workspace_dir / "ctgan_model.pkl"
            model.save(model_path)
            return synthetic, {
                "upstream_commit": "f4fcd21d96e291fb1d6b7a14b83236927560b81e",
                "constructor": "CTGAN()",
                "official_defaults": True,
                "model_path": str(model_path),
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
    csv_path = run_external_baseline_protocol(CTGANAdapter(), config)
    print(csv_path)


if __name__ == "__main__":
    main()

