#!/usr/bin/env python3
"""Generate a static dataset for the custom SCM comparison experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure project root on path
project_root = Path(__file__).parent.parent.parent.parent
import sys
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from causal_experiments.utils import get_experimental_configs  # type: ignore
from causal_experiments.utils.scm_data import (  # type: ignore
    generate_numeric_scm_data,
    generate_mixed_scm_data,
)


def _format_noise_tag(noise_level: float) -> str:
    """Format a compact, filesystem-friendly noise tag (e.g., noise1e-2)."""
    if noise_level == 0:
        return "noise0"
    sci = f"{noise_level:.0e}"
    sci = sci.replace("e-0", "e-").replace("e+0", "e+")
    return f"noise{sci}"


def _build_graph_payload(use_categorical: bool) -> tuple[dict[int, list[int]], dict[int, dict[str, list[int]]], list[str]]:
    configs = get_experimental_configs(use_categorical=use_categorical)

    dag = configs["dag"]["dag"]
    cpdag = configs["cpdag_minimal"]["cpdag"]
    dag_dict = {int(k): [int(v) for v in parents] for k, parents in dag.items()}
    cpdag_dict = {
        int(k): {
            "parents": [int(x) for x in value.get("parents", [])],
            "undirected": [int(x) for x in value.get("undirected", [])],
        }
        for k, value in cpdag.items()
    }
    categorical_cols = ["X4_cat"] if use_categorical else []
    return dag_dict, cpdag_dict, categorical_cols


def generate_dataset(
    *,
    mode: str,
    samples: int,
    seed: int,
    overwrite: bool,
    noise_level: float | None,
) -> None:
    base_dir = Path(__file__).parent / "generated_static_scm"
    base_dir.mkdir(parents=True, exist_ok=True)

    dataset_kind = "mixed" if mode == "mixed" else "numeric"
    base_name = f"custom_{dataset_kind}_scm_{samples}"
    if noise_level is not None:
        base_name = f"{base_name}_{_format_noise_tag(noise_level)}"
    csv_path = base_dir / f"{base_name}.csv"
    graphs_path = base_dir / f"{base_name}.graphs.json"

    if not overwrite and csv_path.exists() and graphs_path.exists():
        print(f"✅ Dataset already exists: {csv_path} (use --overwrite to regenerate)")
        return

    default_noise = 0.3 if mode == "mixed" else 1e-5
    noise_value = default_noise if noise_level is None else noise_level

    if mode == "mixed":
        data = generate_mixed_scm_data(n_samples=samples, random_state=seed, noise_level=noise_value)
        column_names = ["X0", "X1", "X2", "X3", "X4_cat"]
        categorical_cols = ["X4_cat"]
    else:
        data = generate_numeric_scm_data(n_samples=samples, random_state=seed, noise_level=noise_value)
        column_names = ["X0", "X1", "X2", "X3"]
        categorical_cols = []

    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(csv_path, index=False)

    dag_dict, cpdag_dict, categorical_cols_from_configs = _build_graph_payload(mode == "mixed")

    graphs_payload: dict[str, Any] = {
        "column_names": column_names,
        "categorical_columns": categorical_cols or categorical_cols_from_configs,
        "noise_level": noise_value,
        "dag_dict": {str(k): v for k, v in dag_dict.items()},
        "cpdag_dict": {
            str(k): {
                "parents": value["parents"],
                "undirected": value["undirected"],
            }
            for k, value in cpdag_dict.items()
        },
    }

    with graphs_path.open("w", encoding="utf-8") as f:
        json.dump(graphs_payload, f, indent=2)

    print(f"✅ Generated dataset saved to {csv_path}")
    print(f"✅ Graph metadata saved to {graphs_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate static dataset for custom SCM comparison experiment.")
    parser.add_argument("--samples", type=int, default=6000, help="Number of samples to generate (default: 6000)")
    parser.add_argument(
        "--mode",
        choices=["numeric", "mixed"],
        default="numeric",
        help="Dataset variant to generate (default: numeric)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation (default: 42)")
    parser.add_argument(
        "--noise-level",
        type=float,
        default=None,
        help=(
            "Noise level for SEM errors. If omitted, uses defaults (numeric: 1e-5, mixed: 0.3) "
            "and legacy filenames."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset files if present")
    args = parser.parse_args()

    generate_dataset(
        mode=args.mode,
        samples=args.samples,
        seed=args.seed,
        overwrite=args.overwrite,
        noise_level=args.noise_level,
    )


if __name__ == "__main__":
    main()
