#!/usr/bin/env python3
"""Generate static interventional datasets for the custom SCM experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent.parent
import sys
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Column layout for the numeric custom SCM
NUMERIC_COL_NAMES = ["X0", "X1", "X2", "X3"]


def _format_noise_tag(noise_level: float) -> str:
    """Format a compact, filesystem-friendly noise tag (e.g., noise1e-2)."""
    if noise_level == 0:
        return "noise0"
    sci = f"{noise_level:.0e}"
    sci = sci.replace("e-0", "e-").replace("e+0", "e+")
    return f"noise{sci}"


def generate_interventional_data(
    *,
    n_samples: int,
    x3_value: float,
    random_state: int,
    noise_level: float = 1e-5,
) -> np.ndarray:
    """Generate interventional samples with X3 fixed to ``x3_value``."""
    rng = np.random.default_rng(random_state)
    X3 = np.full(n_samples, x3_value, dtype=np.float32)
    X0 = rng.normal(0, 1, n_samples).astype(np.float32)
    X2 = (0.5 * X3 + rng.normal(0, noise_level, n_samples)).astype(np.float32)
    X1 = (5.0 * X0 + 10.0 * X2 + rng.normal(0, noise_level, n_samples)).astype(np.float32)
    return np.column_stack([X0, X1, X2, X3])


def generate_branch_csv(
    *,
    samples: int,
    x3_value: float,
    seed: int,
    mode: str,
    overwrite: bool,
    noise_level: float | None,
) -> Path:
    dataset_kind = "mixed" if mode == "mixed" else "numeric"
    base_dir = Path(__file__).parent / "generated_interventional"
    base_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"custom_{dataset_kind}_intervention_x3_eq_{x3_value:g}"
    if noise_level is not None:
        base_name = f"{base_name}_{_format_noise_tag(noise_level)}"
    file_path = base_dir / f"{base_name}.csv"
    if file_path.exists() and not overwrite:
        print(f"✅ Skipping existing file (use --overwrite to regenerate): {file_path}")
        return file_path

    noise_value = 1e-5 if noise_level is None else noise_level
    data = generate_interventional_data(
        n_samples=samples,
        x3_value=x3_value,
        random_state=seed,
        noise_level=noise_value,
    )
    df = pd.DataFrame(data, columns=NUMERIC_COL_NAMES)
    df.to_csv(file_path, index=False)
    print(f"💾 Saved branch X3={x3_value} to {file_path}")
    return file_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate static interventional datasets for the custom SCM experiment.")
    parser.add_argument("--samples-per-branch", type=int, default=2000, help="Number of samples for each intervention branch (default: 2000)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--mode", choices=["numeric"], default="numeric", help="Dataset variant (currently only numeric is supported)")
    parser.add_argument(
        "--noise-level",
        type=float,
        default=None,
        help="Noise level for SEM errors (default: 1e-5). If set, adds a noise tag to filenames.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV files")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    seed0 = int(rng.integers(0, 1_000_000))
    seed1 = int(rng.integers(0, 1_000_000))

    generate_branch_csv(
        samples=args.samples_per_branch,
        x3_value=0.0,
        seed=seed0,
        mode=args.mode,
        overwrite=args.overwrite,
        noise_level=args.noise_level,
    )
    generate_branch_csv(
        samples=args.samples_per_branch,
        x3_value=1.0,
        seed=seed1,
        mode=args.mode,
        overwrite=args.overwrite,
        noise_level=args.noise_level,
    )

    print("✅ Custom interventional dataset ready!")


if __name__ == "__main__":
    main()
