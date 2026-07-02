#!/usr/bin/env python3
"""Build and validate the lightweight final CI-preservation result bundle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


FINAL_DIR = Path(__file__).resolve().parent
CI_SCRIPT = FINAL_DIR.parent / "conditional_independence_preservation.py"
KEY = ["source", "dataset", "condition", "sample_size", "repetition"]


def normalize_schema(data: pd.DataFrame) -> pd.DataFrame:
    """Remove the legacy duplicate denominator field after validating it."""
    if "n_triples_tested" not in data:
        return data
    legacy = pd.to_numeric(data["n_triples_tested"], errors="coerce")
    canonical = pd.to_numeric(data["n_reference_independent"], errors="coerce")
    if not legacy.equals(canonical):
        raise RuntimeError("Legacy n_triples_tested differs from n_reference_independent")
    return data.drop(columns=["n_triples_tested"])


def load_ci_module():
    spec = importlib.util.spec_from_file_location("ci_preservation_core", CI_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {CI_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_cells(root: Path) -> dict[tuple[str, int], pd.DataFrame]:
    cells: dict[tuple[str, int], pd.DataFrame] = {}
    if not root.exists():
        return cells
    for path in sorted(root.glob("*/ci_preservation_results.csv")):
        try:
            data = pd.read_csv(path)
        except EmptyDataError:
            continue
        if data.empty:
            continue
        data = normalize_schema(data)
        key = (str(data["dataset"].iloc[0]), int(data["sample_size"].iloc[0]))
        cells[key] = data
    return cells


def select_cells(
    roots: list[Path],
    allowed_sizes: set[int],
) -> tuple[dict[tuple[str, int], pd.DataFrame], list[dict[str, object]]]:
    selected: dict[tuple[str, int], pd.DataFrame] = {}
    provenance: list[dict[str, object]] = []
    for root in roots:
        for key, data in read_cells(root).items():
            if key[1] not in allowed_sizes or key in selected:
                continue
            selected[key] = data
            provenance.append({"dataset": key[0], "sample_size": key[1], "source_root": str(root)})
    return selected, provenance


def apply_exact_patches(cells: dict[tuple[str, int], pd.DataFrame], patch_root: Path) -> int:
    patched = 0
    for path in sorted(patch_root.glob("*/ci_preservation_results.csv")):
        patch = normalize_schema(pd.read_csv(path))
        for _, row in patch.iterrows():
            cell_key = (str(row["dataset"]), int(row["sample_size"]))
            target = cells[cell_key]
            mask = np.logical_and.reduce([target[column] == row[column] for column in KEY])
            if int(mask.sum()) != 1:
                raise RuntimeError(f"Expected one patch target for {cell_key}, found {int(mask.sum())}")
            target.loc[mask, :] = row.reindex(target.columns).to_numpy()
            patched += 1
    return patched


def validate_scope(scope: str, cells: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (dataset, sample_size), data in sorted(cells.items()):
        counts = data.groupby("condition", dropna=False)["repetition"].agg(["size", "nunique"])
        duplicate_rows = int(data.duplicated(KEY, keep=False).sum())
        fraction = pd.to_numeric(data["fraction_preserved"], errors="coerce")
        undefined = ~np.isfinite(fraction)
        documented_undefined = undefined & (
            pd.to_numeric(data["n_reference_independent"], errors="coerce").fillna(-1).eq(0)
        )
        invalid_nonfinite = undefined & ~documented_undefined
        invalid_status = ~data["status"].eq("ok")
        structural_valid = (
            len(data) == len(counts) * 100
            and counts["size"].eq(100).all()
            and counts["nunique"].eq(100).all()
            and duplicate_rows == 0
        )
        records.append(
            {
                "scope": scope,
                "dataset": dataset,
                "sample_size": sample_size,
                "n_conditions": len(counts),
                "n_rows": len(data),
                "min_rows_per_condition": int(counts["size"].min()),
                "max_rows_per_condition": int(counts["size"].max()),
                "min_unique_repetitions": int(counts["nunique"].min()),
                "max_unique_repetitions": int(counts["nunique"].max()),
                "duplicate_key_rows": duplicate_rows,
                "invalid_status_rows": int(invalid_status.sum()),
                "invalid_nonfinite_rows": int(invalid_nonfinite.sum()),
                "documented_undefined_reference_rows": int(documented_undefined.sum()),
                "valid": bool(structural_valid and not invalid_status.any() and not invalid_nonfinite.any()),
            }
        )
    return pd.DataFrame.from_records(records)


def sanitize_paths(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for column in ["reference_path", "synthetic_path"]:
        if column in data:
            data[column] = data[column].map(
                lambda value: Path(str(value)).name if pd.notna(value) else value
            )
    return data


def write_scope(scope: str, cells: dict[tuple[str, int], pd.DataFrame], ci) -> pd.DataFrame:
    data = pd.concat([cells[key] for key in sorted(cells)], ignore_index=True)
    data = sanitize_paths(data).sort_values(KEY).reset_index(drop=True)
    data.to_csv(FINAL_DIR / f"{scope}_ci_preservation_results.csv", index=False)
    ci.aggregate_results(data).to_csv(FINAL_DIR / f"{scope}_ci_preservation_aggregate.csv", index=False)
    ci.compute_wilcoxon_table(data, alpha=ci.ALPHA).to_csv(
        FINAL_DIR / f"{scope}_ci_preservation_wilcoxon.csv", index=False
    )
    return validate_scope(scope, cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-primary", type=Path, required=True)
    parser.add_argument("--comparison-completion", type=Path, required=True)
    parser.add_argument("--comparison-patches", type=Path, required=True)
    parser.add_argument("--interventional-primary", type=Path, required=True)
    parser.add_argument("--interventional-completion", type=Path, required=True)
    parser.add_argument("--interventional-n1000", type=Path, required=True)
    parser.add_argument("--simglucose-exclude-cr-cf", type=Path, required=True)
    args = parser.parse_args()

    comparison, comparison_sources = select_cells(
        [args.comparison_completion, args.comparison_primary],
        {20, 50, 100, 200, 500},
    )
    interventional, interventional_sources = select_cells(
        [
            args.interventional_n1000,
            args.simglucose_exclude_cr_cf,
            args.interventional_completion,
            args.interventional_primary,
        ],
        {20, 50, 100, 200, 500, 1000},
    )
    patched_rows = apply_exact_patches(comparison, args.comparison_patches)
    if patched_rows != 11:
        raise RuntimeError(f"Expected exactly 11 comparison patches, found {patched_rows}")

    ci = load_ci_module()
    validation = pd.concat(
        [write_scope("comparison", comparison, ci), write_scope("interventional", interventional, ci)],
        ignore_index=True,
    )
    validation.to_csv(FINAL_DIR / "validation.csv", index=False)
    pd.DataFrame(comparison_sources + interventional_sources).to_csv(
        FINAL_DIR / "provenance_sources.csv", index=False
    )
    summary = {
        "comparison_cells": len(comparison),
        "interventional_cells": len(interventional),
        "patched_comparison_rows": patched_rows,
        "valid_cells": int(validation["valid"].sum()),
        "total_cells": len(validation),
        "all_cells_valid": bool(validation["valid"].all()),
        "cells_with_documented_undefined_reference": int(
            validation["documented_undefined_reference_rows"].gt(0).sum()
        ),
        "documented_undefined_reference_rows": int(
            validation["documented_undefined_reference_rows"].sum()
        ),
    }
    (FINAL_DIR / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["all_cells_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
