#!/usr/bin/env python3
"""Apply the validated CMC N=20 CMD patch and regenerate paired statistics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"
FIGURES_DIR = ROOT / "figures"
PAIRED_PATH = ANALYSIS_DIR / "tabularargn_vs_tabpfn_paired_long.csv"
PATCH_PATH = (
    Path(__file__).resolve().parents[5]
    / "remote_patch_downloads/tabularargn_cmd_patch_20260611/cmd_patch_36_seeds.csv"
)
ANALYSIS_SCRIPT = ROOT / "scripts/analyze_tabularargn_vs_tabpfn.py"


def load_analysis_module():
    spec = importlib.util.spec_from_file_location("tabularargn_analysis", ANALYSIS_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(ANALYSIS_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    paired = pd.read_csv(PAIRED_PATH)
    patch = pd.read_csv(PATCH_PATH)
    patch_seeds = set(patch["seed"].astype(int))
    target = (paired["dataset"] == "csuite_mixed_confounding") & (paired["train_size"] == 20)
    historical_bad = set(
        paired.loc[target & ~np.isfinite(paired["tabularargn_cmd_loss"]), "seed"].astype(int)
    )
    if historical_bad not in (patch_seeds, set()) or len(patch_seeds) != 36:
        raise ValueError(
            f"Patch seed mismatch: historical={sorted(historical_bad)}, patch={sorted(patch_seeds)}"
        )
    if len(patch) != 36 or not np.isfinite(patch["correlation_matrix_difference"]).all():
        raise ValueError("CMD patch must contain exactly 36 finite rows")

    cmd_by_seed = patch.set_index("seed")["correlation_matrix_difference"]
    patch_mask = target & paired["seed"].isin(patch_seeds)
    if int(patch_mask.sum()) != 36 * paired["tabpfn_method"].nunique():
        raise ValueError("Expected each patched seed once per TabPFN comparator")
    if not historical_bad:
        existing = paired.loc[patch_mask, ["seed", "tabularargn_cmd_loss"]].copy()
        expected = existing["seed"].map(cmd_by_seed)
        if not np.allclose(existing["tabularargn_cmd_loss"], expected, rtol=0.0, atol=1e-12):
            raise ValueError("Previously patched CMD values do not match the validated patch")
    paired.loc[patch_mask, "tabularargn_correlation_matrix_difference"] = paired.loc[
        patch_mask, "seed"
    ].map(cmd_by_seed)
    paired.loc[patch_mask, "tabularargn_cmd_loss"] = paired.loc[patch_mask, "seed"].map(cmd_by_seed)
    if not np.isfinite(paired["tabularargn_cmd_loss"]).all():
        raise ValueError("Paired analysis still contains non-finite TabularARGN CMD")

    paired.to_csv(PAIRED_PATH, index=False)

    analysis = load_analysis_module()
    pairwise = analysis.summarize_pairwise(paired)
    wins = analysis.summarize_wins(pairwise)
    dag_dataset_summary = analysis.summarize_dataset_ratios(pairwise, "TabPFN DAG-aware")
    pairwise.to_csv(ANALYSIS_DIR / "pairwise_tabularargn_vs_tabpfn_by_cell.csv", index=False)
    wins.to_csv(ANALYSIS_DIR / "tabularargn_vs_tabpfn_win_counts.csv", index=False)
    dag_dataset_summary.to_csv(
        ANALYSIS_DIR / "tabularargn_vs_tabpfn_dag_aware_dataset_summary.csv", index=False
    )
    analysis.write_markdown_summary(
        wins, pairwise, dag_dataset_summary, ANALYSIS_DIR / "tabularargn_vs_tabpfn_summary.md"
    )
    figures_regenerated = True
    try:
        for method in ["TabPFN vanilla original", "TabPFN DAG-aware"]:
            slug = method.lower().replace(" ", "_").replace("-", "_")
            for metric in ["CMD", "kMTVD", "NNAA |x-0.5|"]:
                metric_slug = metric.lower().replace(" ", "_").replace("|", "").replace(".", "p5")
                analysis.make_heatmap(
                    pairwise,
                    method,
                    metric,
                    FIGURES_DIR / f"loss_ratio_tabularargn_vs_{slug}_{metric_slug}.pdf",
                )
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        figures_regenerated = False

    cmc_cmd = pairwise[
        (pairwise["dataset"] == "csuite_mixed_confounding")
        & (pairwise["train_size"] == 20)
        & (pairwise["metric"] == "CMD")
    ]
    summary = {
        "patched_seeds": sorted(patch_seeds),
        "patched_seed_count": len(patch_seeds),
        "paired_rows_updated": int(patch_mask.sum()),
        "all_tabularargn_cmd_finite": bool(np.isfinite(paired["tabularargn_cmd_loss"]).all()),
        "figures_regenerated": figures_regenerated,
        "cmc_n20_cmd": cmc_cmd.to_dict(orient="records"),
    }
    (Path(__file__).resolve().parent / "cmd_patch_validation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
