#!/usr/bin/env python3
"""Regenerate TabularARGN paired statistics and figures from a validated paired CSV."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_analysis_module(path: Path):
    spec = importlib.util.spec_from_file_location("tabularargn_analysis", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--analysis-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paired = pd.read_csv(args.paired_csv)
    required = [
        "tabularargn_cmd_loss",
        "tabularargn_kmtvd_loss",
        "tabularargn_nnaa_loss",
        "tabpfn_cmd_loss",
        "tabpfn_kmtvd_loss",
        "tabpfn_nnaa_loss",
    ]
    if not np.isfinite(paired[required].to_numpy(dtype=float)).all():
        raise ValueError("Paired CSV contains non-finite metric losses")

    analysis = load_analysis_module(args.analysis_script)
    analysis_dir = args.output_dir / "analysis"
    figures_dir = args.output_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pairwise = analysis.summarize_pairwise(paired)
    wins = analysis.summarize_wins(pairwise)
    dag_dataset_summary = analysis.summarize_dataset_ratios(pairwise, "TabPFN DAG-aware")
    paired.to_csv(analysis_dir / "tabularargn_vs_tabpfn_paired_long.csv", index=False)
    pairwise.to_csv(analysis_dir / "pairwise_tabularargn_vs_tabpfn_by_cell.csv", index=False)
    wins.to_csv(analysis_dir / "tabularargn_vs_tabpfn_win_counts.csv", index=False)
    dag_dataset_summary.to_csv(
        analysis_dir / "tabularargn_vs_tabpfn_dag_aware_dataset_summary.csv", index=False
    )
    analysis.write_markdown_summary(
        wins, pairwise, dag_dataset_summary, analysis_dir / "tabularargn_vs_tabpfn_summary.md"
    )

    for method in ["TabPFN vanilla original", "TabPFN DAG-aware"]:
        slug = method.lower().replace(" ", "_").replace("-", "_")
        for metric in ["CMD", "kMTVD", "NNAA |x-0.5|"]:
            metric_slug = metric.lower().replace(" ", "_").replace("|", "").replace(".", "p5")
            analysis.make_heatmap(
                pairwise,
                method,
                metric,
                figures_dir / f"loss_ratio_tabularargn_vs_{slug}_{metric_slug}.pdf",
            )

    cmc_cmd = pairwise[
        (pairwise["dataset"] == "csuite_mixed_confounding")
        & (pairwise["train_size"] == 20)
        & (pairwise["metric"] == "CMD")
    ]
    summary = {
        "paired_rows": len(paired),
        "all_losses_finite": True,
        "cmc_n20_cmd": cmc_cmd.to_dict(orient="records"),
    }
    (args.output_dir / "regeneration_validation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
