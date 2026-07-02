#!/usr/bin/env python3
"""Build the paper-aligned REX bundle with metric-consistent vanilla sources.

Motivation. The earlier `final_100_paired` bundle filled the vanilla-side
`correlation_matrix_difference` from the historical RAW baseline CSVs, which
only carry the legacy `frobenius_corr_norm` definition. Comparing legacy
vanilla CMD against new-definition REX CMD invalidates the CMD forests there.
This builder uses CLEANED files (new CMD definition) for the vanilla side:

- CLB, CNS, CWA: the official paper cleaned CSVs in `../data/`, i.e. exactly
  the inputs behind the published REX figures. The pairing is therefore the
  published one (95-100 pairs per cell).
- Symprod: the historical cleaned (same campaign as the REX runs, new CMD
  definition) plus the eight recovered REX rows, giving 100 pairs per cell.

Each plot input contains the full cleaned condition set plus REX, mirroring
the original merged dag-discovered inputs so the Holm family matches the
published statistics.

Outputs (all under `final_paper_aligned_20260612/`, nothing else is touched):

- `plot_inputs/raw` and `plot_inputs/nnaa_distance_0p5`;
- `forest_plots/{cmd_kmtvd,nnaa_raw,nnaa_distance_0p5}` and matching
  `comparison_results/`;
- `reports/coverage.csv` and `reports/published_vs_fixed_comparison.csv`.
"""

from __future__ import annotations

import subprocess
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent  # additional_experiments/rex
COMPARISON_ROOT = BUNDLE.parents[1] / "comparison_experiment"
REPO_ROOT = BUNDLE.parents[3]
OUTPUT = BUNDLE

OFFICIAL_DATA = COMPARISON_ROOT / "data"
HISTORICAL_CLEANED = BUNDLE / "sources/leonardo_historical_cleaned"
RAW_REX = BUNDLE / "sources/leonardo_raw_rex"
PATCHED_SYMPROD = (
    BUNDLE / "sources/rex_recovered_symprod/result_csuite_symprod_simpson_rex_patched_100.csv"
)

DATASETS = ("csuite_large_backdoor", "csuite_nonlin_simpson", "csuite_symprod_simpson", "csuite_weak_arrows")
METRICS = ("correlation_matrix_difference", "k_marginal_tvd", "nnaa")
PUBLISHED_FOREST_DIR = (
    "causal_experiments/results/comparison_experiment/forest_plots/paper/comparison_experiment/"
    "dag_discovered_rex/vanilla_original_vs_dag_discovered_rex_topological/csv"
)
PUBLISHED_FILES = {
    "correlation_matrix_difference": "forest_dag_discovered_topological_vs_vanilla_original_correlation_matrix_difference.csv",
    "k_marginal_tvd": "forest_dag_discovered_topological_vs_vanilla_original_2marginal.csv",
    "nnaa": "forest_dag_discovered_topological_vs_vanilla_original_nnaa.csv",
}

sys.path.insert(0, str(REPO_ROOT))
from causal_experiments.results.comparison_experiment import forest_plots  # noqa: E402


def load_rex_rows(dataset: str) -> pd.DataFrame:
    short = dataset.replace("csuite_", "")
    raw = pd.read_csv(RAW_REX / f"csuite_csuite_{short}_dag_discovered_results.csv")
    valid = raw[(raw[list(METRICS)] >= 0).all(axis=1)].copy()

    if dataset == "csuite_symprod_simpson":
        patched = pd.read_csv(PATCHED_SYMPROD)
        rex_patched = patched[
            patched["algorithm"].astype(str).eq("dag_discovered")
            & patched["column_order"].astype(str).eq("topological")
        ].copy()
        if (rex_patched[list(METRICS)] < 0).any().any():
            raise RuntimeError("Patched symprod REX rows contain invalid metrics")
        # Non-recovered patched rows must coincide with the raw export.
        merged = rex_patched.merge(
            valid[["train_size", "seed", *METRICS]],
            on=["train_size", "seed"],
            suffixes=("", "_rawsrc"),
            how="inner",
        )
        for metric in METRICS:
            if not np.allclose(merged[metric], merged[f"{metric}_rawsrc"]):
                raise RuntimeError(f"Patched symprod REX rows diverge from raw on {metric}")
        valid = rex_patched

    recovered_path = (
        BUNDLE / "sources/rex_recovered_remaining" / f"recovered_rex_rows_{dataset}.csv"
    )
    if recovered_path.exists():
        recovered = pd.read_csv(recovered_path)
        if (recovered[list(METRICS)] < 0).any().any():
            raise RuntimeError(f"{dataset}: recovered REX rows contain invalid metrics")
        overlap = recovered.merge(valid[["train_size", "seed"]], on=["train_size", "seed"])
        if not overlap.empty:
            raise RuntimeError(f"{dataset}: recovered REX rows duplicate raw seeds")
        common = [c for c in valid.columns if c in recovered.columns]
        valid = pd.concat([valid[common], recovered[common]], ignore_index=True)

    if not set(valid["algorithm"].astype(str)) == {"dag_discovered"}:
        raise RuntimeError(f"{dataset}: unexpected REX algorithm labels")
    if not set(valid["column_order"].astype(str)) == {"topological"}:
        raise RuntimeError(f"{dataset}: unexpected REX column_order labels")
    return valid


def load_cleaned(dataset: str) -> pd.DataFrame:
    if dataset == "csuite_symprod_simpson":
        path = HISTORICAL_CLEANED / f"result_{dataset}_comparison_experiment_cleaned_reps_100.csv"
    else:
        path = OFFICIAL_DATA / f"result_{dataset}_comparison_experiment_cleaned_reps_100.csv"
    df = pd.read_csv(path)
    if df["correlation_matrix_difference"].lt(0).any():
        raise RuntimeError(f"{dataset}: cleaned source contains invalid CMD values")
    return df


def build_inputs() -> tuple[list[str], pd.DataFrame]:
    raw_dir = OUTPUT / "data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows = []
    outputs = []
    for dataset in DATASETS:
        cleaned = load_cleaned(dataset)
        rex = load_rex_rows(dataset)

        common_cols = [c for c in cleaned.columns if c in rex.columns]
        combined = pd.concat([cleaned[common_cols], rex[common_cols]], ignore_index=True)
        combined["nnaa_raw"] = pd.to_numeric(combined["nnaa"], errors="raise")
        combined["nnaa_distance_0p5"] = (combined["nnaa_raw"] - 0.5).abs()

        out_path = raw_dir / f"result_{dataset}_comparison_experiment_cleaned_reps_100.csv"
        combined.to_csv(out_path, index=False)
        outputs.append(str(out_path))

        vanilla = cleaned[
            cleaned["algorithm"].astype(str).eq("vanilla")
            & cleaned["column_order"].astype(str).eq("original")
        ]
        for train_size in sorted(vanilla["train_size"].unique()):
            van_seeds = set(vanilla.loc[vanilla["train_size"].eq(train_size), "seed"])
            rex_seeds = set(rex.loc[rex["train_size"].eq(train_size), "seed"])
            coverage_rows.append(
                {
                    "dataset": dataset,
                    "train_size": int(train_size),
                    "vanilla_seeds": len(van_seeds),
                    "rex_valid_seeds": len(rex_seeds),
                    "paired_seeds": len(van_seeds & rex_seeds),
                    "vanilla_source": "historical_cleaned" if dataset == "csuite_symprod_simpson" else "official_cleaned",
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    return outputs, coverage


def configure(inputs: list[str], paper_root: Path, stats_root: Path) -> None:
    forest_plots.PAPER_ROOT = paper_root
    forest_plots.COMPARISON_RESULTS_DIR = stats_root
    forest_plots.COMPARISON_RESULT_FILES = tuple(inputs)
    forest_plots._set_expected_dataset_slugs(tuple(inputs))


def run_forest(inputs: list[str], label: str, metrics: tuple[str, ...]) -> None:
    configure(inputs, OUTPUT / f"forest_plots/{label}", OUTPUT / f"comparison_results/{label}")
    forest_plots.main(
        metric_names=metrics,
        comparison_keys=("cross_dag_discovered_topological_vs_vanilla_original",),
        show_caption=False,
        recompute_stats=True,
        no_csv=False,
        single_column=False,
    )


def compare_with_published() -> pd.DataFrame:
    rows = []
    for metric, fname in PUBLISHED_FILES.items():
        blob = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"HEAD:{PUBLISHED_FOREST_DIR}/{fname}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        published = pd.read_csv(StringIO(blob))
        label = "nnaa_raw" if metric == "nnaa" else "cmd_kmtvd"
        new_path = (
            OUTPUT
            / f"forest_plots/{label}/vanilla_original_vs_dag_discovered_rex_topological/csv/{fname}"
        )
        new = pd.read_csv(new_path)
        merged = published.merge(new, on=["dataset", "train_size"], suffixes=("_paper", "_fixed"))
        for _, rec in merged.iterrows():
            rows.append(
                {
                    "metric": metric,
                    "dataset": rec["dataset"],
                    "train_size": int(rec["train_size"]),
                    "effect_paper": float(rec["effect_paper"]),
                    "effect_fixed": float(rec["effect_fixed"]),
                    "abs_delta": abs(float(rec["effect_paper"]) - float(rec["effect_fixed"])),
                    "sig_paper": bool(rec["is_significant_paper"]),
                    "sig_fixed": bool(rec["is_significant_fixed"]),
                    "sig_changed": bool(rec["is_significant_paper"]) != bool(rec["is_significant_fixed"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    reports = OUTPUT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    raw_inputs, coverage = build_inputs()
    coverage.to_csv(reports / "coverage.csv", index=False)
    print(coverage.to_string(index=False))

    symprod = coverage[coverage["dataset"] == "csuite_symprod_simpson"]
    if not symprod["paired_seeds"].eq(100).all():
        raise RuntimeError("Symprod recovery incomplete: expected 100 paired seeds per cell")

    run_forest(raw_inputs, "cmd_kmtvd", ("correlation_matrix_difference", "k_marginal_tvd"))
    run_forest(raw_inputs, "nnaa_raw", ("nnaa",))

    distance_dir = OUTPUT / "data/nnaa_distance_inputs"
    distance_dir.mkdir(parents=True, exist_ok=True)
    distance_inputs = []
    for path in raw_inputs:
        df = pd.read_csv(path)
        df["nnaa"] = df["nnaa_distance_0p5"]
        out = distance_dir / Path(path).name
        df.to_csv(out, index=False)
        distance_inputs.append(str(out))
    # usetex is active: bare "|" in text mode renders as an em dash, so wrap
    # the bars and the minus sign in math mode.
    forest_plots.METRIC_CONFIG["nnaa"]["title"] = (
        "$|$Nearest-Neighbor Adversarial Accuracy $-$ 0.5$|$"
    )
    run_forest(distance_inputs, "nnaa_distance_0p5", ("nnaa",))

    comparison = compare_with_published()
    comparison.to_csv(reports / "published_vs_fixed_comparison.csv", index=False)
    changed = comparison[comparison["sig_changed"] | (comparison["abs_delta"] > 1e-9)]
    print("\nCells differing from the published figures:")
    print(changed.to_string(index=False) if not changed.empty else "none")


if __name__ == "__main__":
    main()
