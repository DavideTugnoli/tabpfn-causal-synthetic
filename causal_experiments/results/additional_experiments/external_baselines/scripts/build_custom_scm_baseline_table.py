#!/usr/bin/env python3
"""Build the Custom SCM external-baseline appendix table.

For the five external generators (TabularARGN, CTGAN, DATGAN, DECAF,
CausalDiffTab) and the two TabPFN reference conditions (vanilla original,
DAG-aware topological), this script computes per-cell medians of CMD, kMTVD,
and |NNAA - 0.5| on Custom SCM, plus paired Wilcoxon-Pratt tests with Holm
correction against both TabPFN references.

Pairing: every baseline bundle uses exactly the canonical cleaned TabPFN seed
set (100 seeds per training size); the script asserts this.

Holm families follow the forest-plot convention: one family per
(baseline, comparator, metric), i.e. the five training-size cells that would
share a panel.

Outputs (tables/):
- custom_scm_baseline_medians.csv
- custom_scm_baseline_wilcoxon_holm.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
REPO = BUNDLE.parents[3]
EXT = REPO / "external_baselines"
CANON = (
    REPO
    / "causal_experiments/results/comparison_experiment/data/"
    "result_custom_scm_comparison_experiment_cleaned_reps_100.csv"
)

TRAIN_SIZES = (20, 50, 100, 200, 500)
METRICS = {
    "correlation_matrix_difference": "CMD",
    "k_marginal_tvd": "kMTVD",
    "nnaa_distance_0p5": "NNAA_distance_0p5",
}
BASELINE_BUNDLES = {
    "CTGAN": EXT / "ctgan/results/final_custom_scm_20260606/result_custom_scm_ctgan_baseline_cleaned_reps_100.csv",
    "DATGAN": EXT / "datgan/results/final_custom_scm_20260606/result_custom_scm_datgan_baseline_cleaned_reps_100.csv",
    "DECAF": EXT / "decaf/results/final_custom_scm_20260606/result_custom_scm_decaf_baseline_cleaned_reps_100.csv",
    "CausalDiffTab": EXT / "causaldifftab/results/final_custom_scm_20260604/result_custom_scm_causaldifftab_baseline_cleaned_reps_100.csv",
}
TABULARARGN_PAIRED_LONG = (
    EXT / "tabularargn/results/final_five_datasets_20260428/analysis/tabularargn_vs_tabpfn_paired_long.csv"
)


def with_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["nnaa_distance_0p5"] = (pd.to_numeric(df["nnaa"], errors="raise") - 0.5).abs()
    return df


def holm(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def main() -> None:
    tables = BUNDLE / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    canon = with_distance(pd.read_csv(CANON))
    vanilla = canon[(canon["algorithm"] == "vanilla") & (canon["column_order"] == "original")]
    dag = canon[(canon["algorithm"] == "dag") & (canon["column_order"] == "topological")]
    canon_seeds = {int(n): set(g["seed"]) for n, g in vanilla.groupby("train_size")}

    baselines: dict[str, pd.DataFrame] = {}
    pl = pd.read_csv(TABULARARGN_PAIRED_LONG)
    ta = (
        pl[pl["dataset"] == "custom_scm"]
        .drop_duplicates(subset=["train_size", "seed"])
        .rename(
            columns={
                "tabularargn_correlation_matrix_difference": "correlation_matrix_difference",
                "tabularargn_k_marginal_tvd": "k_marginal_tvd",
                "tabularargn_nnaa": "nnaa",
            }
        )[["train_size", "seed", "correlation_matrix_difference", "k_marginal_tvd", "nnaa"]]
    )
    baselines["TabularARGN"] = with_distance(ta)
    for name, path in BASELINE_BUNDLES.items():
        baselines[name] = with_distance(pd.read_csv(path))

    for name, df in baselines.items():
        for n in TRAIN_SIZES:
            group = df[df["train_size"] == n]
            if len(group) != 100 or set(group["seed"]) != canon_seeds[n]:
                raise RuntimeError(f"{name} N={n}: seed set does not match the canonical TabPFN seeds")

    methods = list(baselines.items()) + [("TabPFN vanilla", vanilla), ("TabPFN DAG-aware", dag)]
    median_rows = []
    for name, df in methods:
        for n in TRAIN_SIZES:
            group = df[df["train_size"] == n]
            median_rows.append(
                {"method": name, "train_size": n, **{label: group[col].median() for col, label in METRICS.items()}}
            )
    medians = pd.DataFrame(median_rows)
    medians.to_csv(tables / "custom_scm_baseline_medians.csv", index=False)

    test_rows = []
    for bname, bdf in baselines.items():
        for cname, cdf in (("TabPFN vanilla", vanilla), ("TabPFN DAG-aware", dag)):
            for col, label in METRICS.items():
                p_raw, effects = [], []
                for n in TRAIN_SIZES:
                    a = bdf[bdf["train_size"] == n].set_index("seed")[col]
                    c = cdf[cdf["train_size"] == n].set_index("seed")[col]
                    diff = (a - c.reindex(a.index)).dropna()
                    assert len(diff) == 100
                    p_raw.append(wilcoxon(diff, zero_method="pratt").pvalue if (diff != 0).any() else 1.0)
                    effects.append(float(np.median(diff)))
                p_holm = holm(np.array(p_raw))
                for n, p, ph, eff in zip(TRAIN_SIZES, p_raw, p_holm, effects):
                    test_rows.append(
                        {
                            "baseline": bname,
                            "comparator": cname,
                            "metric": label,
                            "train_size": n,
                            "median_paired_diff": eff,
                            "p_raw": p,
                            "p_holm": ph,
                            "significant": bool(ph < 0.05),
                            "baseline_worse": bool(eff > 0),
                        }
                    )
    tests = pd.DataFrame(test_rows)
    tests.to_csv(tables / "custom_scm_baseline_wilcoxon_holm.csv", index=False)

    n_sig = int(tests["significant"].sum())
    print(f"[OK] medians: {len(medians)} rows; tests: {len(tests)} cells, {n_sig} Holm-significant")
    not_sig = tests[~tests["significant"]]
    if len(not_sig):
        print(not_sig[["baseline", "comparator", "metric", "train_size", "p_holm"]].to_string(index=False))


if __name__ == "__main__":
    main()
