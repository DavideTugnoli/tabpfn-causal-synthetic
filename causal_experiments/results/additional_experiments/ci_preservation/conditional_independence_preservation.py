#!/usr/bin/env python3
"""Measure conditional-independence preservation in saved synthetic datasets.

The primary analysis uses the empirical reference strategy: for each selected
triple (X, Y | Z), run the same CI test used by the PC discovery pipeline on a
real reference split and on the corresponding synthetic dataset. A real
independence is considered preserved when both Holm-corrected tests fail to
reject independence at alpha.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_experiments.utils.run_pc_discovery import _build_ci_callable  # noqa: E402


DEFAULT_WORK_ROOT = Path("/path/to/work_root")
DEFAULT_DATA_ROOT = DEFAULT_WORK_ROOT / "data"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR
ALPHA = 0.05
DEFAULT_RANDOM_SEED = 20260426
DEFAULT_MAX_CONDITIONING_SET_SIZE = 2
DEFAULT_MAX_TRIPLES = 200


@dataclass(frozen=True)
class DatasetSpec:
    """Location and metadata for one saved experiment family."""

    source: str
    dataset: str
    root: Path
    family: str

    @property
    def datasets_dir(self) -> Path:
        if (self.root / "synthetic").exists() and (self.root / "train").exists():
            return self.root
        return self.root / "datasets"

    @property
    def synthetic_dir(self) -> Path:
        if (self.root / "synthetic").exists():
            return self.root / "synthetic"
        return self.datasets_dir / "synthetic"


@dataclass(frozen=True)
class SyntheticRun:
    """One saved synthetic NPZ and its parsed identifiers."""

    spec: DatasetSpec
    path: Path
    raw_condition: str
    condition: str
    sample_size: int
    repetition: int


@dataclass(frozen=True)
class TripleSet:
    """A deterministic collection of CI triples for one dataset schema."""

    triples: tuple[tuple[int, int, tuple[int, ...]], ...]
    total_available: int
    sampled: bool
    seed: int


def stable_int_seed(*parts: object, modulo: int = 2**32 - 1) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def normalize_condition(raw_condition: str) -> str:
    condition = raw_condition
    condition = condition.replace("cpdag_v1_both_vanilla_discovered_first", "cpdag_discovered")
    condition = condition.replace("cpdag_v1_both_vanilla_minimal_first", "cpdag_minimal")
    condition = condition.replace("_worst", "_reverse_topological")
    return condition


def parse_synthetic_filename(path: Path) -> tuple[str, int, int] | None:
    match = re.match(r"^synthetic_(?P<condition>.+)_ts(?P<size>\d+)_s(?P<rep>\d+)\.npz$", path.name)
    if not match:
        return None
    return match.group("condition"), int(match.group("size")), int(match.group("rep"))


def discover_dataset_specs(data_root: Path, include_sources: set[str]) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []

    original = data_root / "original_tabpfn_extensions"
    rerun = data_root / "rerun_tabpfn_causal_synthetic"
    cleaned_archive = data_root / "cleaned_npz_archive/comparison/current_cleaned_20260429"
    if not cleaned_archive.exists():
        cleaned_archive = data_root / "cleaned_npz_archive/comparison_current_cleaned_20260429"
    cleaned_interventional_archive = data_root / "cleaned_npz_archive/interventional/current_cleaned_20260429"

    if "cleaned_archive" in include_sources and cleaned_archive.exists():
        archive_datasets = cleaned_archive / "datasets"
        if archive_datasets.exists():
            for dataset_root in sorted(archive_datasets.iterdir()):
                if not dataset_root.is_dir() or not (dataset_root / "synthetic").exists():
                    continue
                dataset = dataset_root.name
                if dataset.startswith("custom_scm"):
                    family = "custom_scm"
                elif dataset.startswith("csuite_"):
                    family = "csuite"
                elif dataset.startswith("simglucose"):
                    family = "simglucose_complete"
                else:
                    family = "archive"
                specs.append(DatasetSpec("cleaned_archive", dataset, dataset_root, family))

    if "cleaned_interventional_archive" in include_sources and cleaned_interventional_archive.exists():
        archive_datasets = cleaned_interventional_archive / "datasets"
        if archive_datasets.exists():
            for dataset_root in sorted(archive_datasets.iterdir()):
                if not dataset_root.is_dir() or not (dataset_root / "synthetic").exists():
                    continue
                dataset = dataset_root.name
                if dataset.startswith("custom_scm"):
                    family = "custom_scm"
                elif dataset.startswith("csuite_"):
                    family = "csuite"
                elif dataset.startswith("simglucose"):
                    family = "simglucose_complete"
                else:
                    family = "archive"
                specs.append(DatasetSpec("cleaned_interventional_archive", dataset, dataset_root, family))

    if "original" in include_sources:
        custom_root = original / "causal_experiments/custom_scm_experiment/comparison_experiment/results"
        if custom_root.exists():
            specs.append(DatasetSpec("original", "custom_scm", custom_root, "custom_scm"))

        csuite_root = original / "causal_experiments/csuite_experiment/comparison_experiment_csuite/results"
        if csuite_root.exists():
            for dataset_root in sorted(csuite_root.iterdir()):
                if dataset_root.is_dir() and (dataset_root / "datasets").exists():
                    specs.append(DatasetSpec("original", dataset_root.name, dataset_root, "csuite"))

        sim_complete = original / (
            "causal_experiments/real_dataset_simglucose/"
            "acyclic_scm_simglucose_complete/comparison_experiment/results"
        )
        if sim_complete.exists():
            specs.append(DatasetSpec("original", "simglucose_complete", sim_complete, "simglucose_complete"))

        sim_old = original / (
            "causal_experiments/real_dataset_simglucose_old/"
            "acyclic_scm_simglucose/comparison_experiment/results"
        )
        if sim_old.exists():
            specs.append(DatasetSpec("original", "simglucose_old", sim_old, "simglucose_old"))

    if "rerun" in include_sources:
        csuite_root = rerun / "causal_experiments/csuite_experiment/comparison_experiment_csuite/results"
        if csuite_root.exists():
            for dataset_root in sorted(csuite_root.iterdir()):
                if dataset_root.is_dir() and (dataset_root / "datasets").exists():
                    specs.append(DatasetSpec("rerun", dataset_root.name, dataset_root, "csuite"))

    return specs


def discover_synthetic_runs(
    specs: Sequence[DatasetSpec],
    datasets: set[str] | None,
    conditions: set[str] | None,
    sample_sizes: set[int] | None,
    max_repetitions: int | None,
    exclude_discovered_cpdag: bool,
) -> list[SyntheticRun]:
    runs: list[SyntheticRun] = []
    for spec in specs:
        if datasets and spec.dataset not in datasets:
            continue
        if not spec.synthetic_dir.exists():
            continue
        for path in sorted(spec.synthetic_dir.glob("synthetic_*.npz")):
            parsed = parse_synthetic_filename(path)
            if parsed is None:
                continue
            raw_condition, sample_size, repetition = parsed
            condition = normalize_condition(raw_condition)
            if exclude_discovered_cpdag and condition.startswith("cpdag_discovered"):
                continue
            if conditions and condition not in conditions and raw_condition not in conditions:
                continue
            if sample_sizes and sample_size not in sample_sizes:
                continue
            runs.append(SyntheticRun(spec, path, raw_condition, condition, sample_size, repetition))
    if max_repetitions is None:
        return runs

    selected: list[SyntheticRun] = []
    grouped: dict[tuple[str, str, str, int], list[SyntheticRun]] = {}
    for run in runs:
        key = (run.spec.source, run.spec.dataset, run.condition, run.sample_size)
        grouped.setdefault(key, []).append(run)
    for group_runs in grouped.values():
        selected.extend(sorted(group_runs, key=lambda item: item.repetition)[:max_repetitions])
    return sorted(selected, key=lambda item: (item.spec.source, item.spec.dataset, item.condition, item.sample_size, item.repetition))


def apply_synthetic_overrides(
    runs: Sequence[SyntheticRun],
    overrides_root: Path | None,
) -> list[SyntheticRun]:
    """Use recovered synthetic files without modifying the canonical archive."""
    if overrides_root is None:
        return list(runs)

    overridden: list[SyntheticRun] = []
    for run in runs:
        candidates = (
            overrides_root / run.spec.dataset / "datasets" / "synthetic" / run.path.name,
            overrides_root / run.spec.dataset / "synthetic" / run.path.name,
        )
        override = next((path for path in candidates if path.exists()), None)
        overridden.append(
            SyntheticRun(
                run.spec,
                override or run.path,
                run.raw_condition,
                run.condition,
                run.sample_size,
                run.repetition,
            )
        )
    return overridden


def select_repetition_chunk(
    runs: Sequence[SyntheticRun],
    chunk_index: int | None,
    num_chunks: int | None,
) -> list[SyntheticRun]:
    """Partition selected runs by deterministic repetition rank.

    Chunking happens after ``max_repetitions`` selection, so the union of all
    chunks is exactly the same run set used by the corresponding monolithic
    execution. All conditions for a repetition stay in the same chunk, which
    preserves reference-test caching within each job.
    """
    if chunk_index is None and num_chunks is None:
        return list(runs)
    if chunk_index is None or num_chunks is None:
        raise ValueError("Both --repetition-chunk-index and --repetition-num-chunks are required")
    if num_chunks <= 0 or chunk_index < 0 or chunk_index >= num_chunks:
        raise ValueError(f"Invalid repetition chunk {chunk_index}/{num_chunks}")

    repetitions_by_cell: dict[tuple[str, str, int], list[int]] = {}
    for run in runs:
        key = (run.spec.source, run.spec.dataset, run.sample_size)
        repetitions_by_cell.setdefault(key, []).append(run.repetition)
    repetition_rank = {
        key: {repetition: rank for rank, repetition in enumerate(sorted(set(repetitions)))}
        for key, repetitions in repetitions_by_cell.items()
    }
    return [
        run
        for run in runs
        if repetition_rank[(run.spec.source, run.spec.dataset, run.sample_size)][run.repetition] % num_chunks
        == chunk_index
    ]


def npz_array(path: Path, preferred_keys: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=True) as archive:
        key = next((name for name in preferred_keys if name in archive), None)
        if key is None:
            raise KeyError(f"None of {preferred_keys} found in {path}; keys={list(archive.keys())}")
        data = np.asarray(archive[key], dtype=float)
        column_names = [str(x) for x in archive["column_names"].tolist()] if "column_names" in archive else [
            f"X{i}" for i in range(data.shape[1])
        ]
    return data, column_names


def load_reference_data(
    spec: DatasetSpec,
    sample_size: int,
    repetition: int,
    reference_source: str,
) -> tuple[np.ndarray, list[str], Path]:
    if reference_source == "global_test":
        path = spec.datasets_dir / "global_test_set.npz"
        data, column_names = npz_array(path, ("X_test", "test_data", "data"))
        return data, column_names, path
    if reference_source == "train":
        path = spec.datasets_dir / f"train_ts{sample_size}_s{repetition}.npz"
        if not path.exists():
            path = spec.datasets_dir / "train" / f"train_ts{sample_size}_s{repetition}.npz"
        data, column_names = npz_array(path, ("X_train", "train_data", "data"))
        return data, column_names, path
    raise ValueError(f"Unsupported reference source: {reference_source}")


def load_synthetic_data(run: SyntheticRun) -> tuple[np.ndarray, list[str]]:
    return npz_array(run.path, ("synthetic_data", "X_synthetic", "data"))


def parse_column_list(values: str | None) -> tuple[str, ...]:
    if values is None or values.strip() == "":
        return tuple()
    return tuple(item.strip() for item in values.split(",") if item.strip())


def exclude_columns_by_name(
    data: np.ndarray,
    column_names: Sequence[str],
    exclude_columns: Sequence[str],
) -> tuple[np.ndarray, list[str], tuple[str, ...]]:
    if not exclude_columns:
        return data, list(column_names), tuple()
    exclude_set = set(exclude_columns)
    keep_indices = [idx for idx, name in enumerate(column_names) if name not in exclude_set]
    excluded = tuple(name for name in column_names if name in exclude_set)
    if len(keep_indices) == len(column_names):
        return data, list(column_names), excluded
    return data[:, keep_indices], [column_names[idx] for idx in keep_indices], excluded


def categorical_indices_for(spec: DatasetSpec, column_names: Sequence[str], repo_root: Path) -> list[int]:
    if spec.family == "custom_scm":
        return []
    if spec.family.startswith("simglucose"):
        # The SimGlucose comparison pipeline declares patient_id/action_CHO_g as
        # categorical for generation, but CPDAG discovery is run with KCI.
        return [idx for idx, name in enumerate(column_names) if name in {"patient_id", "action_CHO_g"}]
    if spec.family == "csuite":
        variables_path = repo_root / "causal_experiments/csuite_experiment/csuite_datasets" / spec.dataset / "variables.json"
        if variables_path.exists():
            with variables_path.open("r", encoding="utf-8") as handle:
                variables = json.load(handle)["variables"]
            categorical_names = {
                var["name"]
                for var in variables
                if str(var.get("type", "")).lower() in {"categorical", "binary"}
            }
            return [idx for idx, name in enumerate(column_names) if name in categorical_names]
    return []


def independence_test_for(spec: DatasetSpec, categorical_cols: Sequence[int]) -> str:
    dataset_lower = spec.dataset.lower()
    if spec.family == "custom_scm":
        return "fisherz"
    if spec.family.startswith("simglucose"):
        return "kci"
    if spec.family == "csuite":
        if categorical_cols:
            if dataset_lower == "csuite_mixed_confounding":
                return "hybrid"
            return "gsq"
        if dataset_lower in {"csuite_lingauss", "csuite_linexp"}:
            return "fisherz"
        return "kci"
    return "fisherz"


def enumerate_triples(
    n_features: int,
    max_conditioning_set_size: int,
    max_triples: int | None,
    seed: int,
) -> TripleSet:
    triples: list[tuple[int, int, tuple[int, ...]]] = []
    for x, y in itertools.combinations(range(n_features), 2):
        remaining = [idx for idx in range(n_features) if idx not in {x, y}]
        max_k = min(max_conditioning_set_size, len(remaining))
        for k in range(max_k + 1):
            for cond in itertools.combinations(remaining, k):
                triples.append((x, y, tuple(cond)))

    total_available = len(triples)
    if max_triples is not None and max_triples > 0 and total_available > max_triples:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(total_available, size=max_triples, replace=False))
        triples = [triples[int(i)] for i in selected]
        sampled = True
    else:
        sampled = False
    return TripleSet(tuple(triples), total_available, sampled, seed)


def holm_adjusted_pvalues(p_values: Sequence[float]) -> np.ndarray:
    p_array = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_array, np.nan, dtype=float)
    valid = np.isfinite(p_array)
    valid_indices = np.where(valid)[0]
    if valid_indices.size == 0:
        return adjusted
    order = valid_indices[np.argsort(p_array[valid_indices], kind="mergesort")]
    m = len(order)
    adjusted_sorted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted_sorted[rank] = min(1.0, p_array[idx] * (m - rank))
    for rank in range(1, m):
        adjusted_sorted[rank] = max(adjusted_sorted[rank], adjusted_sorted[rank - 1])
    adjusted[order] = adjusted_sorted
    return adjusted


def hodges_lehmann_ci_from_diffs(
    diffs: np.ndarray,
    alpha: float = ALPHA,
) -> tuple[float, float, float]:
    """Compute paired Hodges-Lehmann pseudo-median and a Wilcoxon-inversion CI."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan"), float("nan"), float("nan")
    if np.all(np.abs(diffs) < 1e-12):
        return 0.0, 0.0, 0.0
    if diffs.size == 1:
        single = float(diffs[0])
        return single, single, single

    pairwise = np.add.outer(diffs, diffs) * 0.5
    tri = pairwise[np.triu_indices(diffs.size)]
    pseudo_vals = np.unique(np.sort(tri))
    estimate = float(np.median(tri))

    alpha_half = alpha / 2.0
    cache: dict[tuple[int, str], float] = {}

    def pvalue_at(idx: int, alternative: str) -> float:
        key = (idx, alternative)
        if key in cache:
            return cache[key]
        theta = float(pseudo_vals[idx])
        try:
            result = wilcoxon(
                diffs - theta,
                zero_method="wilcox",
                correction=False,
                alternative=alternative,
                method="auto",
            )
            p_value = float(result.pvalue)
        except ValueError:
            p_value = 1.0
        cache[key] = p_value
        return p_value

    n_values = pseudo_vals.size
    if pvalue_at(n_values - 1, "greater") <= alpha_half:
        lower_idx = n_values - 1
    else:
        lo, hi = 0, n_values - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if pvalue_at(mid, "greater") > alpha_half:
                hi = mid
            else:
                lo = mid + 1
        lower_idx = lo

    if pvalue_at(0, "less") <= alpha_half:
        upper_idx = 0
    else:
        lo, hi = 0, n_values - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if pvalue_at(mid, "less") > alpha_half:
                lo = mid
            else:
                hi = mid - 1
        upper_idx = lo

    return estimate, float(pseudo_vals[lower_idx]), float(pseudo_vals[upper_idx])


def prepare_data_for_ci(data: np.ndarray, indep_test: str) -> np.ndarray:
    prepared = np.asarray(data, dtype=float)
    if indep_test.lower() == "fisherz" and prepared.size:
        prepared = prepared.copy()
        means = np.mean(prepared, axis=0, keepdims=True)
        stds = np.std(prepared, axis=0, keepdims=True)
        stds[stds == 0.0] = 1.0
        prepared = (prepared - means) / stds
    return prepared


def run_ci_tests(
    data: np.ndarray,
    triples: Sequence[tuple[int, int, tuple[int, ...]]],
    categorical_cols: Sequence[int],
    indep_test: str,
    alpha: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if data.ndim != 2 or data.shape[0] < 3 or data.shape[1] < 2:
        return np.full(len(triples), np.nan), np.full(len(triples), False), len(triples)

    prepared = prepare_data_for_ci(data, indep_test)
    tester = _build_ci_callable(
        prepared,
        categorical_cols=list(categorical_cols),
        indep_test=indep_test,
        alpha=alpha,
        hybrid_params={"k": 5, "permutations": 500, "random_state": random_state}
        if indep_test == "hybrid"
        else None,
        random_state=random_state,
    )

    p_values = np.full(len(triples), np.nan, dtype=float)
    failures = 0
    for idx, (x, y, cond) in enumerate(triples):
        try:
            p_values[idx] = float(tester(int(x), int(y), list(cond)))
        except Exception:
            failures += 1

    adjusted = holm_adjusted_pvalues(p_values)
    independent = np.isfinite(adjusted) & (adjusted > alpha)
    return adjusted, independent, failures


def deterministic_synthetic_sample(
    synthetic: np.ndarray,
    target_size: int,
    seed: int,
) -> np.ndarray:
    if target_size <= 0 or synthetic.shape[0] <= target_size:
        return synthetic
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(synthetic.shape[0], size=target_size, replace=False))
    return synthetic[indices]


def result_record(
    run: SyntheticRun,
    n_triples_total: int,
    n_reference_independent: int,
    n_preserved: int,
    n_concordant: int,
    n_failed_reference: int,
    n_failed_synthetic: int,
    status: str,
    indep_test: str,
    triple_seed: int,
    reference_path: Path,
    excluded_columns: Sequence[str] = (),
) -> dict[str, Any]:
    denominator = n_reference_independent
    fraction = float(n_preserved / denominator) if denominator > 0 else float("nan")
    fraction_concordant = float(n_concordant / n_triples_total) if n_triples_total > 0 else float("nan")
    return {
        "source": run.spec.source,
        "dataset": run.spec.dataset,
        "condition": run.condition,
        "raw_condition": run.raw_condition,
        "sample_size": run.sample_size,
        "repetition": run.repetition,
        "n_preserved": int(n_preserved),
        "fraction_preserved": fraction,
        "n_triples_total": int(n_triples_total),
        "n_reference_independent": int(n_reference_independent),
        "n_concordant": int(n_concordant),
        "fraction_concordant": fraction_concordant,
        "n_failed_reference": int(n_failed_reference),
        "n_failed_synthetic": int(n_failed_synthetic),
        "independence_test": indep_test,
        "triple_seed": int(triple_seed),
        "reference_path": str(reference_path),
        "synthetic_path": str(run.path),
        "excluded_columns": ",".join(excluded_columns),
        "status": status,
    }


def compute_preservation(args: argparse.Namespace, runs: Sequence[SyntheticRun]) -> pd.DataFrame:
    requested_exclusions = parse_column_list(args.exclude_columns)
    reference_cache: dict[tuple[str, str, int, int, str, tuple[str, ...]], tuple[np.ndarray, int, int, Path]] = {}
    triple_cache: dict[tuple[str, str, tuple[str, ...]], TripleSet] = {}
    records: list[dict[str, Any]] = []

    for run_idx, run in enumerate(runs, start=1):
        if args.progress_every and (run_idx == 1 or run_idx % args.progress_every == 0):
            print(f"[{run_idx}/{len(runs)}] {run.spec.dataset} {run.condition} ts={run.sample_size} rep={run.repetition}")

        try:
            reference, column_names, reference_path = load_reference_data(
                run.spec,
                run.sample_size,
                run.repetition,
                args.reference_source,
            )
            reference, column_names, excluded_columns = exclude_columns_by_name(
                reference,
                column_names,
                requested_exclusions,
            )
            categorical_cols = categorical_indices_for(run.spec, column_names, REPO_ROOT)
            indep_test = independence_test_for(run.spec, categorical_cols)
            triple_seed = stable_int_seed(args.random_seed, run.spec.source, run.spec.dataset, "triples")
            triple_key = (run.spec.source, run.spec.dataset, tuple(column_names))
            if triple_key not in triple_cache:
                triple_cache[triple_key] = enumerate_triples(
                    reference.shape[1],
                    args.max_conditioning_set_size,
                    args.max_triples,
                    triple_seed,
                )
            triple_set = triple_cache[triple_key]

            reference_key = (
                run.spec.source,
                run.spec.dataset,
                run.sample_size if args.reference_source == "train" else -1,
                run.repetition if args.reference_source == "train" else -1,
                args.reference_source,
                tuple(column_names),
            )
            if reference_key not in reference_cache:
                ref_seed = stable_int_seed(args.random_seed, run.spec.dataset, run.sample_size, run.repetition, "reference")
                _, ref_independent, ref_failures = run_ci_tests(
                    reference,
                    triple_set.triples,
                    categorical_cols,
                    indep_test,
                    args.alpha,
                    ref_seed,
                )
                reference_cache[reference_key] = (ref_independent, ref_failures, int(ref_independent.sum()), reference_path)
            ref_independent, ref_failures, n_reference_independent, reference_path = reference_cache[reference_key]

            synthetic, synthetic_columns = load_synthetic_data(run)
            synthetic, synthetic_columns, synthetic_excluded_columns = exclude_columns_by_name(
                synthetic,
                synthetic_columns,
                requested_exclusions,
            )
            if synthetic.ndim != 2 or synthetic.shape[1] != reference.shape[1] or synthetic.shape[0] == 0:
                records.append(
                    result_record(
                        run,
                        len(triple_set.triples),
                        n_reference_independent,
                        0,
                        0,
                        ref_failures,
                        len(triple_set.triples),
                        "invalid_synthetic_shape",
                        indep_test,
                        triple_set.seed,
                        reference_path,
                        excluded_columns,
                    )
                )
                continue
            if list(synthetic_columns) != list(column_names):
                # Saved synthetic data should already be reordered to the
                # original schema. Keep a hard failure visible if that contract
                # is broken.
                status = "column_names_mismatch"
            else:
                status = "ok"
            if tuple(synthetic_excluded_columns) != tuple(excluded_columns):
                status = "excluded_columns_mismatch"

            if args.synthetic_sample_mode == "match_reference":
                sample_seed = stable_int_seed(args.random_seed, run.spec.dataset, run.condition, run.sample_size, run.repetition, "synthetic")
                synthetic = deterministic_synthetic_sample(synthetic, reference.shape[0], sample_seed)

            syn_seed = stable_int_seed(args.random_seed, run.spec.dataset, run.condition, run.sample_size, run.repetition, "synthetic_ci")
            _, syn_independent, syn_failures = run_ci_tests(
                synthetic,
                triple_set.triples,
                categorical_cols,
                indep_test,
                args.alpha,
                syn_seed,
            )
            finite_mask = np.isfinite(ref_independent.astype(float)) & np.isfinite(syn_independent.astype(float))
            real_independent = ref_independent & finite_mask
            preserved = real_independent & syn_independent
            concordant = (ref_independent == syn_independent) & finite_mask

            records.append(
                result_record(
                    run,
                    len(triple_set.triples),
                    int(real_independent.sum()),
                    int(preserved.sum()),
                    int(concordant.sum()),
                    ref_failures,
                    syn_failures,
                    status,
                    indep_test,
                    triple_set.seed,
                    reference_path,
                    excluded_columns,
                )
            )
        except Exception as exc:
            records.append(
                {
                    "source": run.spec.source,
                    "dataset": run.spec.dataset,
                    "condition": run.condition,
                    "raw_condition": run.raw_condition,
                    "sample_size": run.sample_size,
                    "repetition": run.repetition,
                    "n_preserved": 0,
                    "fraction_preserved": float("nan"),
                    "n_triples_total": 0,
                    "n_reference_independent": 0,
                    "n_concordant": 0,
                    "fraction_concordant": float("nan"),
                    "n_failed_reference": 0,
                    "n_failed_synthetic": 0,
                    "independence_test": "unknown",
                    "triple_seed": args.random_seed,
                    "reference_path": "",
                    "synthetic_path": str(run.path),
                    "excluded_columns": ",".join(requested_exclusions),
                    "status": f"error:{type(exc).__name__}:{str(exc)[:200]}",
                }
            )

    return pd.DataFrame.from_records(records)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["status"].eq("ok") | df["status"].eq("column_names_mismatch")].copy()
    if valid.empty:
        return pd.DataFrame()
    grouped = valid.groupby(["source", "dataset", "condition", "sample_size"], dropna=False)
    return grouped.agg(
        n_runs=("fraction_preserved", "count"),
        median_fraction_preserved=("fraction_preserved", "median"),
        q1_fraction_preserved=("fraction_preserved", lambda x: float(np.nanpercentile(x, 25))),
        q3_fraction_preserved=("fraction_preserved", lambda x: float(np.nanpercentile(x, 75))),
        median_fraction_concordant=("fraction_concordant", "median"),
        median_n_reference_independent=("n_reference_independent", "median"),
        median_n_triples_total=("n_triples_total", "median"),
    ).reset_index()


def compute_wilcoxon_table(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    valid = df[df["fraction_preserved"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    baselines = ["vanilla_original", "vanilla_reverse_topological"]
    records: list[dict[str, Any]] = []
    for (source, dataset, sample_size), group in valid.groupby(["source", "dataset", "sample_size"]):
        conditions = sorted(group["condition"].unique())
        for baseline in baselines:
            if baseline not in conditions:
                continue
            baseline_df = group[group["condition"] == baseline][["repetition", "fraction_preserved"]].rename(
                columns={"fraction_preserved": "baseline_fraction"}
            )
            for condition in conditions:
                if condition == baseline or condition.startswith("vanilla_"):
                    continue
                cond_df = group[group["condition"] == condition][["repetition", "fraction_preserved"]].rename(
                    columns={"fraction_preserved": "condition_fraction"}
                )
                paired = pd.merge(baseline_df, cond_df, on="repetition", how="inner").dropna()
                if paired.empty:
                    continue
                diffs = paired["condition_fraction"].to_numpy(float) - paired["baseline_fraction"].to_numpy(float)
                try:
                    res = wilcoxon(diffs, zero_method="pratt", correction=False, alternative="two-sided", method="auto")
                    statistic = float(res.statistic)
                    p_value = float(res.pvalue)
                except ValueError:
                    statistic = float("nan")
                    p_value = 1.0
                hl, ci_lower, ci_upper = hodges_lehmann_ci_from_diffs(diffs, alpha=alpha)
                records.append(
                    {
                        "source": source,
                        "dataset": dataset,
                        "sample_size": sample_size,
                        "baseline": baseline,
                        "condition": condition,
                        "n_pairs": len(paired),
                        "median_baseline": float(paired["baseline_fraction"].median()),
                        "median_condition": float(paired["condition_fraction"].median()),
                        "effect_hl": hl,
                        "effect_ci_lower": ci_lower,
                        "effect_ci_upper": ci_upper,
                        "statistic": statistic,
                        "p_value": p_value,
                    }
                )

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out
    out["p_value_holm"] = np.nan
    out["holm_significant"] = False
    for _, idx in out.groupby(["source", "dataset", "sample_size", "baseline"]).groups.items():
        adjusted = holm_adjusted_pvalues(out.loc[idx, "p_value"].to_numpy(float))
        out.loc[idx, "p_value_holm"] = adjusted
        out.loc[idx, "holm_significant"] = adjusted <= alpha
    return out


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text("_No rows._\n", encoding="utf-8")
        return
    try:
        text = df.to_markdown(index=False)
    except Exception:
        text = df.to_csv(index=False)
    path.write_text(text + "\n", encoding="utf-8")


def plot_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    valid = df[df["fraction_preserved"].notna()].copy()
    if valid.empty:
        return
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    for (source, dataset), group in valid.groupby(["source", "dataset"]):
        sample_sizes = sorted(group["sample_size"].unique())
        conditions = sorted(group["condition"].unique())
        if not sample_sizes or not conditions:
            continue
        fig, axes = plt.subplots(
            1,
            len(sample_sizes),
            figsize=(max(4.0 * len(sample_sizes), 6.0), 4.2),
            sharey=True,
            squeeze=False,
        )
        for ax, sample_size in zip(axes[0], sample_sizes):
            sub = group[group["sample_size"] == sample_size]
            series = [
                sub[sub["condition"] == condition]["fraction_preserved"].dropna().to_numpy(float)
                for condition in conditions
            ]
            ax.boxplot(
                series,
                labels=conditions,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": "#d8dee9", "edgecolor": "#3b4252", "linewidth": 0.9},
                medianprops={"color": "#2e3440", "linewidth": 1.2},
                whiskerprops={"color": "#4c566a", "linewidth": 0.8},
                capprops={"color": "#4c566a", "linewidth": 0.8},
            )
            ax.set_title(f"N={sample_size}", fontsize=10)
            ax.tick_params(axis="x", labelrotation=65, labelsize=7)
            ax.set_ylim(-0.02, 1.02)
        axes[0][0].set_ylabel("Fraction of real independencies preserved")
        fig.suptitle(f"{dataset} ({source})", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        safe_name = re.sub(r"[^0-9A-Za-z._-]+", "_", f"{source}_{dataset}")
        fig.savefig(figures_dir / f"ci_preservation_boxplot_{safe_name}.pdf", bbox_inches="tight")
        fig.savefig(figures_dir / f"ci_preservation_boxplot_{safe_name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def estimate_cost(
    specs: Sequence[DatasetSpec],
    runs: Sequence[SyntheticRun],
    args: argparse.Namespace,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    runs_by_dataset: dict[tuple[str, str], list[SyntheticRun]] = {}
    for run in runs:
        runs_by_dataset.setdefault((run.spec.source, run.spec.dataset), []).append(run)

    for spec in specs:
        key = (spec.source, spec.dataset)
        dataset_runs = runs_by_dataset.get(key, [])
        if not dataset_runs:
            continue
        sample_run = dataset_runs[0]
        try:
            reference, column_names, _ = load_reference_data(
                sample_run.spec,
                sample_run.sample_size,
                sample_run.repetition,
                args.reference_source,
            )
            reference, column_names, excluded_columns = exclude_columns_by_name(
                reference,
                column_names,
                parse_column_list(args.exclude_columns),
            )
            categorical_cols = categorical_indices_for(spec, column_names, REPO_ROOT)
            indep_test = independence_test_for(spec, categorical_cols)
            triple_seed = stable_int_seed(args.random_seed, spec.source, spec.dataset, "triples")
            triple_set = enumerate_triples(
                reference.shape[1],
                args.max_conditioning_set_size,
                args.max_triples,
                triple_seed,
            )
        except Exception as exc:
            records.append(
                {
                    "source": spec.source,
                    "dataset": spec.dataset,
                    "n_variables": math.nan,
                    "independence_test": "unknown",
                    "n_selected_triples": 0,
                    "n_available_triples": 0,
                    "n_synthetic_runs": len(dataset_runs),
                    "estimated_ci_tests": 0,
                    "excluded_columns": args.exclude_columns,
                    "note": f"metadata_error:{type(exc).__name__}:{str(exc)[:120]}",
                }
            )
            continue

        reference_units = {
            (run.sample_size if args.reference_source == "train" else -1, run.repetition if args.reference_source == "train" else -1)
            for run in dataset_runs
        }
        estimated_tests = len(triple_set.triples) * (len(dataset_runs) + len(reference_units))
        records.append(
            {
                "source": spec.source,
                "dataset": spec.dataset,
                "n_variables": reference.shape[1],
                "excluded_columns": ",".join(excluded_columns),
                "independence_test": indep_test,
                "n_selected_triples": len(triple_set.triples),
                "n_available_triples": triple_set.total_available,
                "triples_sampled": triple_set.sampled,
                "triple_seed": triple_set.seed,
                "n_synthetic_runs": len(dataset_runs),
                "n_reference_units": len(reference_units),
                "estimated_ci_tests": estimated_tests,
                "note": "",
            }
        )
    return pd.DataFrame.from_records(records)


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    cost_df: pd.DataFrame,
    results_df: pd.DataFrame | None,
) -> None:
    total_tests = int(cost_df["estimated_ci_tests"].sum()) if not cost_df.empty else 0
    heavy_tests = sorted(set(cost_df[cost_df["independence_test"].isin(["kci", "hybrid"])]["dataset"])) if not cost_df.empty else []
    lines = [
        "# Conditional-Independence Preservation",
        "",
        "## Protocol",
        "",
        "- Primary strategy: empirical reference labels from real NPZ data.",
        f"- Reference source: `{args.reference_source}`.",
        f"- Synthetic sample mode: `{args.synthetic_sample_mode}`.",
        f"- Alpha: `{args.alpha}` with Holm-Bonferroni correction across triples within each run.",
        f"- Max conditioning-set size: `{args.max_conditioning_set_size}`.",
        f"- Max triples per dataset schema: `{args.max_triples}` (`0` means exhaustive).",
        f"- Triple/random seed: `{args.random_seed}`.",
        f"- Excluded columns: `{args.exclude_columns or 'none'}`.",
        "- CI tests reuse the paper discovery pipeline choices: Fisher-Z for the Custom SCM, "
        "KCI for continuous nonlinear CSuite and SimGlucose, G^2 for categorical CSuite, "
        "and the existing hybrid KCI/G^2/kNN-CMI tester for `csuite_mixed_confounding`.",
        "",
        "## Cost Estimate",
        "",
        f"- Estimated CI test calls: `{total_tests}`.",
        f"- Datasets using expensive KCI/hybrid tests: `{', '.join(heavy_tests) if heavy_tests else 'none'}`.",
        "",
    ]
    if total_tests > 500_000 or heavy_tests:
        lines.extend(
            [
                "Given the use of KCI/hybrid tests, the full run should not be executed on a login node.",
                "Use `--estimate-only` or small smoke filters on the login node, then submit dataset-level jobs on compute nodes.",
                "",
            ]
        )
    if results_df is not None:
        ok_rows = int(results_df["status"].eq("ok").sum()) if "status" in results_df else 0
        lines.extend(
            [
                "## Execution",
                "",
                f"- Rows produced: `{len(results_df)}`.",
                f"- Successful rows: `{ok_rows}`.",
                "",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_csv_set(values: str | None, cast: type = str) -> set[Any] | None:
    if values is None or values.strip() == "":
        return None
    return {cast(item.strip()) for item in values.split(",") if item.strip()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sources", default="original", help="Comma-separated sources: original,rerun")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names to include")
    parser.add_argument("--conditions", default=None, help="Comma-separated normalized or raw condition names")
    parser.add_argument("--sample-sizes", default=None, help="Comma-separated sample sizes")
    parser.add_argument(
        "--repetitions",
        default=None,
        help="Comma-separated exact repetitions retained after the standard max-repetitions selection",
    )
    parser.add_argument("--max-repetitions", type=int, default=None)
    parser.add_argument(
        "--synthetic-overrides-root",
        type=Path,
        default=None,
        help="Optional read-only recovery root containing dataset/datasets/synthetic overrides",
    )
    parser.add_argument(
        "--repetition-chunk-index",
        type=int,
        default=None,
        help="Zero-based deterministic repetition-rank chunk to execute",
    )
    parser.add_argument(
        "--repetition-num-chunks",
        type=int,
        default=None,
        help="Number of deterministic repetition-rank chunks",
    )
    parser.add_argument("--exclude-discovered-cpdag", action="store_true")
    parser.add_argument("--reference-source", choices=["train", "global_test"], default="train")
    parser.add_argument("--synthetic-sample-mode", choices=["full", "match_reference"], default="match_reference")
    parser.add_argument("--max-conditioning-set-size", type=int, default=DEFAULT_MAX_CONDITIONING_SET_SIZE)
    parser.add_argument("--max-triples", type=int, default=DEFAULT_MAX_TRIPLES)
    parser.add_argument("--exclude-columns", default="", help="Comma-separated column names removed before CI testing")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)

    include_sources = parse_csv_set(args.sources) or {"original"}
    datasets = parse_csv_set(args.datasets)
    conditions = parse_csv_set(args.conditions)
    sample_sizes = parse_csv_set(args.sample_sizes, int)
    repetitions = parse_csv_set(args.repetitions, int)

    specs = discover_dataset_specs(args.data_root, include_sources)
    runs = discover_synthetic_runs(
        specs,
        datasets=datasets,
        conditions=conditions,
        sample_sizes=sample_sizes,
        max_repetitions=args.max_repetitions,
        exclude_discovered_cpdag=args.exclude_discovered_cpdag,
    )
    if repetitions:
        runs = [run for run in runs if run.repetition in repetitions]
    runs = apply_synthetic_overrides(runs, args.synthetic_overrides_root)
    runs = select_repetition_chunk(runs, args.repetition_chunk_index, args.repetition_num_chunks)
    print(f"Discovered {len(specs)} dataset specs and {len(runs)} synthetic runs.")

    selected_runs_df = pd.DataFrame.from_records(
        [
            {
                "source": run.spec.source,
                "dataset": run.spec.dataset,
                "condition": run.condition,
                "raw_condition": run.raw_condition,
                "sample_size": run.sample_size,
                "repetition": run.repetition,
                "synthetic_path": str(run.path),
            }
            for run in runs
        ]
    )
    selected_runs_df.to_csv(output_dir / "selected_runs.csv", index=False)

    cost_df = estimate_cost(specs, runs, args)
    cost_path = output_dir / "tables" / "ci_preservation_cost_estimate.csv"
    cost_df.to_csv(cost_path, index=False)
    write_markdown_table(cost_df, output_dir / "tables" / "ci_preservation_cost_estimate.md")

    if args.estimate_only:
        write_summary(output_dir, args, cost_df, None)
        print(f"Wrote cost estimate to {cost_path}")
        return

    results_df = compute_preservation(args, runs)
    results_path = output_dir / "ci_preservation_results.csv"
    results_df.to_csv(results_path, index=False)

    aggregate_df = aggregate_results(results_df)
    aggregate_df.to_csv(output_dir / "tables" / "ci_preservation_aggregate.csv", index=False)
    write_markdown_table(aggregate_df, output_dir / "tables" / "ci_preservation_aggregate.md")

    wilcoxon_df = compute_wilcoxon_table(results_df, alpha=args.alpha)
    wilcoxon_df.to_csv(output_dir / "tables" / "ci_preservation_wilcoxon.csv", index=False)
    write_markdown_table(wilcoxon_df, output_dir / "tables" / "ci_preservation_wilcoxon.md")

    if not args.no_plots:
        plot_boxplots(results_df, output_dir)
    write_summary(output_dir, args, cost_df, results_df)
    print(f"Wrote results to {results_path}")


if __name__ == "__main__":
    main()
