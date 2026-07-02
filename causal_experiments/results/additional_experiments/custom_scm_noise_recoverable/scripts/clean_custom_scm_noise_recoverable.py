#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = EXPERIMENT_ROOT / "data"
RAW_ROOT = OUTPUT_ROOT / "raw"

METRICS = ["correlation_matrix_difference", "k_marginal_tvd", "nnaa"]
TRAIN_SIZES = [20, 50, 100, 200, 500]
N_CLEAN_REPS = 100

RUNS = {
    "noise2e-1": {
        "dataset_name": "custom_scm_noise0p2",
        "input_dir": RAW_ROOT / "results_noise2e-1",
        "output": OUTPUT_ROOT / "result_custom_scm_noise0p2_comparison_experiment_cleaned_reps_100.csv",
        "files": [
            "results_vanilla_original.csv",
            "results_vanilla_topological.csv",
            "results_vanilla_reverse_topological.csv",
            "results_dag.csv",
            "results_cpdag_minimal.csv",
            "results_cpdag_discovered.csv",
        ],
    },
    "noise1e-1": {
        "dataset_name": "custom_scm_noise0p1_robustness",
        "input_dir": RAW_ROOT / "results_noise1e-1",
        "output": OUTPUT_ROOT
        / "result_custom_scm_noise0p1_robustness_comparison_experiment_cleaned_reps_100.csv",
        "files": [
            "results_vanilla_original.csv",
            "results_cpdag_discovered.csv",
        ],
    },
    "noise5e-1": {
        "dataset_name": "custom_scm_noise0p5_robustness",
        "input_dir": RAW_ROOT / "results_noise5e-1",
        "output": OUTPUT_ROOT
        / "result_custom_scm_noise0p5_robustness_comparison_experiment_cleaned_reps_100.csv",
        "files": [
            "results_vanilla_original.csv",
            "results_cpdag_discovered.csv",
        ],
    },
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def metric_is_valid(row: dict[str, str]) -> bool:
    for metric in METRICS:
        try:
            if float(row.get(metric, "nan")) == -1.0:
                return False
        except ValueError:
            return False
    return True


def row_seed(row: dict[str, str]) -> int:
    return int(float(row["seed"]))


def row_train_size(row: dict[str, str]) -> int:
    return int(float(row["train_size"]))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_run(run_key: str, config: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    input_dir = Path(config["input_dir"])
    output = Path(config["output"])
    dataset_name = str(config["dataset_name"])
    file_names = list(config["files"])

    rows_by_file: dict[str, list[dict[str, str]]] = {}
    indexed: dict[str, dict[tuple[int, int], dict[str, str]]] = {}
    fieldnames: list[str] | None = None

    for file_name in file_names:
        path = input_dir / file_name
        rows = read_rows(path)
        rows_by_file[file_name] = rows
        indexed[file_name] = {(row_train_size(row), row_seed(row)): row for row in rows}
        if fieldnames is None:
            with path.open(newline="") as handle:
                fieldnames = list(csv.DictReader(handle).fieldnames or [])

    assert fieldnames is not None
    selected_seed_rows: list[dict[str, object]] = []
    cleaned_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, object]] = []

    for train_size in TRAIN_SIZES:
        valid_seed_sets = []
        for file_name, index in indexed.items():
            valid_seeds = {
                seed
                for (ts, seed), row in index.items()
                if ts == train_size and metric_is_valid(row)
            }
            valid_seed_sets.append(valid_seeds)
            summary_rows.append(
                {
                    "run": run_key,
                    "dataset": dataset_name,
                    "raw_file": file_name,
                    "train_size": train_size,
                    "raw_rows": sum(1 for row in rows_by_file[file_name] if row_train_size(row) == train_size),
                    "valid_rows": len(valid_seeds),
                    "invalid_rows": sum(
                        1
                        for row in rows_by_file[file_name]
                        if row_train_size(row) == train_size and not metric_is_valid(row)
                    ),
                }
            )
        common_valid = sorted(set.intersection(*valid_seed_sets))
        if len(common_valid) < N_CLEAN_REPS:
            raise RuntimeError(
                f"{run_key} train_size={train_size}: only {len(common_valid)} common valid seeds"
            )
        selected = common_valid[:N_CLEAN_REPS]
        selected_seed_rows.append(
            {
                "run": run_key,
                "dataset": dataset_name,
                "train_size": train_size,
                "n_common_valid_seeds": len(common_valid),
                "n_selected_seeds": len(selected),
                "selected_seed_min": selected[0],
                "selected_seed_max": selected[-1],
                "selected_seeds_json": json.dumps(selected),
            }
        )
        for file_name in file_names:
            for repetition, seed in enumerate(selected, start=1):
                row = dict(indexed[file_name][(train_size, seed)])
                row["repetition"] = str(repetition)
                cleaned_rows.append(row)

    write_rows(output, cleaned_rows, fieldnames)
    return summary_rows, selected_seed_rows


def main() -> None:
    all_summary_rows: list[dict[str, object]] = []
    all_selected_seed_rows: list[dict[str, object]] = []
    for run_key, config in RUNS.items():
        summary_rows, selected_seed_rows = clean_run(run_key, config)
        all_summary_rows.extend(summary_rows)
        all_selected_seed_rows.extend(selected_seed_rows)

    write_rows(
        RAW_ROOT / "cleaning_summary.csv",
        [{key: str(value) for key, value in row.items()} for row in all_summary_rows],
        [
            "run",
            "dataset",
            "raw_file",
            "train_size",
            "raw_rows",
            "valid_rows",
            "invalid_rows",
        ],
    )
    write_rows(
        RAW_ROOT / "selected_common_valid_seeds.csv",
        [{key: str(value) for key, value in row.items()} for row in all_selected_seed_rows],
        [
            "run",
            "dataset",
            "train_size",
            "n_common_valid_seeds",
            "n_selected_seeds",
            "selected_seed_min",
            "selected_seed_max",
            "selected_seeds_json",
        ],
    )


if __name__ == "__main__":
    main()
