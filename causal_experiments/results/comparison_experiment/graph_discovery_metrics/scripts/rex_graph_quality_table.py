#!/usr/bin/env python3
"""
Build graph-quality summary metrics for REX DAG-discovered outputs.

For each dataset/train size pair, the script computes:
- inversion_rate: fraction of recovered skeleton edges whose orientation is reversed.
- skeleton_recall: fraction of true skeleton edges recovered by REX.

The comparison is performed in variable-name space by remapping graph_structure indices
through each row's actual_column_order, so row-wise permutations are handled correctly.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd


Edge = Tuple[str, str]
UndirectedEdge = Tuple[str, str]


def _parse_dag_edges(graph_structure: str, actual_column_order: str) -> Set[Edge]:
    """Parse directed edges from graph_structure and map indices to variable names."""
    graph = ast.literal_eval(graph_structure)
    order = [item.strip() for item in str(actual_column_order).split(",")]

    edges: Set[Edge] = set()
    for child_idx, parents in graph.items():
        child_idx_int = int(child_idx)
        child_name = order[child_idx_int]
        for parent_idx in parents:
            parent_name = order[int(parent_idx)]
            edges.add((parent_name, child_name))
    return edges


def _to_skeleton(edges: Iterable[Edge]) -> Set[UndirectedEdge]:
    return {tuple(sorted(edge)) for edge in edges}


def _signature(edges: Set[Edge]) -> Tuple[Edge, ...]:
    return tuple(sorted(edges))


def _dataset_key_from_rex_filename(file_name: str) -> str:
    return file_name.replace("csuite_csuite_", "").replace("_dag_discovered_rex_results.csv", "")


def _load_true_graphs_by_train_size(base_file: Path) -> Dict[int, Set[Edge]]:
    """Load true DAG topological graphs keyed by train_size from the base cleaned CSV."""
    base_df = pd.read_csv(base_file)
    dag_topo = base_df[
        (base_df["algorithm"].astype(str).str.strip() == "dag")
        & (base_df["column_order"].astype(str).str.strip() == "topological")
    ].copy()
    if dag_topo.empty:
        raise ValueError(f"No dag/topological rows found in {base_file}")

    true_edges_by_ts: Dict[int, Set[Edge]] = {}
    for train_size, group in dag_topo.groupby("train_size"):
        parsed_edges = [
            _parse_dag_edges(row.graph_structure, row.actual_column_order)
            for row in group.itertuples(index=False)
        ]
        signatures = {_signature(edges) for edges in parsed_edges}
        if len(signatures) > 1:
            raise ValueError(
                f"Inconsistent true DAG across seeds for train_size={train_size} in {base_file}"
            )
        true_edges_by_ts[int(train_size)] = parsed_edges[0]

    return true_edges_by_ts


def build_quality_table(
    rex_files: Sequence[Path],
    base_data_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for rex_file in rex_files:
        dataset_key = _dataset_key_from_rex_filename(rex_file.name)
        dataset_name = f"csuite_{dataset_key}"
        base_file = base_data_dir / f"result_csuite_{dataset_key}_comparison_experiment_cleaned_reps_100.csv"
        if not base_file.exists():
            raise FileNotFoundError(f"Missing base comparison file for {dataset_name}: {base_file}")

        true_edges_by_ts = _load_true_graphs_by_train_size(base_file)
        rex_df = pd.read_csv(rex_file)

        per_row: List[Dict[str, object]] = []
        for row in rex_df.itertuples(index=False):
            train_size = int(row.train_size)
            if train_size not in true_edges_by_ts:
                raise ValueError(
                    f"train_size={train_size} in {rex_file} not found in base dag/topological rows ({base_file})"
                )

            discovered_edges = _parse_dag_edges(row.graph_structure, row.actual_column_order)
            true_edges = true_edges_by_ts[train_size]

            discovered_skeleton = _to_skeleton(discovered_edges)
            true_skeleton = _to_skeleton(true_edges)
            shared_skeleton = discovered_skeleton & true_skeleton

            inverted = 0
            for node_a, node_b in shared_skeleton:
                if (node_a, node_b) in true_edges:
                    true_dir = (node_a, node_b)
                    opp_dir = (node_b, node_a)
                else:
                    true_dir = (node_b, node_a)
                    opp_dir = (node_a, node_b)

                if true_dir in discovered_edges:
                    continue
                if opp_dir in discovered_edges:
                    inverted += 1

            inversion_rate = float("nan")
            if shared_skeleton:
                inversion_rate = inverted / len(shared_skeleton)

            skeleton_recall = float("nan")
            if true_skeleton:
                skeleton_recall = len(shared_skeleton) / len(true_skeleton)

            per_row.append(
                {
                    "dataset": dataset_name,
                    "train_size": train_size,
                    "inversion_rate": inversion_rate,
                    "skeleton_recall": skeleton_recall,
                    "shared_edges": len(shared_skeleton),
                    "true_edges": len(true_skeleton),
                }
            )

        per_row_df = pd.DataFrame(per_row)
        grouped = (
            per_row_df.groupby(["dataset", "train_size"], as_index=False)
            .agg(
                inversion_rate=("inversion_rate", "mean"),
                skeleton_recall=("skeleton_recall", "mean"),
                n_rows=("train_size", "size"),
                shared_edges_mean=("shared_edges", "mean"),
                true_edges=("true_edges", "first"),
            )
            .sort_values(["dataset", "train_size"], kind="mergesort")
        )
        rows.extend(grouped.to_dict("records"))

    out_df = pd.DataFrame(rows).sort_values(["dataset", "train_size"], kind="mergesort").reset_index(drop=True)
    return out_df


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    comparison_dir = next(
        (parent for parent in script_dir.parents if parent.name == "comparison_experiment"),
        script_dir.parents[1],
    )
    preferred_rex_dir = comparison_dir / "data" / "dag_discovered_rex"
    legacy_rex_dir = comparison_dir / "data" / "dag_discovered"
    default_rex_dir = preferred_rex_dir if preferred_rex_dir.exists() else legacy_rex_dir
    default_base_data_dir = comparison_dir / "data"
    default_output_dir = script_dir / "outputs" / "rex_graph_quality"
    default_output_csv = default_output_dir / "rex_graph_quality_table.csv"
    default_output_md = default_output_dir / "rex_graph_quality_table.md"

    parser = argparse.ArgumentParser(description="Build REX DAG graph-quality summary table.")
    parser.add_argument(
        "--rex-dir",
        type=Path,
        default=default_rex_dir,
        help=(
            "Directory containing *_dag_discovered_rex_results.csv files "
            "(preferred: data/dag_discovered_rex; legacy: data/dag_discovered)."
        ),
    )
    parser.add_argument(
        "--base-data-dir",
        type=Path,
        default=default_base_data_dir,
        help="Directory containing result_csuite_*_comparison_experiment_cleaned_reps_100.csv files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=default_output_csv,
        help="Output CSV path for the summary table.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=default_output_md,
        help="Output Markdown table path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rex_files = sorted(args.rex_dir.glob("csuite_csuite_*_dag_discovered_rex_results.csv"))
    if not rex_files:
        raise FileNotFoundError(f"No REX files found in {args.rex_dir}")

    summary = build_quality_table(rex_files=rex_files, base_data_dir=args.base_data_dir)
    summary["inversion_rate"] = summary["inversion_rate"].astype(float)
    summary["skeleton_recall"] = summary["skeleton_recall"].astype(float)
    summary["shared_edges_mean"] = summary["shared_edges_mean"].astype(float)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    rounded = summary.round(
        {
            "inversion_rate": 4,
            "skeleton_recall": 4,
            "shared_edges_mean": 3,
        }
    )
    md_lines = [
        "| " + " | ".join(rounded.columns) + " |",
        "| " + " | ".join(["---"] * len(rounded.columns)) + " |",
    ]
    for values in rounded.itertuples(index=False, name=None):
        rendered = ["" if pd.isna(v) else str(v) for v in values]
        md_lines.append("| " + " | ".join(rendered) + " |")
    args.output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[INFO] Wrote CSV: {args.output_csv}")
    print(f"[INFO] Wrote Markdown: {args.output_md}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
