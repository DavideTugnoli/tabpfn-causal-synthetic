#!/usr/bin/env python3
"""
Generates i.i.d. single-slice snapshot data from the STATIC SimGlucose SCM,
exporting BOTH the original 13 endogenous variables AND the patient-level
parameters used by the structural equations, row by row.

It also exports the extended ground-truth DAG (variables + patient parameters)
in the same JSON format you used before (column_names, dag_dict, cpdag_dict).

This script is designed to run from its location and automatically find
the necessary sibling directories ('IIT_simglucose').
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pydot
import sys
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import math

# --- Robust path handling (same strategy as your version) ---
try:
    current_dir = Path(__file__).resolve().parent
    iit_simglucose_path = current_dir.parent / "IIT_simglucose"

    if not iit_simglucose_path.exists():
        raise FileNotFoundError("Folder 'IIT_simglucose' not found.")

    utils_path = iit_simglucose_path / "utils"
    causal_experiments_path = current_dir.parents[1]  # .../causal_experiments/

    if str(iit_simglucose_path) not in sys.path:
        sys.path.insert(0, str(iit_simglucose_path))
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))
    if str(causal_experiments_path) not in sys.path:
        sys.path.insert(0, str(causal_experiments_path))

    from t1dpatient_static_scm import T1DPatientStatic, Action, VAR_NAMES  # type: ignore
    import t1dpatient_static_scm as patient_module  # type: ignore
    import pump as pump_module  # type: ignore

    # Optional import for CPDAG calculation
    try:
        from utils.dag_utils import dag_to_ideal_cpdag  # type: ignore
    except Exception:
        dag_to_ideal_cpdag = None

except ImportError as e:
    print(f"Error while importing necessary modules: {e}")
    print("Ensure the directory structure is correct and that `__init__.py` files exist:")
    print("- .../acyclic_scm_simglucose/ (contains this script)")
    print("- .../IIT_simglucose/utils/t1dpatient_static_scm.py")
    print("- .../causal_experiments/utils/dag_utils.py")
    sys.exit(1)

# -------------------------------
# Patient parameters we make observable (exogenous roots)
# -------------------------------
PATIENT_PARAM_NAMES: List[str] = [
    # Core size/volumes + baselines
    "BW", "Vi", "Ib", "Fsnc",
    # Gut / appearance
    "f", "kabs", "kmax", "kmin", "b", "d",
    # Subcutaneous insulin kinetics
    "ka1", "ka2", "kd",
    # Endogenous glucose production & excretion
    "kp1", "kp3", "ke1", "ke2",
    # Plasma insulin balance
    "m1", "m2", "m4",
    # Insulin steady-state utility (used to compute basal infusion in the original simulator)
    "u2ss",
]

# Validate uniqueness
assert len(PATIENT_PARAM_NAMES) == len(set(PATIENT_PARAM_NAMES)), "Duplicate names in PATIENT_PARAM_NAMES"

# -------------------------------
# Controller parameters exposed (Quest table)
# -------------------------------
CONTROLLER_PARAM_NAMES: List[str] = ["CR", "CF"]
assert len(CONTROLLER_PARAM_NAMES) == len(set(CONTROLLER_PARAM_NAMES)), "Duplicate names in CONTROLLER_PARAM_NAMES"

# -------------------------------
# Observable exogenous actions (kept deterministic per row)
# -------------------------------
ACTION_NAMES: List[str] = [
    "action_CHO_g",               # grams of carbohydrates over the sampling minute
    "action_insulin_U_per_min",   # insulin infusion rate (U/min)
]

assert len(ACTION_NAMES) == len(set(ACTION_NAMES)), "Duplicate names in ACTION_NAMES"

# -------------------------------
# Extended DAG: child -> [parents]
# (includes endogenous edges as in your ground truth + param -> variable edges)
# -------------------------------
def get_static_dag_with_params() -> Dict[str, List[str]]:
    # Start with all nodes
    all_nodes: List[str] = list(VAR_NAMES) + PATIENT_PARAM_NAMES + CONTROLLER_PARAM_NAMES + ACTION_NAMES
    child_to_parents: Dict[str, List[str]] = {name: [] for name in all_nodes}

    # --- Endogenous edges (verbatim from your original ground truth) ---
    # child <- parents
    child_to_parents["sto_solid"]           += []
    child_to_parents["sto_liquid"]          += ["sto_solid"]
    child_to_parents["intestine"]           += ["sto_solid", "sto_liquid"]
    child_to_parents["glucose_kinetics_1"]  += ["intestine", "glucose_kinetics_2", "ins_action_prod_2"]
    child_to_parents["glucose_kinetics_2"]  += []
    child_to_parents["insulin_kinetics"]    += ["insulin_liver", "subcut_insulin_1", "subcut_insulin_2"]
    child_to_parents["ins_action_utilization"] += ["insulin_kinetics"]
    child_to_parents["ins_action_prod_1"]   += ["insulin_kinetics"]
    child_to_parents["ins_action_prod_2"]   += ["ins_action_prod_1"]
    child_to_parents["insulin_liver"]       += []
    child_to_parents["subcut_insulin_1"]    += []
    child_to_parents["subcut_insulin_2"]    += ["subcut_insulin_1"]
    child_to_parents["subcut_glucose"]      += ["glucose_kinetics_1"]

    # --- Parameter -> variable edges (from structural equations) ---
    # Stomach / intestine
    child_to_parents["sto_solid"]           += ["kmax"]
    child_to_parents["sto_liquid"]          += ["kmax", "kmin", "b", "d"]
    child_to_parents["intestine"]           += ["kabs", "kmax", "kmin", "b", "d"]

    # Subcutaneous insulin chain
    child_to_parents["subcut_insulin_1"]    += ["ka1", "kd", "BW"]
    child_to_parents["subcut_insulin_2"]    += ["kd", "ka2"]

    # Plasma insulin kinetics
    child_to_parents["insulin_kinetics"]    += ["m1", "m2", "m4", "ka1", "ka2"]

    # Insulin actions
    child_to_parents["ins_action_utilization"] += ["Vi", "Ib"]
    child_to_parents["ins_action_prod_1"]      += ["Vi"]

    # Glucose kinetics (primary)
    child_to_parents["glucose_kinetics_1"]  += ["f", "kabs", "BW", "kp1", "kp3", "ke1", "ke2", "Fsnc"]

    # --- Exogenous actions ---
    child_to_parents["sto_solid"]           += ["action_CHO_g"]
    child_to_parents["sto_liquid"]          += ["action_CHO_g"]
    child_to_parents["intestine"]           += ["action_CHO_g"]
    child_to_parents["subcut_insulin_1"]    += ["action_insulin_U_per_min"]
    # In the original simulator, basal insulin depends on u2ss and BW, and bolus reacts to carbohydrate intake
    child_to_parents["action_insulin_U_per_min"] += ["BW", "u2ss", "CR", "CF", "action_CHO_g"]

    # Parameters are exogenous roots, except for physiological relationships
    for p in PATIENT_PARAM_NAMES:
        child_to_parents[p] += []
    for c in CONTROLLER_PARAM_NAMES:
        child_to_parents[c] += []
    
    # Add physiological relationship: m4 → m2 (based on m2 = -m4/HE_b)
    child_to_parents["m2"] += ["m4"]

    # Ensure actions appear in the adjacency dict even if parents are added above
    for a in ACTION_NAMES:
        child_to_parents[a] += []

    return child_to_parents


def export_graphs_with_params(out_path: Path):
    """
    Exports the extended DAG (variables + patient params) in JSON (graphs.json-like) and PNG.
    Uses original_2 ordering (deterministic shuffle) to match CSV column order.
    """
    named_dag = get_static_dag_with_params()
    column_names_canonical = list(VAR_NAMES) + PATIENT_PARAM_NAMES + CONTROLLER_PARAM_NAMES + ACTION_NAMES
    
    # Apply original_2 ordering (same as CSV) to column_names
    original_2_indices = get_original_2_ordering(column_names_canonical)
    column_names = [column_names_canonical[i] for i in original_2_indices]
    
    # Create mapping: canonical index -> original_2 index
    canonical_to_original2 = {canonical_idx: orig2_idx for orig2_idx, canonical_idx in enumerate(original_2_indices)}
    
    # Build DAG with original_2 indices (DAG structure remains the same, just reindexed)
    name_to_idx_canonical = {name: i for i, name in enumerate(column_names_canonical)}
    name_to_idx = {name: canonical_to_original2[i] for name, i in name_to_idx_canonical.items()}
    dag_idx = {str(name_to_idx[c]): [name_to_idx[p] for p in parents] for c, parents in named_dag.items()}

    # --- CPDAG (optional) ---
    cpdag_idx = None
    if dag_to_ideal_cpdag is not None:
        try:
            # dag_to_ideal_cpdag expects {int: [int]} → use canonical indices for calculation
            dag_int_canonical = {name_to_idx_canonical[c]: [name_to_idx_canonical[p] for p in parents] for c, parents in named_dag.items()}
            cpdag_canonical = dag_to_ideal_cpdag(dag_int_canonical)
            # Reindex CPDAG to original_2 ordering
            cpdag_idx = {}
            for canonical_node_idx, connections in cpdag_canonical.items():
                orig2_node_idx = canonical_to_original2[canonical_node_idx]
                cpdag_idx[str(orig2_node_idx)] = {
                    "parents": [str(canonical_to_original2[p]) for p in connections.get("parents", [])],
                    "undirected": [str(canonical_to_original2[u]) for u in connections.get("undirected", [])]
                }
            print("[INFO] Successfully calculated CPDAG and reindexed to original_2 ordering.")
        except Exception as e:
            print(f"[WARN] CPDAG calculation failed: {e}. Proceeding without it.")

    # Export JSON
    graphs_json = {'column_names': column_names, 'dag_dict': dag_idx, 'cpdag_dict': cpdag_idx}
    json_path = out_path.with_suffix('.graphs.json')
    with open(json_path, 'w') as f:
        json.dump(graphs_json, f, indent=2)
    print(f"[OK] Saved extended graphs JSON: {json_path}")

    # Draw a PNG (directed DAG) - use column_names in original_2 order
    draw_dag_png(out_path.with_suffix('.dag.png'), dag_idx, column_names)

    # Draw CPDAG if available - use column_names in original_2 order
    if isinstance(cpdag_idx, dict):
        draw_cpdag_png(out_path.with_suffix('.cpdag.png'), cpdag_idx, column_names)

    return dag_idx, cpdag_idx, column_names


def draw_dag_png(path: Path, dag_idx: Dict[int, List[int]], column_names: List[str]):
    dot = pydot.Dot(graph_type="digraph", label="Static SCM + Patient Params (DAG)", fontsize=20, labeljust="t")
    # Use column_names (which is in original_2 order) for node labels
    for name in column_names:
        if name in VAR_NAMES:
            dot.add_node(pydot.Node(name, shape='box', style='rounded,filled', fillcolor="#eef5ff"))
        elif name in PATIENT_PARAM_NAMES:
            dot.add_node(pydot.Node(name, shape='ellipse', style='filled', fillcolor="#f6f6f6"))
        elif name in CONTROLLER_PARAM_NAMES:
            dot.add_node(pydot.Node(name, shape='ellipse', style='filled', fillcolor="#fff5f0"))
        elif name in ACTION_NAMES:
            dot.add_node(pydot.Node(name, shape='diamond', style='filled', fillcolor="#fff2cc"))
    # Edges (dag_idx uses original_2 indices)
    for child_idx_str, parents in dag_idx.items():
        child_idx = int(child_idx_str)
        for parent_idx in parents:
            dot.add_edge(pydot.Edge(
                column_names[parent_idx],
                column_names[child_idx]
            ))
    # Render
    import io
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(plt.imread(io.BytesIO(dot.create_png(prog="dot"))))
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved DAG PNG: {path}")


def draw_cpdag_png(path: Path, cpdag_idx: Dict[str, Dict[str, List[str]]], column_names: List[str]):
    dot = pydot.Dot(graph_type="graph", label="Static SCM + Patient Params (CPDAG)", fontsize=20, labeljust="t")
    # Use column_names (which is in original_2 order) for node labels
    for name in column_names:
        if name in VAR_NAMES:
            dot.add_node(pydot.Node(name, shape='box', style='rounded,filled', fillcolor="#eef5ff"))
        elif name in PATIENT_PARAM_NAMES:
            dot.add_node(pydot.Node(name, shape='ellipse', style='filled', fillcolor="#f6f6f6"))
        elif name in CONTROLLER_PARAM_NAMES:
            dot.add_node(pydot.Node(name, shape='ellipse', style='filled', fillcolor="#fff5f0"))
        elif name in ACTION_NAMES:
            dot.add_node(pydot.Node(name, shape='diamond', style='filled', fillcolor="#fff2cc"))

    processed_undirected = set()
    for node_idx_str, connections in cpdag_idx.items():
        node_idx = int(node_idx_str)
        # Directed edges (parents -> node)
        for parent_idx_str in connections.get('parents', []):
            parent_idx = int(parent_idx_str)
            dot.add_edge(pydot.Edge(
                column_names[parent_idx],
                column_names[node_idx],
                dir='forward'))
        # Undirected edges (node -- neighbor)
        for neighbor_idx_str in connections.get('undirected', []):
            neighbor_idx = int(neighbor_idx_str)
            u, v = min(node_idx, neighbor_idx), max(node_idx, neighbor_idx)
            if (u, v) not in processed_undirected:
                dot.add_edge(pydot.Edge(
                    column_names[u],
                    column_names[v],
                    dir='none'))
                processed_undirected.add((u, v))

    import io
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(plt.imread(io.BytesIO(dot.create_png(prog="dot"))))
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved CPDAG PNG: {path}")


# -------------------------------
# Data generation (with params)
# -------------------------------
def _safe_float(x: Optional[float]) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return 0.0
    except Exception:
        return 0.0


def _load_quest_tables(data_dir: Path) -> pd.DataFrame:
    quest_sampled = data_dir / "sampled_insilico_quest.csv"
    quest_test = data_dir / "insilico_quest.csv"
    frames = []
    for path in (quest_sampled, quest_test):
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No Quest controller tables found under {data_dir}")
    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df.drop_duplicates(subset="Name", keep="first")
    df = df.set_index("Name")
    return df


def _quantized_basal_and_bolus(
    pump,
    basal_rate_nominal: float,
    bolus_rate_nominal: float,
) -> Tuple[float, float]:
    basal = float(pump.basal(basal_rate_nominal))
    bolus = float(pump.bolus(bolus_rate_nominal))
    return basal, bolus


def _sample_glucose_observation(rng: np.random.Generator, target: float = 140.0) -> float:
    # Mimic CGM readings around the target with physiologically plausible spread
    reading = rng.normal(loc=target, scale=25.0)
    return float(np.clip(reading, 60.0, 300.0))


def _compute_controller_insulin_rate(
    rng: np.random.Generator,
    patient_params: pd.Series,
    quest_df: pd.DataFrame,
    pump,
    meal_g: float,
) -> Tuple[float, float, float]:
    # Nominal basal directly from simulator definition
    basal_nominal = (
        _safe_float(patient_params.get("u2ss"))
        * _safe_float(patient_params.get("BW"))
        / 6000.0
    )
    basal_nominal = max(basal_nominal, 0.0)

    # Retrieve Quest parameters (carb ratio CR in g/U, correction factor CF in mg/dL per U)
    default_cr = 12.0  # Conservative adult average (g/U)
    default_cf = 40.0  # Conservative correction factor (mg/dL per U)
    carb_ratio = default_cr
    correction_factor = default_cf
    try:
        quest_row = quest_df.loc[str(patient_params.Name)]
        if hasattr(quest_row, "__iter__") and getattr(quest_row, "size", 0) > 0:
            carb_ratio = _safe_float(quest_row.get("CR"))
            correction_factor = _safe_float(quest_row.get("CF"))
        else:
            carb_ratio = default_cr
            correction_factor = default_cf
    except KeyError:
        pass

    carb_ratio = carb_ratio if carb_ratio > 0 else default_cr
    correction_factor = correction_factor if correction_factor > 0 else default_cf

    # Meal-derived bolus in insulin units (U)
    meal_units = meal_g / carb_ratio

    # Simple correction bolus (positive if CGM above target, negative otherwise)
    glucose_obs = _sample_glucose_observation(rng)
    target_glucose = 140.0
    correction_units = (glucose_obs - target_glucose) / correction_factor

    total_bolus_units = max(0.0, meal_units + correction_units)

    # Allow small random temporary basal modulation (±10 %) to mimic controller behaviour
    temp_basal_factor = rng.uniform(0.9, 1.1)
    basal_nominal *= temp_basal_factor

    # Pump expects U/min – distribute the bolus over the controller window (~3 minutes)
    controller_window = 3.0
    pump_sample_time = float(pump._params.get("sample_time", 1.0))
    pump_sample_time = pump_sample_time if pump_sample_time > 0 else 1.0
    effective_window = max(pump_sample_time, 1.0) * controller_window
    bolus_rate_nominal = total_bolus_units / effective_window

    # Inject noise so that even identical meals don't collapse to identical rates
    bolus_rate_nominal = max(
        0.0,
        bolus_rate_nominal + rng.normal(0.0, 0.08 * (bolus_rate_nominal + 1e-3)),
    )

    basal_rate_quantized, bolus_rate_quantized = _quantized_basal_and_bolus(
        pump,
        basal_nominal,
        bolus_rate_nominal,
    )
    return basal_rate_quantized + bolus_rate_quantized, carb_ratio, correction_factor


def get_original_2_ordering(column_names: List[str]) -> List[int]:
    """Calculate original_2 ordering (deterministic shuffle of original order).
    
    Uses the same seed (314159) as comparison_experiment.py to ensure consistency.
    """
    rng = np.random.default_rng(314159)
    original_indices = list(range(len(column_names)))
    original_2_indices = original_indices.copy()
    for _ in range(10):
        rng.shuffle(original_2_indices)
        if original_2_indices != original_indices:
            break
    return original_2_indices


def generate_static_csv_with_params(out_csv: Path, n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []

    # Locate data dir and overwrite global paths in the patient module
    DATA_DIR = iit_simglucose_path / "data"
    PARAM_SAMPLED = DATA_DIR / "sampled_insilico_vparams.csv"
    PARAM_TEST = DATA_DIR / "insilico_vparams.csv"

    if not PARAM_SAMPLED.exists() or not PARAM_TEST.exists():
        print(f"Error: Parameter files not found in the expected path: {DATA_DIR}")
        sys.exit(1)

    patient_module.PATIENT_PARA_FILE = str(PARAM_SAMPLED)
    patient_module.PATIENT_PARA_FILE_TEST = str(PARAM_TEST)

    pump_module.INSULIN_PUMP_PARA_FILE = str(DATA_DIR / "pump_params.csv")
    pump = pump_module.InsulinPump.withName("Insulet")

    df_sampled = pd.read_csv(PARAM_SAMPLED)
    df_test    = pd.read_csv(PARAM_TEST)
    quest_df = _load_quest_tables(DATA_DIR)
    names_sampled = df_sampled['Name'].tolist()
    names_test    = df_test['Name'].tolist()
    patient_names = names_sampled + names_test

    print(f"Loaded {len(patient_names)} patients in total.")
    print(f"Generating {n_samples} static samples...")

    for _ in range(n_samples):
        pname = patient_names[rng.integers(0, len(patient_names))]
        seed_i = int(rng.integers(0, 1_000_000))
        p = T1DPatientStatic.withName(pname, seed=seed_i)

        # Exogenous inputs: align with original simulator semantics
        # Meal size is exogenous; insulin basal depends on u2ss and BW
        meal_g = int(rng.choice([0, 30, 60, 90], p=[0.15, 0.35, 0.30, 0.20]))
        u2ss = p._params.get('u2ss', np.nan)
        BW   = p._params.get('BW',   np.nan)
        basal_rate = _safe_float(u2ss) * _safe_float(BW) / 6000.0  # U/min

        insulin_rate, carb_ratio, correction_factor = _compute_controller_insulin_rate(
            rng=rng,
            patient_params=p._params,
            quest_df=quest_df,
            pump=pump,
            meal_g=meal_g,
        )

        p.sample_once(Action(CHO=meal_g, insulin=insulin_rate))

        # Build one record: endogenous variables + parameters + book-keeping
        rec = {}

        # 13 endogenous variables (canonical order)
        for k, v in zip(VAR_NAMES, p.state):
            rec[k] = float(v)

        # Patient parameters (only those we declare; missing → NaN)
        for par in PATIENT_PARAM_NAMES:
            rec[par] = p._params.get(par, np.nan)

        # Controller parameters (Quest) observed for transparency
        rec["CR"] = carb_ratio
        rec["CF"] = correction_factor

        # Observed actions (kept for causal completeness)
        rec[ACTION_NAMES[0]] = meal_g
        rec[ACTION_NAMES[1]] = insulin_rate

        rows.append(rec)

    column_names = list(VAR_NAMES) + PATIENT_PARAM_NAMES + CONTROLLER_PARAM_NAMES + ACTION_NAMES
    df = pd.DataFrame(rows, columns=column_names)

    # Apply original_2 ordering (deterministic shuffle) before saving
    original_2_indices = get_original_2_ordering(column_names)
    df_reordered = df.iloc[:, original_2_indices]
    column_names_reordered = [column_names[i] for i in original_2_indices]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_reordered.to_csv(out_csv, index=False)
    print(f"[OK] Saved static SCM data (with params) using original_2 ordering: {out_csv}")
    print(f"     Column order: {column_names_reordered[:5]}... (first 5 columns)")


def main():
    script_dir = Path(__file__).resolve().parent
    default_out_dir = script_dir / 'generated_static_scm'

    parser = argparse.ArgumentParser(description="Generate data from the Static SimGlucose SCM (with patient params).")
    parser.add_argument('--n-samples', type=int, default=6000, help="Number of i.i.d. samples to generate.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--out-dir', type=str, default=str(default_out_dir), help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f'simglucose_static_scm_with_params_{args.n_samples}.csv'

    # 1) Data with params
    generate_static_csv_with_params(out_csv, args.n_samples, args.seed)

    # 2) Extended DAG (+ CPDAG if utility is available)
    graphs_out = out_dir / 'simglucose_static_scm_with_params'
    dag_idx, cpdag_idx, column_names = export_graphs_with_params(graphs_out)

if __name__ == '__main__':
    main()
