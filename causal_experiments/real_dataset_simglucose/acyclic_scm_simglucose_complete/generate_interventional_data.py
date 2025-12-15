#!/usr/bin/env python3
"""
Generates INTERVENTIONAL datasets from the STATIC Simglucose SCM
to estimate the Average Treatment Effect (ATE).

This script creates two separate, PAIRED datasets:
1. One where the treatment variable is fixed to a value `t0`.
2. One where the treatment variable is fixed to a value `t1`.
The pairing ensures that for each row `i`, the patient, meal, and internal
randomness are identical across both datasets, isolating the treatment effect.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import sys
from typing import Dict, List, Optional, Tuple

# --- Robust path and import handling ---
try:
    current_dir = Path(__file__).resolve().parent
    # Assume that the IIT_simglucose folder is at the same level as the experiment folder
    iit_simglucose_path = current_dir.parent / "IIT_simglucose"
    
    if not iit_simglucose_path.exists():
        raise FileNotFoundError("The 'IIT_simglucose' folder was not found.")

    utils_path = iit_simglucose_path / "utils"
    
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))
    
    from t1dpatient_static_scm import T1DPatientStatic, Action, VAR_NAMES  # type: ignore
    import t1dpatient_static_scm as patient_module  # type: ignore

except (ImportError, FileNotFoundError) as e:
    print(f"Error while importing modules: {e}")
    print("Ensure that the folder structure is correct.")
    exit(1)


# -------------------------------
# Patient parameters to expose (aligned with static generator)
# -------------------------------
PATIENT_PARAM_NAMES: List[str] = [
    "BW", "Vi", "Ib", "Fsnc",
    "f", "kabs", "kmax", "kmin", "b", "d",
    "ka1", "ka2", "kd",
    "kp1", "kp3", "ke1", "ke2",
    "m1", "m2", "m4",
    "u2ss",
]


def _safe_float(x: Optional[float]) -> float:
    """Return a finite float value, defaulting to 0.0 for invalid inputs."""
    try:
        v = float(x)  # type: ignore[arg-type]
        if math.isfinite(v):
            return v
        return 0.0
    except Exception:
        return 0.0


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


def generate_interventional_csv(
    out_csv: Path,
    n_samples: int,
    intervention_var: str,
    intervention_val: float,
    shared_draws: List[Tuple[str, int, int]]
):
    """Generate a dataset by forcing an intervention, using a pre-sampled list of scenarios.
    
    Supports three types of interventions:
    - Exogenous action variable (action_insulin_U_per_min): Sets the insulin action directly
    - Exogenous action variable (action_CHO_g): Sets the CHO/meal action directly
    - Endogenous variable (e.g., subcut_insulin_1): Uses default values and passes interventions dict
    """
    rows: List[Dict[str, float]] = []
    
    # Check if we're intervening on an exogenous action variable
    is_action_intervention = (intervention_var in {"action_insulin_U_per_min", "action_CHO_g"})
    is_cho_intervention = (intervention_var == "action_CHO_g")
    
    if is_action_intervention:
        intervention = None  # No interventions dict for action variables
        print(f"Starting generation of {n_samples} samples with ACTION intervention: {intervention_var} = {intervention_val}")
    else:
        intervention = {intervention_var: intervention_val}
        print(f"Starting generation of {n_samples} samples with ENDOGENOUS intervention: {intervention_var} = {intervention_val}")

    for i, (pname, seed_i, meal_g) in enumerate(shared_draws):
        if (i + 1) % 1000 == 0:
            print(f"  ...generated {i + 1}/{n_samples} samples")
            
        p = T1DPatientStatic.withName(pname, seed=seed_i)

        # Compute default insulin rate from patient parameters (used if NOT intervening on insulin action)
        u2ss = getattr(p._params, "u2ss", p._params.get("u2ss", np.nan))
        BW = getattr(p._params, "BW", p._params.get("BW", np.nan))
        default_insulin_rate = _safe_float(u2ss) * _safe_float(BW) / 6000.0

        # Determine CHO and insulin values based on intervention type
        if is_cho_intervention:
            # Intervening on CHO: force meal_g to intervention value, use default insulin
            cho_value = float(intervention_val)
            insulin_rate = default_insulin_rate
        elif intervention_var == "action_insulin_U_per_min":
            # Intervening on insulin: use intervention value, keep original meal_g
            cho_value = float(meal_g)
            insulin_rate = intervention_val
        else:
            # Endogenous intervention: use original meal_g and default insulin
            cho_value = float(meal_g)
            insulin_rate = default_insulin_rate

        # Execute simulation: pass interventions dict only for endogenous variables
        p.sample_once(Action(CHO=cho_value, insulin=insulin_rate), interventions=intervention)

        rec: Dict[str, float] = {}

        for k, v in zip(VAR_NAMES, p.state):
            rec[k] = float(v)

        for par in PATIENT_PARAM_NAMES:
            rec[par] = p._params.get(par, np.nan)

        rec["action_CHO_g"] = float(cho_value)
        rec["action_insulin_U_per_min"] = float(insulin_rate)
        
        # Controller parameters (set to NaN for interventional data - not computed)
        rec["CR"] = np.nan
        rec["CF"] = np.nan

        rows.append(rec)

    columns = list(VAR_NAMES) + PATIENT_PARAM_NAMES + [
        "CR",
        "CF",
        "action_CHO_g",
        "action_insulin_U_per_min",
    ]
    df = pd.DataFrame(rows, columns=columns)
    
    # Apply original_2 ordering (deterministic shuffle) before saving
    original_2_indices = get_original_2_ordering(columns)
    df_reordered = df.iloc[:, original_2_indices]
    column_names_reordered = [columns[i] for i in original_2_indices]
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_reordered.to_csv(out_csv, index=False)
    print(f"[OK] Interventional data saved using original_2 ordering: {out_csv}")
    print(f"     Column order: {column_names_reordered[:5]}... (first 5 columns)")


def main():
    """
    Generate interventional datasets for ATE estimation.
    
    PARAMETERS USED FOR SIMGLUCOSE EXPERIMENT:
    ------------------------------------------
    
    PRIMARY RECOMMENDATION (CHO intervention - BEST for robust ATE):
        python generate_interventional_data_new.py \\
            --treatment action_CHO_g \\
            --t0 30 \\
            --t1 90 \\
            --n-samples 2000 \\
            --seed 42
    
    Rationale: Intervening on action_CHO_g (exogenous) is:
    - Aligned with do-calculus (surgical intervention on manipulable cause)
    - Clinically interpretable (carbohydrate intake is directly controllable)
    - More robust ATE: CHO directly increases glucose production (no truncation effects)
    - Larger and more stable effect compared to insulin (which has truncation at zero)
    - Tests direct causal pathway: CHO → glucose_kinetics → subcut_glucose
    
    SECONDARY OPTION (insulin intervention - for comparison):
        python generate_interventional_data_new.py \\
            --treatment action_insulin_U_per_min \\
            --t0 0.634 \\
            --t1 2.652 \\
            --n-samples 2000 \\
            --seed 42
    
    Rationale: Intervening on action_insulin_U_per_min (exogenous) is:
    - Aligned with do-calculus (surgical intervention on manipulable cause)
    - Clinically interpretable (what clinicians actually control)
    - Fair to synthetic model (action is in training distribution)
    - Tests full causal chain (action → subcut_insulin_1 → ... → subcut_glucose)
    - NOTE: May have smaller/less stable ATE due to truncation effects
    
    SECONDARY OPTION (endogenous variable - stress test for downstream pathway):
        python generate_interventional_data_new.py \\
            --treatment subcut_insulin_1 \\
            --t0 2597.22 \\
            --t1 10953.03 \\
            --n-samples 2000 \\
            --seed 42
    
    Rationale: Intervening on subcut_insulin_1 (endogenous) directly tests:
    - Downstream insulin→glucose pathway integrity
    - Whether synthetic model preserved insulin action effects
    - Use as secondary check if primary intervention passes
    
    Previous command (for reference - small effect, NOT recommended):
        python generate_interventional_data_new.py \\
            --treatment subcut_insulin_1 \\
            --t0 246.2194 \\
            --t1 279.3355 \\
            --n-samples 2000 \\
            --seed 42
    
    Variable statistics:
    - action_CHO_g (exogenous, RECOMMENDED): Range [0, 90], Mean=46.65, Median=60.0
      Unique values: [0, 30, 60, 90]
      Recommended: t0=30 (25th percentile), t1=90 (max) for robust ATE
    - action_insulin_U_per_min (exogenous): Range [0.004850, 19.029375], Mean=1.899, Median=1.466
      Percentiles: 25°=0.634, 75°=2.652 (318.1% difference recommended)
    - subcut_insulin_1 (endogenous): Range [27.55, 80476.96], Mean=7955.41, Median=6122.55
      Percentiles: 25°=2597.22, 75°=10953.03 (321.7% difference recommended)
    
    - n-samples = 2000: Standard sample size for interventional experiments
    - seed = 42: Fixed seed for reproducibility
    
    Output files (example for CHO intervention):
    - data_action_CHO_g_eq_30.csv (reference/t0 arm)
    - data_action_CHO_g_eq_90.csv (intervention/t1 arm)
    """
    script_dir = Path(__file__).resolve().parent
    default_out_dir = script_dir / 'generated_interventional'

    parser = argparse.ArgumentParser(description="Generate interventional data for ATE estimation.")
    parser.add_argument('--n-samples', type=int, default=2000, help="Number of samples to generate for each intervention arm.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--out-dir', type=str, default=str(default_out_dir), help="Output directory for CSV files.")
    parser.add_argument('--treatment', type=str, default='subcut_insulin_1', help="Treatment variable to intervene on.")
    parser.add_argument('--t0', type=float, required=True, help="Value for the first intervention arm (e.g., low dose).")
    parser.add_argument('--t1', type=float, required=True, help="Value for the second intervention arm (e.g., high dose).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()

    # --- Pre-sampling of scenarios ---
    print(f"Pre-sampling {args.n_samples} shared scenarios using seed {args.seed}...")
    rng = np.random.default_rng(args.seed)
    
    DATA_DIR = iit_simglucose_path / "data"
    PATIENT_PARA_FILE_SAMPLED = DATA_DIR / "sampled_insilico_vparams.csv"
    PATIENT_PARA_FILE_TEST = DATA_DIR / "insilico_vparams.csv"
    patient_module.PATIENT_PARA_FILE = str(PATIENT_PARA_FILE_SAMPLED)
    patient_module.PATIENT_PARA_FILE_TEST = str(PATIENT_PARA_FILE_TEST)
    params_sampled = pd.read_csv(PATIENT_PARA_FILE_SAMPLED)
    params_test = pd.read_csv(PATIENT_PARA_FILE_TEST)
    patient_names = params_sampled['Name'].tolist() + params_test['Name'].tolist()

    shared_draws: List[Tuple[str, int, int]] = []
    for _ in range(args.n_samples):
        pname = patient_names[rng.integers(0, len(patient_names))]
        seed_i = int(rng.integers(0, 1_000_000))
        meal_g = int(rng.choice([30, 60, 90]))
        shared_draws.append((pname, seed_i, meal_g))
    print("[OK] Scenarios pre-sampled.\n")
    
    # --- Generate dataset for the first world (T=t0) using shared scenarios ---
    out_csv_t0 = out_dir / f'data_{args.treatment}_eq_{args.t0}.csv'
    generate_interventional_csv(out_csv_t0, args.n_samples, args.treatment, args.t0, shared_draws=shared_draws)
    
    # --- Generate dataset for the second world (T=t1) using THE SAME scenarios ---
    out_csv_t1 = out_dir / f'data_{args.treatment}_eq_{args.t1}.csv'
    generate_interventional_csv(out_csv_t1, args.n_samples, args.treatment, args.t1, shared_draws=shared_draws)

    print("\n[COMPLETED] Generation of the two PAIRED interventional datasets finished.")

if __name__ == '__main__':
    main()
