import os
import numpy as np
import pandas as pd
from collections import namedtuple
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

Action = namedtuple("patient_action", ["CHO", "insulin"])  # CHO in g/min over this minute; insulin in U/min
Observation = namedtuple("observation", ["Gsub"])

PATIENT_PARA_FILE = os.path.join("data", "sampled_insilico_vparams.csv")
PATIENT_PARA_FILE_TEST = os.path.join("data", "insilico_vparams.csv")

VAR_NAMES = [
    "sto_solid", "sto_liquid", "intestine",
    "glucose_kinetics_1", "glucose_kinetics_2",
    "insulin_kinetics",
    "ins_action_utilization", "ins_action_prod_1", "ins_action_prod_2",
    "insulin_liver", "subcut_insulin_1", "subcut_insulin_2",
    "subcut_glucose",
]

def _clip_nonneg(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)

class T1DPatientStatic:
    """
    Static Feed-Forward Surrogate (S3-Glucose).
    One-shot SCM (no time). Generates 13 endogenous variables in a DAG-consistent topological order:
      stomach → intestine → glucose_kinetics_1 → subcut_glucose
      insulin chain → insulin actions → glucose_kinetics_1
      glucose_kinetics_2 is a root (patient baseline) → glucose_kinetics_1

    Parents and nonlinearities mirror your acyclic ODE; leak/self terms removed and absorbed into node-specific noise.
    """
    SAMPLE_TIME = 1.0  # minutes (kept for compatibility with env-like interfaces)

    def __init__(
        self,
        params: pd.Series,
        seed: Optional[int] = None,
        noise_scales: Optional[Dict[str, float]] = None,
    ):
        self._params = params
        self._seed = int(seed) if seed is not None else None
        self._rng = np.random.default_rng(self._seed)
        # Small, node-specific relative noise (fraction of nominal magnitude)
        default_scales = {
            "sto_solid": 0.03, "sto_liquid": 0.03, "intestine": 0.03,
            "glucose_kinetics_1": 0.03, "glucose_kinetics_2": 0.02,
            "insulin_kinetics": 0.03,
            "ins_action_utilization": 0.03, "ins_action_prod_1": 0.03, "ins_action_prod_2": 0.03,
            "insulin_liver": 0.02, "subcut_insulin_1": 0.03, "subcut_insulin_2": 0.03,
            "subcut_glucose": 0.02,
        }
        self._noise_scales = default_scales if noise_scales is None else {**default_scales, **noise_scales}

        self.name = str(self._params.Name) if "Name" in self._params else "unknown"
        self.state = np.zeros(13, dtype=float)
        self.state_hist = []  # keep for compatibility

        # Useful constants
        self.Vg = float(self._params.Vg)     # distribution volume for glucose
        self.Vi = float(self._params.Vi)     # insulin distribution volume

    # ----- Constructors identical to your T1DPatient API -----
    @classmethod
    def withID(cls, patient_id: int, **kwargs):
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params.iloc[int(patient_id) - 1, :]
        return cls(params, **kwargs)

    @classmethod
    def withName(cls, name: str, **kwargs):
        if "test" in name:
            patient_params = pd.read_csv(PATIENT_PARA_FILE_TEST)
            params = patient_params.loc[patient_params.Name == name].squeeze()
        else:
            patient_params = pd.read_csv(PATIENT_PARA_FILE)
            params = patient_params.loc[patient_params.Name == name].squeeze()
        return cls(params, **kwargs)

    # ----- Observation (CGM-like) -----
    @property
    def observation(self) -> Observation:
        # x[12] is subcutaneous glucose (mg/kg); divide by Vg to get mg/dL-like
        Gsub = float(self.state[12]) / self.Vg if self.Vg != 0 else 0.0
        return Observation(Gsub=Gsub)

    # ----- One-shot sample (core) -----
    def _noise(self, key: str, ref: float) -> float:
        scale = self._noise_scales.get(key, 0.03)
        return float(self._rng.normal(0.0, scale * (abs(ref) + 1e-6)))

    def _kgut(self, x0: float, meal_mg: float) -> float:
        p = self._params
        # Use the same form as ODE, with Dbar approximated from current meal (no history)
        qsto = x0  # approximate qsto ~ sto_solid for static mapping
        Dbar = meal_mg
        if Dbar <= 0:
            return float(p.kmax)
        aa = 5.0 / 2.0 / (1.0 - p.b) / Dbar
        cc = 5.0 / 2.0 / p.d / Dbar
        kgut = p.kmin + (p.kmax - p.kmin) / 2.0 * (
            np.tanh(aa * (qsto - p.b * Dbar)) - np.tanh(cc * (qsto - p.d * Dbar)) + 2.0
        )
        return float(kgut)

    def sample_once(self, action: Action, interventions: Optional[Dict[str, float]] = None):
        """
        Generate a single i.i.d. sample (13 variables) from the static SCM given the exogenous inputs.
        Writes into self.state and appends to self.state_hist for compatibility.
        """
        p = self._params

        meal_gpm = float(action.CHO)           # g/min over this minute
        meal_mgpm = meal_gpm * 1000.0          # mg/min
        insulin_Upm = float(action.insulin)    # U/min
        insulin_pmolkgpm = insulin_Upm * 6000.0 / float(p.BW) if p.BW != 0 else 0.0

        x = np.zeros(13, dtype=float)

        # --- Stomach / Intestine path ---
        # sto_solid ~ steady value where inflow (meal) balances emptying at kmax
        x0_nom = meal_mgpm / float(p.kmax) if p.kmax != 0 else 0.0
        x[0] = max(0.0, x0_nom + self._noise("sto_solid", x0_nom))

        kgut = self._kgut(x[0], meal_mgpm)
        x1_nom = (float(p.kmax) * x[0] / kgut) if kgut > 0 else 0.0
        x[1] = max(0.0, x1_nom + self._noise("sto_liquid", x1_nom))

        x2_nom = kgut * x[1] / float(p.kabs) if p.kabs != 0 else 0.0
        x[2] = max(0.0, x2_nom + self._noise("intestine", x2_nom))

        # --- Insulin subcutaneous chain ---
        if interventions and "subcut_insulin_1" in interventions:
            x[10] = interventions["subcut_insulin_1"]
        else:
            # Use the physiologically scaled insulin flow (pmol/kg/min); previous code mistakenly used U/min.
            x10_nom = insulin_pmolkgpm / (float(p.ka1) + float(p.kd)) if (p.ka1 + p.kd) != 0 else 0.0
            x[10] = max(0.0, x10_nom + self._noise("subcut_insulin_1", x10_nom))

        x11_nom = float(p.kd) * x[10] / float(p.ka2) if p.ka2 != 0 else 0.0
        x[11] = max(0.0, x11_nom + self._noise("subcut_insulin_2", x11_nom))

        # Hepatic insulin (acyclic version had pure decay → near zero without IV hepatic input)
        x9_nom = 0.0
        x[9] = max(0.0, x9_nom + self._noise("insulin_liver", 1.0))

        # Plasma insulin (steady algebraic balance)
        denom = (float(p.m2) + float(p.m4))
        num = float(p.m1) * x[9] + float(p.ka1) * x[10] + float(p.ka2) * x[11]
        x5_nom = num / denom if denom != 0 else 0.0
        x[5] = max(0.0, x5_nom + self._noise("insulin_kinetics", x5_nom))

        It = x[5] / float(p.Vi) if p.Vi != 0 else 0.0
        x6_nom = It - float(p.Ib)  # action on utilization
        x[6] = x6_nom + self._noise("ins_action_utilization", x6_nom)

        x7_nom = It  # action on production (1)
        x[7] = x7_nom + self._noise("ins_action_prod_1", x7_nom)

        x8_nom = x[7]  # action on production (2)
        x[8] = x8_nom + self._noise("ins_action_prod_2", x8_nom)

        # --- Glucose kinetics auxiliary state (root here) ---
        # In the acyclic ODE, x4 had no upstream inflow (after removing +k1*x3). Treat as baseline + noise.
        # Use patient initial baseline from parameter table if available:
        try:
            init_vec = np.array(self._params.iloc[2:15], dtype=float)
            x4_base = float(init_vec[4])
        except (IndexError, TypeError, Exception) as e:
            raise ValueError(f"Error getting initial baseline glucose (glucose_kinetics_2): {e}")
        x[4] = max(0.0, x4_base + self._noise("glucose_kinetics_2", x4_base))

        # --- Glucose kinetics primary state (depends on intestine, insulin action, x4) ---
        # Rat (mg/kg/min) from intestine:
        Rat = float(p.f) * float(p.kabs) * x[2] / float(p.BW) if p.BW != 0 else 0.0
        # Endogenous glucose production suppressed by insulin action (no self term -kp2*x3):
        EGP = max(0.0, float(p.kp1) - float(p.kp3) * x[8])
        # Renal excretion approx. via x4 vs ke2 threshold:
        Et = float(p.ke1) * max(0.0, (x[4] - float(p.ke2)))

        # Combine drivers (scalars chosen to map magnitudes; you can adjust α's below if you need tighter calibration)
        alpha_Rat = 1.0
        alpha_EGP = 1.0
        alpha_x4  = 0.05  # small coupling from x4 to x3 to retain influence without creating cycles

        x3_nom = alpha_Rat * Rat + alpha_EGP * EGP - Et + alpha_x4 * x[4] + float(p.Fsnc)  # include Fsnc as baseline use
        x3_nom = max(0.0, x3_nom)
        x[3] = max(0.0, x3_nom + self._noise("glucose_kinetics_1", x3_nom))

        # --- Subcutaneous glucose (steady algebraic: x12 ≈ x3) ---
        x12_nom = x[3]
        x[12] = max(0.0, x12_nom + self._noise("subcut_glucose", x12_nom))

        # Finalize (non-negative physiology)
        self.state = _clip_nonneg(x)
        self.state_hist.append(self.state.copy())

    # ----- Compatibility shims -----
    def reset(self):
        self.state = np.zeros(13, dtype=float)
        self.state_hist = []

    def step(self, action: Action, **kwargs):
        """Compatibility with env-like callers: generate one new i.i.d. sample."""
        self.sample_once(action)
