from datetime import datetime
import torch
from torch import nn
import numpy as np

# Static SCM patient (no env/sensor needed)
from utils.t1dpatient_static_scm import T1DPatientStatic, Action

class Simglucose(nn.Module):
    """
    Teacher wrapper for the static feed-forward SCM (S3-Glucose).
    Matches your teacher API but returns a single-snapshot state vector and CGM-like output.
    """
    def __init__(self, pred_horizon, timeseries_iit=False):
        super().__init__()
        self.model = self.simulator
        self.loss = nn.MSELoss(reduction="mean")
        self.start_time = datetime(2024, 2, 14, 8, 0, 0, 0)  # unused but kept for parity
        self.pred_horizon = pred_horizon
        self.timeseries_iit = timeseries_iit
        self.time_env_sample = 3  # kept to mimic your scaling of meal_size

    @torch.no_grad()
    def simulator(
        self,
        pat_name: str,
        meal_size: float,
        insulin_dosage: float,
        pred_horizon: int,
        interchanged_variables=None,
        variable_names=None,
        interchanged_activations=None
    ):
        # Instantiate static SCM patient
        pat_name = pat_name.replace("_hyper", "").replace("_hypo", "")
        patient = T1DPatientStatic.withName(name=pat_name)

        # Emulate your scaling of meal_size by time_env_sample
        # We treat 'meal_size' as grams delivered in this minute.
        meal_gpm = float(meal_size) * float(self.time_env_sample)

        # Insulin is provided as U/min (already in your controller)
        insulin_Upm = float(insulin_dosage)

        # Generate one i.i.d. sample
        patient.reset()
        patient.step(Action(CHO=meal_gpm, insulin=insulin_Upm))

        # Hidden states: the 13-dim vector (single snapshot)
        x = np.array(patient.state_hist)  # shape (1, 13)
        # Output CGM-like scalar (mg/dL-scale)
        Gsub = patient.observation.Gsub

        return x, Gsub

    def forward(
        self,
        input_ids,
        labels=None,
        look_up=None,
        interchanged_variables=None,
        variable_names=None,
        interchanged_activations=None
    ):
        """
        Inputs:
          input_ids: pre-meal parameters (your pipeline)
          labels:    post meal parameters (unused here)
          look_up:   patient name
        """
        teacher_outputs = {}
        # Your convention: last elements carry insulin_dosage and meal_size
        meal_size = float(input_ids[-11])
        insulin_dosage = float(input_ids[-12])

        x, output = self.simulator(
            look_up,
            meal_size,
            insulin_dosage,
            self.pred_horizon,
            variable_names=variable_names,
            interchanged_variables=interchanged_variables,
            interchanged_activations=interchanged_activations
        )

        # hidden_states: (13,) like your original "last state" usage
        teacher_outputs["hidden_states"] = np.transpose(x)[ :, -1]  # (13,)

        # outputs: keep your scaling by 0.01 to stay compatible downstream
        teacher_outputs["outputs"] = torch.tensor(output * 0.01, dtype=torch.float32)

        return teacher_outputs
