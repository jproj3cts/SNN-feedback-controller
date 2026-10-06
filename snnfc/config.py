"""Shared experiment settings so every step uses the same task."""
from pathlib import Path

from .plant import PlantParams
from .eye import GaussianEye

LAM = 1e-3                 # control-effort weight in <e^2 + lam u^2>
EVAL_SECONDS = 60.0
EVAL_BATCH = 64
EVAL_SEED = 1234           # final evaluation
TUNE_SECONDS = 20.0
TUNE_BATCH = 32
TUNE_SEED = 99             # baseline tuning (kept separate from EVAL_SEED)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def plant_params() -> PlantParams:
    return PlantParams()


def make_eye(straight_through: bool = True) -> GaussianEye:
    p = plant_params()
    return GaussianEye(n=16, x_min=-1.2, x_max=1.2, r_max=100.0, r_base=1.0,
                       dt=p.dt, straight_through=straight_through)
