"""Closed-loop simulation and evaluation shared by all controllers.

A controller implements
    reset(batch)                      -> None
    act(obs: dict[str, Tensor])       -> command (B,)
where obs contains 'x', 'v' (true state, for privileged baselines only)
and 'spikes' (eye output, the only thing the learned controllers see).
"""
import torch

from .plant import Oscillator, PlantParams
from .eye import GaussianEye


@torch.no_grad()
def evaluate(controller, p: PlantParams, eye: GaussianEye, seconds: float = 60.0,
             batch: int = 64, seed: int = 1234, warmup: float = 2.0, lam: float = 1e-3,
             record: int = 0):
    """Run the closed loop and return mean cost <e^2 + lam u^2> after warm-up.

    Plant kicks and eye spikes use separate seeded generators, so every
    controller faces the same disturbance sequence (common random numbers).
    """
    g_plant = torch.Generator().manual_seed(seed)
    g_eye = torch.Generator().manual_seed(seed + 1)
    plant = Oscillator(p, batch, g_plant)
    controller.reset(batch)
    n_steps = int(seconds / p.dt)
    n_warm = int(warmup / p.dt)
    e2 = torch.zeros(batch)
    u2 = torch.zeros(batch)
    rec = {"t": [], "x": [], "v": [], "u": []} if record else None
    for i in range(n_steps):
        spikes = eye(plant.x, g_eye)
        cmd = controller.act({"x": plant.x, "v": plant.v, "spikes": spikes})
        u = plant.step(cmd)
        if (i + 1) % 500 == 0:
            plant.handle_blowups()
        if i >= n_warm:
            e2 += plant.x ** 2
            u2 += u ** 2
        if record:
            rec["t"].append(i * p.dt)
            rec["x"].append(plant.x[:record].clone())
            rec["v"].append(plant.v[:record].clone())
            rec["u"].append(u[:record].clone())
    n = n_steps - n_warm
    mse = (e2 / n).mean().item()
    mu2 = (u2 / n).mean().item()
    out = {"mse": mse, "u2": mu2, "cost": mse + lam * mu2, "resets": plant.n_resets}
    if record:
        out["trace"] = {k: (torch.tensor(v) if k == "t" else torch.stack(v, 1)) for k, v in rec.items()}
    return out
