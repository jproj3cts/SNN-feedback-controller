"""Continuous truncated-BPTT training on control performance.

The plant and the network run without ever being reset: each training
window continues from the exact state the previous one ended in, and the
gradient is simply cut (detached) at window boundaries. Only plants that
leave the working range (|x| > x_reset) are re-initialised.

The loss is <e^2 + lam u^2> over the window. Gradients flow through the
plant dynamics and, via the eye's straight-through estimator, around the
whole closed loop.
"""
from dataclasses import dataclass, asdict
import csv
import time

import torch

from .models import ModelController, detach_state
from .plant import Oscillator
from .sim import evaluate


@dataclass
class TrainConfig:
    iters: int = 400
    window: int = 1000          # truncation length in steps (2 s at dt=2 ms)
    batch: int = 64
    lr: float = 1e-3
    clip: float = 1.0
    lam: float = 1e-3
    eval_every: int = 25
    seed: int = 0


def train(model, p, eye, cfg: TrainConfig, log_path=None, eval_kw=None):
    torch.manual_seed(cfg.seed)
    gp = torch.Generator().manual_seed(cfg.seed + 100)
    ge = torch.Generator().manual_seed(cfg.seed + 200)
    plant = Oscillator(p, cfg.batch, gp)
    state = model.init_state(cfg.batch)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    eval_kw = eval_kw or {}
    rows = []
    t0 = time.time()
    for it in range(1, cfg.iters + 1):
        e2 = u2 = 0.0
        for _ in range(cfg.window):
            spikes = eye(plant.x, ge)
            state, cmd = model.step(state, spikes)
            u = plant.step(cmd)
            e2 = e2 + (plant.x ** 2).mean()
            u2 = u2 + (u ** 2).mean()
        e2, u2 = e2 / cfg.window, u2 / cfg.window
        loss = e2 + cfg.lam * u2
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip).item()
        opt.step()
        # cut the gradient, keep the state: the system runs on seamlessly
        plant.detach()
        state = detach_state(state)
        plant.handle_blowups()

        row = {"iter": it, "loss": loss.item(), "mse": e2.item(), "u2": u2.item(),
               "grad_norm": gnorm, "resets": plant.n_resets, "time_s": time.time() - t0}
        if it % cfg.eval_every == 0 or it == cfg.iters:
            with torch.no_grad():
                row["eval_cost"] = evaluate(ModelController(model), p, eye, lam=cfg.lam, **eval_kw)["cost"]
        rows.append(row)
        if it % 10 == 0 or "eval_cost" in row:
            msg = f"it {it:4d} loss {row['loss']:.5f} mse {row['mse']:.5f} |g| {gnorm:.3f} resets {plant.n_resets} t {row['time_s']:.0f}s"
            if "eval_cost" in row:
                msg += f"  eval {row['eval_cost']:.5f}"
            print(msg, flush=True)
    if log_path:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return rows


__all__ = ["TrainConfig", "train", "asdict"]
