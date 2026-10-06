"""Batched, differentiable harmonic-oscillator plant.

    m * x'' = u + F - k * x - c * x' + kicks

The commanded force is passed through a smooth actuator limit
u = u_max * tanh(command / u_max) so gradients never vanish at saturation.
Random velocity kicks (Poisson in time) keep the system excited so a
continuously running controller always has something to do.
"""
from dataclasses import dataclass
import math

import torch


@dataclass
class PlantParams:
    m: float = 1.0
    k: float = (2 * math.pi / 2.0) ** 2  # natural period 2 s
    c: float = 0.0                        # viscous damping
    F: float = 0.0                        # constant unknown force
    u_max: float = 5.0
    kick_rate: float = 0.5                # kicks per second per plant
    kick_std: float = 1.0                 # std of velocity jump (m/s)
    x_reset: float = 3.0                  # plants beyond this are re-initialised
    dt: float = 0.002

    @property
    def omega0(self) -> float:
        return math.sqrt(self.k / self.m)


class Oscillator:
    def __init__(self, p: PlantParams, batch: int, generator: torch.Generator | None = None):
        self.p = p
        self.batch = batch
        self.gen = generator
        self.n_resets = 0
        self.x = torch.zeros(batch)
        self.v = torch.zeros(batch)
        self.reset()

    def _randn(self, n):
        return torch.randn(n, generator=self.gen)

    def reset(self):
        self.x = 0.3 * self._randn(self.batch)
        self.v = 0.5 * self._randn(self.batch)

    def step(self, command: torch.Tensor) -> torch.Tensor:
        """Advance one dt (semi-implicit Euler). Returns the applied force u."""
        p = self.p
        u = p.u_max * torch.tanh(command / p.u_max)
        a = (u + p.F - p.k * self.x - p.c * self.v) / p.m
        v = self.v + p.dt * a
        x = self.x + p.dt * v
        kick = torch.rand(self.batch, generator=self.gen) < p.kick_rate * p.dt
        v = v + kick.float() * p.kick_std * self._randn(self.batch)
        self.x, self.v = x, v
        return u

    def detach(self):
        self.x = self.x.detach()
        self.v = self.v.detach()

    def handle_blowups(self):
        """Re-initialise plants that left the working range (call between windows)."""
        bad = self.x.detach().abs() > self.p.x_reset
        if bad.any():
            self.n_resets += int(bad.sum())
            self.x = torch.where(bad, 0.3 * self._randn(self.batch), self.x)
            self.v = torch.where(bad, 0.5 * self._randn(self.batch), self.v)
