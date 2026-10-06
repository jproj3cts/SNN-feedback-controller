"""Population-coded "eye": Gaussian tuning curves tiling position.

Each neuron fires Poisson spikes with rate
    r_i(x) = r_base + r_max * exp(-(x - c_i)^2 / (2 * sigma^2)).
Position is the only external information the controller receives.

Spikes use a straight-through estimator: the forward pass is a true
Bernoulli sample, the backward pass uses d(rate)/dx, so BPTT can follow the
closed loop plant -> eye -> network -> plant.
"""
import torch


class GaussianEye:
    def __init__(self, n: int = 16, x_min: float = -1.2, x_max: float = 1.2,
                 width: float | None = None, r_max: float = 100.0, r_base: float = 1.0,
                 dt: float = 0.002, straight_through: bool = True):
        self.n = n
        self.centers = torch.linspace(x_min, x_max, n)
        spacing = (x_max - x_min) / (n - 1)
        self.sigma = width if width is not None else spacing
        self.r_max = r_max
        self.r_base = r_base
        self.dt = dt
        self.straight_through = straight_through

    def rates(self, x: torch.Tensor) -> torch.Tensor:
        """Firing rates in Hz, shape (B, n)."""
        d = x[:, None] - self.centers[None, :]
        return self.r_base + self.r_max * torch.exp(-0.5 * (d / self.sigma) ** 2)

    def __call__(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        p = (self.rates(x) * self.dt).clamp(max=1.0)
        s = torch.bernoulli(p.detach(), generator=generator)
        if self.straight_through:
            return p + (s - p).detach()
        return s


class EyeDecoder:
    """Exponentially filtered eye spikes -> linear readout of position.

    Used to give the classical baselines the same sensor as the network.
    """

    def __init__(self, w: torch.Tensor, b: float, tau: float, dt: float):
        self.w, self.b, self.tau, self.dt = w, b, tau, dt
        self.f = None

    def reset(self, batch: int, n: int):
        self.f = torch.zeros(batch, n)

    def __call__(self, spikes: torch.Tensor) -> torch.Tensor:
        self.f = self.f + (self.dt / self.tau) * (spikes.detach() - self.f)
        return self.f @ self.w + self.b

    def state_dict(self):
        return {"w": self.w, "b": self.b, "tau": self.tau, "dt": self.dt}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["w"], d["b"], d["tau"], d["dt"])
