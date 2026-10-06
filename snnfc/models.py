"""Learned controllers.

Every model sees only eye spikes and drives two separate output
populations (push / pull). Their activity is exponentially filtered (a
differentiable rolling window) and the difference becomes the force command:

    command = gain * (filt(mean push activity) - filt(mean pull activity))

Step 2 uses a non-spiking leaky rate RNN with this exact interface, so the
training loop can be validated before swapping in spiking neurons.
"""
import torch
import torch.nn as nn


class PushPullReadout(nn.Module):
    def __init__(self, tau_out: float, dt: float, gain: float):
        super().__init__()
        self.alpha = dt / tau_out
        self.gain = gain

    def forward(self, filt, push, pull):
        act = torch.stack([push.mean(1), pull.mean(1)], 1)        # (B, 2)
        filt = filt + self.alpha * (act - filt)
        return filt, self.gain * (filt[:, 0] - filt[:, 1])


class RateRNN(nn.Module):
    """Leaky rate network: random sparse recurrence + push/pull output pops.

    tau dv/dt = -v + W_in s + W_rec tanh(v) + b      (recurrent pop)
    tau do/dt = -o + W_out tanh(v) + b_out           (output pops)
    output activity = sigmoid(o)  in [0, 1], like a spike probability.
    """

    def __init__(self, n_in: int, n_rec: int = 128, n_out: int = 16, tau: float = 0.02,
                 tau_out: float = 0.05, dt: float = 0.002, p_conn: float = 0.2,
                 in_scale: float = 5.0, gain: float = 10.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n_rec, self.n_out = n_rec, n_out
        self.alpha = dt / tau
        self.in_scale = in_scale
        self.W_in = nn.Parameter(torch.randn(n_rec, n_in, generator=g) / n_in ** 0.5)
        mask = (torch.rand(n_rec, n_rec, generator=g) < p_conn).float()
        mask.fill_diagonal_(0)
        self.register_buffer("mask", mask)
        self.W_rec = nn.Parameter(0.9 * torch.randn(n_rec, n_rec, generator=g) / (p_conn * n_rec) ** 0.5)
        self.b = nn.Parameter(torch.zeros(n_rec))
        self.W_out = nn.Parameter(torch.randn(2 * n_out, n_rec, generator=g) / n_rec ** 0.5)
        self.b_out = nn.Parameter(torch.zeros(2 * n_out))
        self.readout = PushPullReadout(tau_out, dt, gain)

    def init_state(self, batch: int):
        return {"v": torch.zeros(batch, self.n_rec),
                "o": torch.zeros(batch, 2 * self.n_out),
                "filt": torch.full((batch, 2), 0.5)}

    def step(self, state, spikes):
        r = torch.tanh(state["v"])
        drive = self.in_scale * spikes @ self.W_in.T + r @ (self.W_rec * self.mask).T + self.b
        v = state["v"] + self.alpha * (drive - state["v"])
        o = state["o"] + self.alpha * (torch.tanh(v) @ self.W_out.T + self.b_out - state["o"])
        a = torch.sigmoid(o)
        filt, cmd = self.readout(state["filt"], a[:, :self.n_out], a[:, self.n_out:])
        return {"v": v, "o": o, "filt": filt, "rec": r, "out": a}, cmd


def detach_state(state):
    return {k: v.detach() for k, v in state.items()}


class ModelController:
    """Adapter so a trained model can be run by sim.evaluate()."""

    def __init__(self, model):
        self.model = model

    def reset(self, batch):
        self.state = self.model.init_state(batch)

    def act(self, obs):
        self.state, cmd = self.model.step(self.state, obs["spikes"])
        return cmd
