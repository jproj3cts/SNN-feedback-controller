"""Classical reference controllers.

Privileged (see true x, v):
    Zero, LQR, PID
Fair (see only the eye, through the fitted EyeDecoder):
    PIDEye, LQGEye
"""
import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt
import torch

from .plant import PlantParams


def lqr_gain(p: PlantParams, lam: float) -> np.ndarray:
    """Continuous LQR for loss x^2 + lam u^2 (ignores saturation)."""
    A = np.array([[0.0, 1.0], [-p.k / p.m, -p.c / p.m]])
    B = np.array([[0.0], [1.0 / p.m]])
    Q = np.diag([1.0, 0.0])
    R = np.array([[lam]])
    P = sla.solve_continuous_are(A, B, Q, R)
    return (np.linalg.solve(R, B.T @ P)).ravel()  # u = -K [x, v]


class Zero:
    def reset(self, batch):
        pass

    def act(self, obs):
        return torch.zeros_like(obs["x"])


class LQR:
    def __init__(self, p: PlantParams, lam: float):
        self.K = torch.tensor(lqr_gain(p, lam), dtype=torch.float32)

    def reset(self, batch):
        pass

    def act(self, obs):
        return -(self.K[0] * obs["x"] + self.K[1] * obs["v"])


class PID:
    """PID on the true state (derivative = true velocity)."""

    def __init__(self, kp, ki, kd, dt):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt

    def reset(self, batch):
        self.I = torch.zeros(batch)

    def act(self, obs):
        self.I = self.I + self.dt * obs["x"]
        return -(self.kp * obs["x"] + self.ki * self.I + self.kd * obs["v"])


class PIDEye:
    """PID on decoded eye position, with a first-order filtered derivative."""

    def __init__(self, kp, ki, kd, tau_d, decoder, n_eye, dt):
        self.kp, self.ki, self.kd, self.tau_d = kp, ki, kd, tau_d
        self.dec, self.n_eye, self.dt = decoder, n_eye, dt

    def reset(self, batch):
        self.dec.reset(batch, self.n_eye)
        self.I = torch.zeros(batch)
        self.z = None

    def act(self, obs):
        y = self.dec(obs["spikes"])
        if self.z is None:
            self.z = y.clone()
        d = (y - self.z) / self.tau_d
        self.z = self.z + self.dt * d
        self.I = self.I + self.dt * y
        return -(self.kp * y + self.ki * self.I + self.kd * d)


class LQGEye:
    """LQR + steady-state Kalman filter on the decoded eye position.

    The decoder's exponential filter is modelled as an extra lag state
    xf' = (x - xf) / tau, so the measurement is y = xf + noise.
    """

    def __init__(self, p: PlantParams, lam: float, decoder, n_eye, meas_var: float,
                 r_scale: float = 1.0, q_scale: float = 1.0):
        self.p, self.dec, self.n_eye = p, decoder, n_eye
        dt, tau = p.dt, decoder.tau
        A = np.array([[0.0, 1.0, 0.0],
                      [-p.k / p.m, -p.c / p.m, 0.0],
                      [1.0 / tau, 0.0, -1.0 / tau]])
        B = np.array([[0.0], [1.0 / p.m], [0.0]])
        M = sla.expm(np.block([[A, B], [np.zeros((1, 4))]]) * dt)
        Ad, Bd = M[:3, :3], M[:3, 3:]
        C = np.array([[0.0, 0.0, 1.0]])
        Qw = np.diag([1e-9, q_scale * p.kick_rate * dt * p.kick_std ** 2, 1e-9])
        Rm = np.array([[r_scale * meas_var]])
        P = sla.solve_discrete_are(Ad.T, C.T, Qw, Rm)
        L = P @ C.T @ np.linalg.inv(C @ P @ C.T + Rm)
        f32 = lambda a: torch.tensor(a, dtype=torch.float32)
        self.Ad, self.Bd, self.C, self.L = f32(Ad), f32(Bd).ravel(), f32(C).ravel(), f32(L).ravel()
        self.K = f32(lqr_gain(p, lam))

    def reset(self, batch):
        self.dec.reset(batch, self.n_eye)
        self.s = torch.zeros(batch, 3)
        self.u_prev = torch.zeros(batch)

    def act(self, obs):
        y = self.dec(obs["spikes"])
        # predict with the force actually applied last step, then update
        self.s = self.s @ self.Ad.T + self.u_prev[:, None] * self.Bd[None, :]
        innov = y - self.s @ self.C
        self.s = self.s + innov[:, None] * self.L[None, :]
        cmd = -(self.K[0] * self.s[:, 0] + self.K[1] * self.s[:, 1])
        self.u_prev = self.p.u_max * torch.tanh(cmd / self.p.u_max)
        return cmd


def tune(make, x0, cost_fn, maxiter=120):
    """Nelder-Mead over log-parameters. Returns (best params, best cost)."""
    def f(logx):
        c = cost_fn(make(np.exp(logx)))
        return c if np.isfinite(c) else 1e6
    # explicit initial simplex: scipy's default perturbs a log-param of 0
    # (i.e. x0 = 1) by only 2.5e-4, which leaves it effectively frozen
    l0 = np.log(np.asarray(x0, dtype=float))
    simplex = np.vstack([l0] + [l0 + 0.7 * np.eye(len(l0))[i] for i in range(len(l0))])
    res = sopt.minimize(f, l0, method="Nelder-Mead",
                        options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-5,
                                 "initial_simplex": simplex})
    return np.exp(res.x), res.fun
