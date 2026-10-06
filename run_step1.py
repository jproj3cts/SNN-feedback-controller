"""Step 1: check the eye in isolation, then re-run baselines through it.

1. Simulate a lightly damped, kicked oscillator so x covers the working
   range (x_std ~0.4, almost always inside the eye's +-1.2 coverage),
   record eye spikes and decode position with a population-vector readout
   of exponentially filtered spikes, for several filter time constants.
   Report error, lag and noise.
2. Tune PID and LQG controllers that only see the decoded eye position.
   This is the fair comparison for the learned controllers.
"""
import dataclasses
import json

import numpy as np
import torch

from snnfc import config as C
from snnfc.baselines import LQGEye, PIDEye, lqr_gain, tune
from snnfc.eye import EyeDecoder
from snnfc.plant import Oscillator
from snnfc.plotting import plot_traces
from snnfc.sim import evaluate

torch.set_num_threads(4)
TAUS = [0.010, 0.020, 0.040]


def record_eye(p, eye, seconds, batch, seed):
    gp = torch.Generator().manual_seed(seed)
    ge = torch.Generator().manual_seed(seed + 1)
    plant = Oscillator(p, batch, gp)
    xs, ss = [], []
    for _ in range(int(seconds / p.dt)):
        ss.append(eye(plant.x, ge))
        xs.append(plant.x.clone())
        plant.step(torch.zeros(batch))
    return torch.stack(xs), torch.stack(ss)  # (T, B), (T, B, n)


def exp_filter(sig, tau, dt):
    out = torch.zeros_like(sig)
    f = torch.zeros_like(sig[0])
    for t in range(sig.shape[0]):
        f = f + (dt / tau) * (sig[t] - f)
        out[t] = f
    return out


def best_lag(x, xhat, dt, max_lag=0.2):
    """Lag (s) maximising correlation between x(t - lag) and xhat(t)."""
    best, best_c = 0, -1
    for L in range(int(max_lag / dt)):
        a = x[: x.shape[0] - L].flatten()
        b = xhat[L:].flatten()
        c = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        if c > best_c:
            best, best_c = L, c
    return best * dt


def fit_decoder(p, eye, tau):
    """Build the population-vector decoder for this tau and measure it."""
    warm = int(1.0 / p.dt)
    x_te, s_te = record_eye(p, eye, 30.0, 32, seed=8)
    f_te = exp_filter(s_te, tau, p.dt)[warm:]
    xf_te = exp_filter(x_te, tau, p.dt)[warm:]
    x_te = x_te[warm:]
    xhat = (f_te * eye.centers).sum(-1) / f_te.sum(-1).clamp(min=1e-6)
    near = x_te.abs() < 0.5
    stats = {
        "tau": tau,
        "rmse_vs_x": (xhat - x_te).pow(2).mean().sqrt().item(),
        "rmse_vs_x_near_target": (xhat - x_te)[near].pow(2).mean().sqrt().item(),
        "rmse_vs_lowpassed_x": (xhat - xf_te).pow(2).mean().sqrt().item(),
        "lag_s": best_lag(x_te, xhat, p.dt),
        "x_std": x_te.std().item(),
        "r2": 1 - (xhat - x_te).pow(2).mean().item() / x_te.var().item(),
    }
    return EyeDecoder(eye.centers, tau, p.dt), stats


def main():
    p = C.plant_params()
    eye = C.make_eye()
    # excitation for decoder evaluation: some damping -> x_rms ~0.4
    p_fit = dataclasses.replace(p, c=0.2, kick_std=1.0)
    K = lqr_gain(p, C.LAM)
    step0 = json.loads((C.RESULTS / "step0.json").read_text())
    pid0 = step0["_params"]["pid"]

    def tune_cost(ctrl):
        return evaluate(ctrl, p, eye, C.TUNE_SECONDS, C.TUNE_BATCH, C.TUNE_SEED, lam=C.LAM)["cost"]

    def eval_full(ctrl, record=0):
        return evaluate(ctrl, p, eye, C.EVAL_SECONDS, C.EVAL_BATCH, C.EVAL_SEED, lam=C.LAM, record=record)

    out = {"eye": {"n": eye.n, "r_max": eye.r_max, "sigma": eye.sigma}, "decoders": [], "controllers": {}}
    best = {}
    for tau in TAUS:
        dec, st = fit_decoder(p_fit, eye, tau)
        print(f"tau={tau*1e3:.0f}ms  rmse(x)={st['rmse_vs_x']:.4f}  rmse(|x|<0.5)={st['rmse_vs_x_near_target']:.4f}  rmse(lowpass x)={st['rmse_vs_lowpassed_x']:.4f}"
              f"  lag={st['lag_s']*1e3:.0f}ms  R2={st['r2']:.4f}")
        meas_var = st["rmse_vs_lowpassed_x"] ** 2

        (kp, ki, kd, td), _ = tune(lambda x: PIDEye(*x, dec, eye.n, p.dt),
                                   [pid0["kp"], max(pid0["ki"], 0.1), pid0["kd"], 0.02], tune_cost)
        pid = PIDEye(kp, ki, kd, td, dec, eye.n, p.dt)
        r_pid = eval_full(pid)
        (rs, qs), _ = tune(lambda x: LQGEye(p, C.LAM, dec, eye.n, meas_var, *x), [1.0, 1.0], tune_cost, maxiter=60)
        lqg = LQGEye(p, C.LAM, dec, eye.n, meas_var, rs, qs)
        r_lqg = eval_full(lqg)
        print(f"   PID-eye cost={r_pid['cost']:.5f} (kp={kp:.1f} ki={ki:.3f} kd={kd:.2f} tau_d={td*1e3:.1f}ms)"
              f"   LQG-eye cost={r_lqg['cost']:.5f} (r_scale={rs:.3g} q_scale={qs:.3g})")
        st.update(pid=dict(kp=kp, ki=ki, kd=kd, tau_d=td, cost=r_pid["cost"]),
                  lqg=dict(r_scale=rs, q_scale=qs, meas_var=meas_var, cost=r_lqg["cost"]))
        out["decoders"].append(st)
        for kind, r, c in (("PID (eye)", r_pid, pid), ("LQG (eye)", r_lqg, lqg)):
            if kind not in best or r["cost"] < best[kind][0]["cost"]:
                best[kind] = (r, c, dec, tau)

    traces = {}
    for name in ("zero", "LQR (true state)"):
        out["controllers"][name] = step0[name]
    for kind, (r, c, dec, tau) in best.items():
        rr = eval_full(c, record=1)
        traces[kind] = rr.pop("trace")
        rr["decoder_tau"] = tau
        out["controllers"][kind] = rr
        torch.save(dec.state_dict(), C.RESULTS / f"decoder_{kind.split()[0].lower()}.pt")
    from snnfc.baselines import LQR
    traces["LQR (true state)"] = eval_full(LQR(p, C.LAM), record=1)["trace"]
    plot_traces(traces, C.RESULTS / "step1_traces.png")
    (C.RESULTS / "step1.json").write_text(json.dumps(out, indent=2))
    for k, v in out["controllers"].items():
        print(f"{k:20s} cost={v['cost']:.5f}")


if __name__ == "__main__":
    main()
