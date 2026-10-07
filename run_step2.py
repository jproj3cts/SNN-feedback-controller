"""Step 2: validate the continuous training loop with a non-spiking rate RNN.

Same task, same eye, same push/pull readout and same truncated-BPTT loop
(no state resets) the SNN will use. If this fails, the bug is in the
setup, not in the spiking neurons.
"""
import argparse
import dataclasses
import json

import numpy as np
import torch

from snnfc import config as C
from snnfc.baselines import LQR, LQGEye, PIDEye
from snnfc.eye import EyeDecoder
from snnfc.models import ModelController, RateRNN
from snnfc.plotting import plot_traces
from snnfc.sim import evaluate
from snnfc.train import TrainConfig, train

torch.set_num_threads(1)


def fit_u(trace, dt):
    """Least-squares u ~ a x + b v + c int(x) on a recorded trajectory (preview of step 4)."""
    x, v, u = (trace[k].flatten().numpy() for k in ("x", "v", "u"))
    ix = np.cumsum(trace["x"].numpy(), axis=1).flatten() * dt
    X = np.stack([x, v, ix, np.ones_like(x)], 1)
    coef, *_ = np.linalg.lstsq(X, u, rcond=None)
    r2 = 1 - np.mean((X @ coef - u) ** 2) / np.var(u)
    return {"kx": float(-coef[0]), "kv": float(-coef[1]), "ki": float(-coef[2]), "r2": float(r2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--eval-only", action="store_true", help="load results/step2_rnn.pt instead of training")
    args = ap.parse_args()

    p = C.plant_params()
    eye = C.make_eye(straight_through=True)
    cfg = TrainConfig(iters=args.iters, lam=C.LAM)
    model = RateRNN(eye.n, n_rec=128, n_out=16, dt=p.dt, in_scale=1 / (eye.r_max * p.dt))
    if args.eval_only:
        model.load_state_dict(torch.load(C.RESULTS / "step2_rnn.pt"))
    else:
        train(model, p, eye, cfg, log_path=C.RESULTS / "step2_log.csv",
              eval_kw=dict(seconds=C.TUNE_SECONDS, batch=C.TUNE_BATCH, seed=C.TUNE_SEED))
        torch.save(model.state_dict(), C.RESULTS / "step2_rnn.pt")

    step1 = json.loads((C.RESULTS / "step1.json").read_text())
    table = dict(step1["controllers"])
    r = evaluate(ModelController(model), p, eye, C.EVAL_SECONDS, C.EVAL_BATCH, C.EVAL_SEED, lam=C.LAM, record=8)
    tr = r.pop("trace")
    table["RNN (eye, learned)"] = r

    # traces for the comparison plot (same disturbance seed for all)
    traces = {"RNN (eye, learned)": tr}
    for kind, Ctor in (("PID (eye)", PIDEye), ("LQG (eye)", LQGEye)):
        dec = EyeDecoder.from_state_dict(torch.load(C.RESULTS / f"decoder_{kind.split()[0].lower()}.pt"))
        d = next(d for d in step1["decoders"] if d["tau"] == step1["controllers"][kind]["decoder_tau"])
        if kind == "PID (eye)":
            c = PIDEye(d["pid"]["kp"], d["pid"]["ki"], d["pid"]["kd"], d["pid"]["tau_d"], dec, eye.n, p.dt)
        else:
            c = LQGEye(p, C.LAM, dec, eye.n, d["lqg"]["meas_var"], d["lqg"]["r_scale"], d["lqg"]["q_scale"])
        traces[kind] = evaluate(c, p, eye, C.EVAL_SECONDS, C.EVAL_BATCH, C.EVAL_SEED, lam=C.LAM, record=1)["trace"]
    traces["LQR (true state)"] = evaluate(LQR(p, C.LAM), p, eye, C.EVAL_SECONDS, C.EVAL_BATCH,
                                          C.EVAL_SEED, lam=C.LAM, record=1)["trace"]
    plot_traces(traces, C.RESULTS / "step2_traces.png")

    lqr_cost = table["LQR (true state)"]["cost"]
    print("\nfinal evaluation (60 s x 64 plants, held-out disturbance seed)")
    print(f"{'controller':22s} {'cost':>9s} {'x LQR':>7s} {'mse':>9s} {'u^2':>7s}")
    for k, v in table.items():
        print(f"{k:22s} {v['cost']:9.5f} {v['cost']/lqr_cost:7.2f} {v['mse']:9.5f} {v['u2']:7.3f}")
    fit = fit_u(tr, p.dt)
    print(f"\npreview: u ~ -({fit['kx']:.2f} x + {fit['kv']:.2f} v + {fit['ki']:.3f} int x), R^2 = {fit['r2']:.3f}")
    (C.RESULTS / "step2.json").write_text(json.dumps(
        {"table": table, "u_fit": fit, "train_config": dataclasses.asdict(cfg)}, indent=2))
    plot_learning_curve([C.RESULTS / "step2_log.csv"], C.RESULTS / "step2_learning_curve.png")


def plot_learning_curve(log_paths, out_path, labels=None):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    step1 = json.loads((C.RESULTS / "step1.json").read_text())["controllers"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for n, path in enumerate(log_paths):
        rows = list(csv.DictReader(open(path)))
        it = [int(r["iter"]) for r in rows]
        lab = labels[n] if labels else ""
        line, = ax.semilogy(it, [float(r["loss"]) for r in rows], lw=0.6, alpha=0.4,
                            label=f"training window loss {lab}".strip())
        ev = [(int(r["iter"]), float(r["eval_cost"])) for r in rows if r.get("eval_cost")]
        ax.semilogy(*zip(*ev), "o-", ms=4, color=line.get_color(), label=f"eval cost {lab}".strip())
    for name, ls in (("LQR (true state)", "--"), ("PID (eye)", ":"), ("LQG (eye)", "-.")):
        ax.axhline(step1[name]["cost"], color="k", ls=ls, lw=1, label=name)
    ax.set_xlabel("training window (2 s each, never reset)")
    ax.set_ylabel("cost  <x² + λu²>")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
