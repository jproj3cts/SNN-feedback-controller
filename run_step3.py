"""Step 3: swap the rate RNN for a recurrent spiking (LIF) network.

Everything else is unchanged from step 2: same eye, same push/pull
readout, same continuous truncated-BPTT loop, same baselines. Gradients
through spikes use a fast-sigmoid surrogate.

    python run_step3.py --seed 0          # train one network (run several seeds in parallel)
    python run_step3.py --report 0 1 2    # evaluate trained seeds, write table + figures
"""
import argparse
import dataclasses
import json

import numpy as np
import torch

from snnfc import config as C
from snnfc.baselines import LQGEye
from snnfc.eye import EyeDecoder
from snnfc.models import LIFSNN, ModelController, RateRNN
from snnfc.plant import Oscillator
from snnfc.plotting import plot_traces
from snnfc.sim import evaluate
from snnfc.train import TrainConfig, train
from run_step2 import fit_u, plot_learning_curve

torch.set_num_threads(1)


def ckpt(seed):
    return C.RESULTS / f"step3_snn_s{seed}.pt"


def log(seed):
    return C.RESULTS / f"step3_log_s{seed}.csv"


def make_snn(eye, p, seed):
    return LIFSNN(eye.n, n_rec=128, n_out=16, dt=p.dt, seed=seed)


def run_train(seed, iters):
    p = C.plant_params()
    eye = C.make_eye(straight_through=True)
    model = make_snn(eye, p, seed)
    cfg = TrainConfig(iters=iters, lam=C.LAM, seed=seed)
    train(model, p, eye, cfg, log_path=log(seed),
          eval_kw=dict(seconds=C.TUNE_SECONDS, batch=C.TUNE_BATCH, seed=C.TUNE_SEED))
    torch.save(model.state_dict(), ckpt(seed))


@torch.no_grad()
def record_spikes(model, p, eye, seconds=6.0, seed=C.EVAL_SEED, batch=8):
    """Run the closed loop and keep every spike of plant 0 (for rasters / rate stats)."""
    gp = torch.Generator().manual_seed(seed)
    ge = torch.Generator().manual_seed(seed + 1)
    plant = Oscillator(p, batch, gp)
    state = model.init_state(batch)
    keep = {"eye": [], "rec": [], "out": [], "x": [], "u": []}
    for _ in range(int(seconds / p.dt)):
        s = eye(plant.x, ge)
        state, cmd = model.step(state, s)
        u = plant.step(cmd)
        keep["eye"].append(s[0]); keep["rec"].append(state["rec"]); keep["out"].append(state["out"])
        keep["x"].append(plant.x[0]); keep["u"].append(u[0])
    return {k: torch.stack(v) for k, v in keep.items()}


def plot_raster(rec, p, path, t0=1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = rec["x"].shape[0]
    t = np.arange(T) * p.dt
    m = t >= t0
    fig, axs = plt.subplots(5, 1, figsize=(10, 9), sharex=True,
                            gridspec_kw={"height_ratios": [1, 1.2, 2.5, 1.2, 1]})
    axs[0].plot(t[m], rec["x"][m], color="C0"); axs[0].axhline(0, color="k", lw=0.5)
    axs[0].set_ylabel("x")
    for ax, spikes, lab in ((axs[1], rec["eye"], "eye (16)"), (axs[2], rec["rec"][:, 0], "recurrent (128)")):
        tt, nn_ = torch.nonzero(spikes, as_tuple=True)
        keep = t[tt.numpy()] >= t0
        ax.scatter(t[tt.numpy()][keep], nn_.numpy()[keep], s=0.6, color="k")
        ax.set_ylabel(lab)
    out = rec["out"][:, 0]
    n_out = out.shape[1] // 2
    tt, nn_ = torch.nonzero(out, as_tuple=True)
    keep = t[tt.numpy()] >= t0
    col = np.where(nn_.numpy() < n_out, "C3", "C2")
    axs[3].scatter(t[tt.numpy()][keep], nn_.numpy()[keep], s=1.0, c=col[keep])
    axs[3].set_ylabel("out: push (red)\npull (green)", fontsize=8)
    axs[4].plot(t[m], rec["u"][m], color="C1"); axs[4].set_ylabel("u")
    axs[4].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def report(seeds):
    torch.set_num_threads(4)
    p = C.plant_params()
    eye = C.make_eye(straight_through=True)
    step2 = json.loads((C.RESULTS / "step2.json").read_text())
    table = dict(step2["table"])
    lqr_cost = table["LQR (true state)"]["cost"]

    per_seed, models = {}, {}
    for s in seeds:
        m = make_snn(eye, p, s)
        m.load_state_dict(torch.load(ckpt(s)))
        r = evaluate(ModelController(m), p, eye, C.EVAL_SECONDS, C.EVAL_BATCH, C.EVAL_SEED, lam=C.LAM, record=8)
        per_seed[s], models[s] = r, m
        print(f"SNN seed {s}: cost {r['cost']:.5f} ({r['cost']/lqr_cost:.2f}x LQR) resets {r['resets']}")
    best = min(per_seed, key=lambda s: per_seed[s]["cost"])
    costs = np.array([per_seed[s]["cost"] for s in seeds])
    for s in seeds:
        tr = per_seed[s].pop("trace")
        if s == best:
            best_trace = tr
    table["SNN (eye, learned, best seed)"] = per_seed[best]

    # firing statistics + raster from the best network
    rec = record_spikes(models[best], p, eye)
    warm = int(1.0 / p.dt)
    rates = {
        "eye_hz": rec["eye"][warm:].mean().item() / p.dt,
        "rec_hz": rec["rec"][warm:].mean().item() / p.dt,
        "out_hz": rec["out"][warm:].mean().item() / p.dt,
        "rec_silent_frac": (rec["rec"][warm:].mean((0, 1)) == 0).float().mean().item(),
        "out_silent_frac": (rec["out"][warm:].mean((0, 1)) == 0).float().mean().item(),
    }
    plot_raster(rec, p, C.RESULTS / "step3_raster.png")

    # trajectories vs the RNN from step 2 and the eye-only LQG
    rnn = RateRNN(eye.n, n_rec=128, n_out=16, dt=p.dt, in_scale=1 / (eye.r_max * p.dt))
    rnn.load_state_dict(torch.load(C.RESULTS / "step2_rnn.pt"))
    traces = {"SNN (eye, learned)": best_trace}
    traces["RNN (eye, learned)"] = evaluate(ModelController(rnn), p, eye, C.EVAL_SECONDS, C.EVAL_BATCH,
                                            C.EVAL_SEED, lam=C.LAM, record=1)["trace"]
    step1 = json.loads((C.RESULTS / "step1.json").read_text())
    d = next(d for d in step1["decoders"] if d["tau"] == step1["controllers"]["LQG (eye)"]["decoder_tau"])
    dec = EyeDecoder.from_state_dict(torch.load(C.RESULTS / "decoder_lqg.pt"))
    lqg = LQGEye(p, C.LAM, dec, eye.n, d["lqg"]["meas_var"], d["lqg"]["r_scale"], d["lqg"]["q_scale"])
    traces["LQG (eye)"] = evaluate(lqg, p, eye, C.EVAL_SECONDS, C.EVAL_BATCH, C.EVAL_SEED,
                                   lam=C.LAM, record=1)["trace"]
    plot_traces(traces, C.RESULTS / "step3_traces.png")
    plot_learning_curve([log(s) for s in seeds], C.RESULTS / "step3_learning_curve.png",
                        labels=[f"seed {s}" for s in seeds])

    fit = fit_u(best_trace, p.dt)
    print("\nfinal evaluation (60 s x 64 plants, held-out disturbance seed)")
    print(f"{'controller':32s} {'cost':>9s} {'x LQR':>7s} {'mse':>9s} {'u^2':>7s}")
    for k, v in table.items():
        print(f"{k:32s} {v['cost']:9.5f} {v['cost']/lqr_cost:7.2f} {v['mse']:9.5f} {v['u2']:7.3f}")
    print(f"SNN across seeds: mean {costs.mean():.5f} ({costs.mean()/lqr_cost:.2f}x LQR), "
          f"range {costs.min():.5f}-{costs.max():.5f}")
    print(f"firing: eye {rates['eye_hz']:.1f} Hz, recurrent {rates['rec_hz']:.1f} Hz "
          f"(silent {rates['rec_silent_frac']:.2f}), output {rates['out_hz']:.1f} Hz (silent {rates['out_silent_frac']:.2f})")
    print(f"preview: u ~ -({fit['kx']:.2f} x + {fit['kv']:.2f} v + {fit['ki']:.3f} int x), R^2 = {fit['r2']:.3f}")
    (C.RESULTS / "step3.json").write_text(json.dumps({
        "table": table, "best_seed": best,
        "per_seed_cost": {str(s): per_seed[s]["cost"] for s in seeds},
        "firing": rates, "u_fit": fit,
        "train_config": dataclasses.asdict(TrainConfig(lam=C.LAM)),
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--report", type=int, nargs="+", metavar="SEED")
    args = ap.parse_args()
    if args.report:
        report(args.report)
    else:
        run_train(args.seed, args.iters)


if __name__ == "__main__":
    main()
