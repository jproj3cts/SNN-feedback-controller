"""Step 0: plant + privileged baselines (true x, v). No neural networks.

Establishes the scale of the task: uncontrolled cost (worst), LQR (optimal
for the linear-quadratic problem with full state) and a tuned PID.
"""
import json

import torch

from snnfc import config as C
from snnfc.baselines import LQR, PID, Zero, lqr_gain, tune
from snnfc.plotting import plot_traces
from snnfc.sim import evaluate

torch.set_num_threads(4)


def main():
    p = C.plant_params()
    eye = C.make_eye()
    C.RESULTS.mkdir(exist_ok=True)

    def tune_cost(ctrl):
        return evaluate(ctrl, p, eye, C.TUNE_SECONDS, C.TUNE_BATCH, C.TUNE_SEED, lam=C.LAM)["cost"]

    K = lqr_gain(p, C.LAM)
    print(f"LQR gain K = {K}")
    print("tuning PID on true state ...")
    (kp, ki, kd), _ = tune(lambda x: PID(*x, p.dt), [K[0] + 1.0, 1.0, K[1]], tune_cost)
    print(f"PID kp={kp:.3f} ki={ki:.4f} kd={kd:.3f}")

    ctrls = {
        "zero": Zero(),
        "LQR (true state)": LQR(p, C.LAM),
        "PID (true state)": PID(kp, ki, kd, p.dt),
    }
    results, traces = {}, {}
    for name, c in ctrls.items():
        r = evaluate(c, p, eye, C.EVAL_SECONDS, C.EVAL_BATCH, C.EVAL_SEED, lam=C.LAM, record=1)
        traces[name] = r.pop("trace")
        results[name] = r
        print(f"{name:20s} cost={r['cost']:.5f} mse={r['mse']:.5f} u2={r['u2']:.3f} resets={r['resets']}")

    results["_params"] = {"lqr_K": K.tolist(), "pid": {"kp": kp, "ki": ki, "kd": kd}}
    (C.RESULTS / "step0.json").write_text(json.dumps(results, indent=2))
    plot_traces(traces, C.RESULTS / "step0_traces.png")


if __name__ == "__main__":
    main()
