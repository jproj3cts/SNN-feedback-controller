import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_traces(traces: dict, path, seconds: float = 12.0, which: int = 0):
    """traces: {label: evaluate(..., record=k)['trace']} sharing a disturbance seed."""
    fig, (ax_x, ax_u) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for label, tr in traces.items():
        t = tr["t"]
        m = t <= seconds
        ax_x.plot(t[m], tr["x"][which][m], label=label, lw=1.2)
        ax_u.plot(t[m], tr["u"][which][m], label=label, lw=1.0)
    ax_x.axhline(0, color="k", lw=0.5)
    ax_x.set_ylabel("position x")
    ax_u.set_ylabel("force u")
    ax_u.set_xlabel("time (s)")
    ax_x.legend(loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
