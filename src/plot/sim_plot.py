import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

dt = 0.002
Tend = 3.0

Nrk = round(Tend / dt)


if __name__ == "__main__":
    # exp data
    df = pd.read_csv("data/beki/output_te0.8_se45.csv")
    s_cyc = np.array(df["θ"])
    ds_cyc = np.array(df["dθ"])
    w1_cyc = np.array(df["w1"])
    w2_cyc = np.array(df["w2"])
    trq_cyc = np.array(df["trq"])

    # sim data
    df = pd.read_csv(
        "data/2020-11-04/楠1104実験用データ/te0.8se45/ガウス関数/6/0_3_output_pso_gauss_n6_te0.8_se45.csv"
    )
    s_sim = np.array(df["θ"])
    ds_sim = np.array(df["dθ"])
    w1_sim = np.array(df["w1"])
    w2_sim = np.array(df["w2"])
    trq_sim = np.array(df["trq"])

    t = np.linspace(0, Tend, Nrk + 1)

    # plot setting
    fig = plt.figure(figsize=(8, 12))
    gs = GridSpec(nrows=5, ncols=1)

    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["mathtext.default"] = "default"
    plt.rcParams["font.size"] = 15
    plt.rcParams["axes.grid"] = True

    for i, (sim, cyc, name, yx, axis) in enumerate(
        zip(
            [s_sim, ds_sim, w1_sim, w2_sim, trq_sim],
            [s_cyc, ds_cyc, w1_cyc, w2_cyc, trq_cyc],
            ["s", "ds", "w1", "w2", "trq"],
            [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
            [
                r"$\theta \rm{[rad]}$",
                r"$\dot{\theta} \rm{[rad / s]}$",
                r"$w_1(l) \rm{[cm]}$",
                r"$w_2(l) \rm{[cm]}$",
                r"$\tau \rm{[Nm]}$",
            ],
        )
    ):
        y, x = yx
        ax = fig.add_subplot(gs[y, x])

        # plot
        if name == "s":
            ax.plot(t[:1001], sim[:1001], label="Simulation")
            ax.plot(t[:1001], cyc[:1001], label="Cycloidal Motion")
            plt.legend(loc="lower right")
        else:
            ax.plot(t[:1001], sim[:1001])
            ax.plot(t[:1001], cyc[:1001])

        ax.set_ylabel(axis)

        if name == "trq":
            ax.set_xlabel(r"$t [s]$")
        else:
            ax.tick_params(
                labelbottom=False, labelleft=True, labelright=False, labeltop=False,
            )

    fig.patch.set_alpha(0)
    plt.tight_layout()

    savedir = "data/plot/sim/gauss_pso_08_45/"
    os.makedirs(savedir, exist_ok=True)
    fig.savefig(f"{savedir}plot.png")
