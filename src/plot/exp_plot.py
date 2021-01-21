# import matplotlib as mpl
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

dt = 0.002
Tend = 3.0

Nrk = round(Tend / dt)


if __name__ == "__main__":
    # sim data
    df = pd.read_csv(
        "data/2020-11-04/楠1104実験用データ/te1.0se135/ガウス関数/6/0_8_output_pso_gauss_n6_te1.0_se135.csv"
    )
    s_sim = np.array(df["θ"])
    ds_sim = np.array(df["dθ"])
    w1_sim = np.array(df["w1"])
    w2_sim = np.array(df["w2"])
    trq_sim = np.array(df["trq"])

    # exp data
    df = pd.read_csv("data/2020-11-04/楠1104実験用データ/te1.0se135/ガウス関数/6/g_s135_t10.csv")
    s_exp = np.array(df['"Model Root"/"EX_S2"'])
    ds_exp = np.array(df['"Model Root"/"EX_V2"'])
    w1_exp = np.array(df['"Model Root"/"w1"']) * 2.7244 / np.rad2deg(1)
    w2_exp = np.array(df['"Model Root"/"w2"']) * 2.7244 / np.rad2deg(1)
    trq_exp = np.array(df['"Model Root"/"Trq"'])

    # cyc data
    df = pd.read_csv("data/2020-11-04/楠1104実験用データ/te1.0se135/c_s135_t10.csv")
    s_cyc = np.array(df['"Model Root"/"EX_S2"'])
    ds_cyc = np.array(df['"Model Root"/"EX_V2"'])
    w1_cyc = np.array(df['"Model Root"/"w1"']) * 2.7244 / np.rad2deg(1)
    w2_cyc = np.array(df['"Model Root"/"w2"']) * 2.7244 / np.rad2deg(1)
    trq_cyc = np.array(df['"Model Root"/"Trq"'])

    t = np.linspace(0, Tend, Nrk + 1)

    # plot setting
    fig = plt.figure(figsize=(8, 12))
    gs = GridSpec(nrows=5, ncols=1)

    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["mathtext.default"] = "default"
    plt.rcParams["font.size"] = 15

    for i, (exp, sim, cyc, name, yx, axis) in enumerate(
        zip(
            [s_exp, ds_exp, w1_exp, w2_exp, trq_exp],
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
        print(name)

        y, x = yx
        ax = fig.add_subplot(gs[y, x])

        # plot
        if name == "s":
            ax.plot(t[:1001], cyc[:1001], label="Cycloidal Motion")
            ax.plot(t[:1001], exp[:1001], label="Experiment")
            ax.plot(t[:1001], sim[:1001], label="Simulation")
            plt.legend(loc="lower right")
        else:
            ax.plot(t[:1001], cyc[:1001])
            ax.plot(t[:1001], exp[:1001])
            ax.plot(t[:1001], sim[:1001])

        ax.set_ylabel(axis)

        if name == "trq":
            ax.set_xlabel(r"$t [s]$")
        else:
            ax.tick_params(
                labelbottom=False, labelleft=True, labelright=False, labeltop=False,
            )

    fig.patch.set_alpha(0)
    plt.tight_layout()

    savedir = "data/plot/exp/10_135/"
    os.makedirs(savedir, exist_ok=True)
    fig.savefig(f"{savedir}plot.png")
