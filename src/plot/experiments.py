# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

dt = 0.002
Tend = 3.0

Nrk = round(Tend / dt)


if __name__ == "__main__":
    df = pd.read_csv("data/2020-12-14/data_TE09_S90.csv")

    # sim data
    s_sim = np.array(df["SIM_S"])
    ds_sim = np.array(df["SIM_DS"])
    w1_sim = np.array(df["SIM_W1"]) * 100
    w2_sim = np.array(df["SIM_W2"]) * 100
    trq_sim = np.array(df["SIM_τ"])

    # exp data
    s_exp = np.array(df["EX_S"])
    ds_exp = np.array(df["EX_DS"])
    w1_exp = np.array(df["EX_W1"]) * 100
    w2_exp = np.array(df["EX_W2"]) * 100
    trq_exp = np.array(df["EX_τ"])

    # cyc data
    s_cyc = np.array(df["CYC_S"])
    ds_cyc = np.array(df["CYC_DS"])
    w1_cyc = np.array(df["CYC_W1"]) * 100
    w2_cyc = np.array(df["CYC_W2"]) * 100
    trq_cyc = np.array(df["CYC_τ"])

    t = np.linspace(0, Tend, Nrk + 1)

    # plot setting
    savedir = "data/plot/exp/"

    fig = plt.figure(figsize=(10, 12))
    gs = GridSpec(nrows=3, ncols=2)

    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["mathtext.default"] = "default"
    plt.rcParams["font.size"] = 15

    for i, (exp, sim, cyc, name, axis) in enumerate(
        zip(
            [s_exp, ds_exp, w1_exp, w2_exp, trq_exp],
            [s_sim, ds_sim, w1_sim, w2_sim, trq_sim],
            [s_cyc, ds_cyc, w1_cyc, w2_cyc, trq_cyc],
            ["s", "ds", "w1", "w2", "trq"],
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

        y = i + 1 if i < 2 else i - 2
        x = 0 if i < 2 else 1
        ax = fig.add_subplot(gs[y, x])

        # plot
        if name == "s":
            ax.plot(t[:1001], cyc[:1001], label="Cycloidal Motion")
            ax.plot(t[:1001], exp[:1001], label="Experiment")
            ax.plot(t[:1001], sim[:1001], label="Simulation")
            plt.legend()
        else:
            ax.plot(t[:1001], cyc[:1001])
            ax.plot(t[:1001], exp[:1001])
            ax.plot(t[:1001], sim[:1001])

        ax.set_ylabel(axis)
        ax.set_xlabel(r"$t [s]$")

    fig.patch.set_alpha(0)
    plt.tight_layout()

    fig.savefig(f"{savedir}plot.png")
