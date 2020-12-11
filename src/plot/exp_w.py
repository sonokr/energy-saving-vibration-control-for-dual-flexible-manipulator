import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dt = 0.002
Tend = 3.0

Nrk = round(Tend / dt)


if __name__ == "__main__":
    # exp data
    df = pd.read_csv("data/2020-11-04/楠1104実験用データ/te0.8se45/ガウス関数/6/g_s45_t08.csv")
    w1_exp = np.array(df['"Model Root"/"w1"'])
    w2_exp = np.array(df['"Model Root"/"w2"'])
    trq_exp = np.array(df['"Model Root"/"Trq"']) - 0.04

    # sim data
    df = pd.read_csv(
        "data/2020-11-04/楠1104実験用データ/te0.8se45/ガウス関数/6/0_3_output_pso_gauss_n6_te0.8_se45.csv"
    )
    w1_sim = np.array(df["w1"]) * np.rad2deg(1) / 2.7244
    w2_sim = np.array(df["w2"]) * np.rad2deg(1) / 2.7244
    trq_sim = np.array(df["trq"])

    t = np.linspace(0, Tend, Nrk + 1)

    # plot setting
    savedir = "data/plot/exp/"

    mpl.rcParams["figure.figsize"] = [6.0, 3.0]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["mathtext.default"] = "default"

    for exp, sim, name, axis in zip(
        [w1_exp, w2_exp, trq_exp],
        [w1_sim, w2_sim, trq_sim],
        ["w1", "w2", "trq"],
        [r"\theta [deg]", r"\theta [deg]", r"\tau [J]"],
    ):
        print(name)

        fig, ax = plt.subplots(1)

        fig.patch.set_alpha(0)
        fig.subplots_adjust(bottom=0.2)

        # plot
        ax.plot(t, exp[:1501], label="Experiment")
        ax.plot(t, sim, label="Simulation")

        ax.set_ylabel(axis)
        ax.set_xlabel(r"$t [s]$")

        plt.legend()
        fig.savefig(f"{savedir}{name}.png", dpi=600)
