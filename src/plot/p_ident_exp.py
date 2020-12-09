import csv
import os

import bokeh
import numpy as np
import pandas as pd
from bokeh.io import export_png
from bokeh.plotting import figure
from numba import njit

dt = 0.002
Tend = 3.0

TE = 0.8
SE = np.pi * (2 / 4)

Nrk = round(Tend / dt)
Nte = round(TE / dt)

with open("./data/p_ident_exp/w1/param/0_param.csv") as file:
    reader = csv.reader(file)
    a1_ = np.array([float(row[0]) for row in reader])
ome1 = 12.68 * np.float64(a1_[0])
z1 = 15.49 * 10.0 ** -3 * np.float64(a1_[1])
a1 = 2.266 * 10.0 ** -1 * np.float64(a1_[2])
b1 = 2.004 * 10.0 ** -3 * np.float64(a1_[3])

with open("./data/p_ident_exp/w2/param/0_param.csv") as file:
    reader = csv.reader(file)
    a2_ = np.array([float(row[0]) for row in reader])
ome2 = 10.71 * np.float64(a2_[0])
z2 = 14.27 * 10.0 ** -3 * np.float64(a2_[1])
a2 = 2.570 * 10.0 ** -1 * np.float64(a2_[2])
b2 = 5.822 * 10.0 ** -3 * np.float64(a2_[3])


@njit("Tuple((f8[:], f8[:]))(f8, f8, f8, f8, f8, f8)")
def f(x11, x12, x21, x22, ds, dds):
    """運動方程式
    """
    dx11 = x12
    dx12 = -(2 * z1 * ome1 * x12 + ome1 ** 2 * x11 + a1 * dds + b1 * x11 * ds ** 2)
    dx21 = x22
    dx22 = -(2 * z2 * ome2 * x22 + ome2 ** 2 * x21 + a2 * dds + b2 * x21 * ds ** 2)

    k1 = np.array([dx11, dx12]) * dt
    k2 = np.array([dx21, dx22]) * dt

    return k1, k2


@njit("Tuple((f8[:,:], f8[:,:]))(f8[:,:])")
def RK4(S):
    """ルンゲクッタ法
    """
    k1 = np.empty((2, 4), np.float64)
    k2 = np.empty((2, 4), np.float64)
    X1 = np.zeros((2, Nrk + 1))
    X2 = np.zeros((2, Nrk + 1))

    for i in range(0, Nrk):
        x11, x12 = X1[:, i]
        x21, x22 = X2[:, i]

        k1[:, 0], k2[:, 0] = f(x11, x12, x21, x22, S[2 * i, 1], S[2 * i, 2])
        k1[:, 1], k2[:, 1] = f(
            x11 + k1[0, 0] / 2,
            x12 + k1[1, 0] / 2,
            x21 + k2[0, 0] / 2,
            x22 + k2[1, 0] / 2,
            S[2 * i + 1, 1],
            S[2 * i + 1, 2],
        )
        k1[:, 2], k2[:, 2] = f(
            x11 + k1[0, 1] / 2,
            x12 + k1[1, 1] / 2,
            x21 + k2[0, 1] / 2,
            x22 + k2[1, 1] / 2,
            S[2 * i + 1, 1],
            S[2 * i + 1, 2],
        )
        k1[:, 3], k2[:, 3] = f(
            x11 + k1[0, 2],
            x12 + k1[1, 2],
            x21 + k2[0, 2],
            x22 + k2[1, 2],
            S[2 * i + 2, 1],
            S[2 * i + 2, 2],
        )

        X1[:, i + 1] = X1[:, i] + ((k1[:, 0] + 2 * k1[:, 1] + 2 * k1[:, 2] + k1[:, 3]) / 6.0)
        X2[:, i + 1] = X2[:, i] + ((k2[:, 0] + 2 * k2[:, 1] + 2 * k2[:, 2] + k2[:, 3]) / 6.0)

    return X1, X2


@njit("f8[:,:]()")
def cycloid():
    """サイクロイド軌道
    """
    a = np.array([0, 0, 0, 0])
    c = np.array([1.0, -1.0])
    sig = 10.0 ** (-a[2:4])
    t = np.linspace(0.0, TE, 2 * Nte + 1)
    T = -1 + 2 * t / TE
    abe1 = np.abs((1 - T ** 2) / np.exp((-c[0] + T) ** 2 / sig[0]))
    abe2 = np.abs((1 - T ** 2) / np.exp((-c[1] + T) ** 2 / sig[1]))
    W = np.array([a[0] / abe1.max(), a[1] / abe2.max()])

    S = np.zeros((2 * Nrk + 1, 3))
    for i in range(0, 2 * Nte + 1):
        net = t[i] / TE
        dnet = 1.0 / TE
        ddnet = 0.0
        for j in range(0, 2):
            net = net + (W[j] * (1 - T[i] ** 2)) / np.exp((-c[j] + T[i]) ** 2 / sig[j])
            dnet = dnet + (-4.0 * W[j] * (-c[j] + T[i] * (1 + sig[j] + (c[j] - T[i]) * T[i]))) / (
                np.exp((c[j] - T[i]) ** 2 / sig[j]) * sig[j] * TE
            )
            ddnet = ddnet + (
                -8.0
                * W[j]
                * (
                    -2.0 * c[j] ** 2
                    + sig[j]
                    + sig[j] ** 2
                    + T[i]
                    * (
                        4.0 * c[j] * (1 + sig[j])
                        + T[i] * (-2 + 2 * c[j] ** 2 - 5 * sig[j] + 2 * T[i] * (-2 * c[j] + T[i]))
                    )
                )
            ) / (np.exp((c[j] - T[i]) ** 2 / sig[j]) * sig[j] ** 2 * TE ** 2)
        S[i, 0] = SE * (net - np.sin(2 * np.pi * net) / 2 / np.pi)
        S[i, 1] = 2 * SE * np.sin(np.pi * net) ** 2 * dnet
        S[i, 2] = (
            2
            * SE
            * np.sin(np.pi * net)
            * (2 * np.pi * np.cos(np.pi * net) * dnet ** 2 + np.sin(np.pi * net) * ddnet)
        )
    S[2 * Nte + 1 :, 0] = SE

    return S


def torque(S, X1, X2, a):
    """トルクを計算する
    """
    g1 = 2.996 * 10 ** -2 * a[0]
    g2 = 6.652 * 10 ** -2 * a[1]
    g3 = 5.237 * 10 ** -2 * a[2]
    cs = 4.202 * 10 ** -2 * a[3]

    ddW1 = -(
        2 * z1 * ome1 * X1[1, :]
        + ome1 ** 2 * X1[0, :]
        + a1 * S[0 : 2 * Nrk + 1 : 2, 2]
        + b1 * X1[0, :] * S[0 : 2 * Nrk + 1 : 2, 1] ** 2
    )
    ddW2 = -(
        2 * z2 * ome2 * X2[1, :]
        + ome2 ** 2 * X2[0, :]
        + a2 * S[0 : 2 * Nrk + 1 : 2, 2]
        + b2 * X2[0, :] * S[0 : 2 * Nrk + 1 : 2, 1] ** 2
    )
    return g1 * S[0 : 2 * Nrk + 1 : 2, 2] + g2 * ddW1 + g3 * ddW2 + cs * S[0 : 2 * Nrk + 1 : 2, 1]


if __name__ == "__main__":
    v = {
        "target": "trq",
    }
    v["datadir"] = "./data/plot/p_ident_exp/"
    os.makedirs(v["datadir"], exist_ok=True)

    ###################
    # experience data #
    ###################
    df = pd.read_csv("./data/p_ident_exp/exp9_8.csv")
    trq_exp = np.array(df["トルク[ノイズカット]"])

    ################
    # plot setting #
    ################
    # output_file(v["datadir"] + "plot.png")
    width, height = 350 * 3, 250 * 3
    fig = figure(width=width, plot_height=height, title="トルク")
    palette = bokeh.palettes.Category10[10]

    ##############
    # simulation #
    ##############
    S = cycloid()

    X1, X2 = RK4(S)
    w1 = X1[0, :] * 2.7244
    w2 = X2[0, :] * 2.7244

    trq = torque(S, X1, X2, [1.016, 1.018, 1.052, 1.131])

    ###############
    # deta output #
    ###############
    df = pd.DataFrame(
        {
            "t": np.linspace(0, Tend, Nrk + 1),
            "θ": S[0 : 2 * Nrk + 1 : 2, 0],
            "dθ": S[0 : 2 * Nrk + 1 : 2, 1],
            "ddθ": S[0 : 2 * Nrk + 1 : 2, 2],
            "trq": trq,
            "w1": w1,
            "w2": w2,
        }
    )

    ################
    # plot setting #
    ################
    fig.line(df["t"], df["trq"], line_color=palette[0])
    fig.line(df["t"], trq_exp, line_color=palette[1])

    # show(fig)
    export_png(fig, filename=v["datadir"] + "tr1q.png")
