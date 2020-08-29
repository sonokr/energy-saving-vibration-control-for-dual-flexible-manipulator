import argparse
import csv
import time

import numpy as np
import pandas as pd
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from numba import njit
from scipy import integrate

from condition import *


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


def cycloid(a):
    """サイクロイド軌道
    """
    S = np.zeros((2 * Nrk + 1, 3))
    t = np.linspace(0.0, TE, 2 * Nte + 1)

    T = -1 + 2 * t / TE
    u = t / TE + (1 - T ** 2) ** 2 * sum([a[n] * T ** n for n in range(len(a))])
    du = np.gradient(u, TE / (2 * Nte + 1))
    ddu = np.gradient(du, TE / (2 * Nte + 1))

    S[: 2 * Nte + 1, 0] = SE * (u - np.sin(2 * np.pi * t / TE) / 2 / np.pi)
    S[: 2 * Nte + 1, 1] = SE * (du - np.cos(2 * np.pi * u) * du)
    S[: 2 * Nte + 1, 2] = SE * (
        ddu - np.cos(2 * np.pi * u) * ddu + 2 * np.pi * np.sin(2 * np.pi * u) * du ** 2
    )
    S[2 * Nte + 1 :, 0] = SE

    return S


def torque(S, ddW1, ddW2):
    """トルクを計算する
    """
    return g1 * S[0 : 2 * Nrk + 1 : 2, 2] + g2 * ddW1 + g3 * ddW2 + cs * S[0 : 2 * Nrk + 1 : 2, 1]


def find_inflection_point(x):
    """変曲点を見つける
    """
    inflection_point = [
        0,
    ]
    for i in range(1, len(x) - 1):
        if (x[i] >= x[i - 1] and x[i] >= x[i + 1]) or (x[i] <= x[i - 1] and x[i] <= x[i + 1]):
            inflection_point.append(i + 1)
    if inflection_point[-1] != len(x) - 1:
        inflection_point.append(len(x) - 1)
    return inflection_point


def energy(f, x):
    """消費エネルギーを計算する
    """
    p = find_inflection_point(x)
    ene = 0
    for i in range(1, len(p)):
        ene = ene + abs(integrate.simps(np.abs(f[p[i - 1] : p[i]]), x[p[i - 1] : p[i]]))
    return ene


def run(a):
    S = cycloid(a)

    X1, X2 = RK4(S)
    w1 = X1[0, :] * 2.7244
    w2 = X2[0, :] * 2.7244

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
    trq = torque(S, ddW1, ddW2)

    return pd.DataFrame(
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


if __name__ == "__main__":
    start = time.time()

    #################
    # Get Arguments #
    #################
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--PARAM_FILE_PATH", help="PATH of parameter csv file of power series"
    )
    args = parser.parse_args()

    #############################
    # Parameter of Power Series #
    #############################
    a = [0.0]
    if (param_path := args.PARAM_FILE_PATH) :
        with open(param_path) as file:
            reader = csv.reader(file)
            a = [float(row[0]) for row in reader]
    print(f"Param: {a}")

    ###########
    # Execute #
    ###########
    df = run(a)
    df.to_csv("./data/dst/output.csv")

    ################
    # Plot Setting #
    ################
    output_file("./data/plot/graph.html")

    width, height = 350, 250
    fig1 = figure(width=width, plot_height=height, title="θ")
    fig2 = figure(width=width, plot_height=height, title="dθ")
    fig3 = figure(width=width, plot_height=height, title="ddθ")
    fig4 = figure(width=width, plot_height=height, title="trq")
    fig5 = figure(width=width, plot_height=height, title="w1")
    fig6 = figure(width=width, plot_height=height, title="w2")

    fig1.line(df["t"], df["θ"])
    fig2.line(df["t"], df["dθ"])
    fig3.line(df["t"], df["ddθ"])
    fig4.line(df["t"], df["trq"])
    fig5.line(df["t"], df["w1"])
    fig6.line(df["t"], df["w2"])

    fig = gridplot([[fig1, fig4], [fig2, fig5], [fig3, fig6],])
    show(fig)

    print(f"Elapsed time: {time.time()-start}")
