import time

import numpy as np
import pandas as pd
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.plotting import figure

# 条件値（いずれ設定ファイルにまとめる）
dt = 0.002
Tend = 3.0

TE = 0.8  # 駆動時間
SE = np.pi / 2  # 目標角

Nrk = round(Tend / dt)
Nte = round(TE / dt)

# リンク1のパラメータ
l1p = {
    "ome": 12.68,
    "z": 15.49 * 10 ** (-3),
    "a": 2.266 * 10 ** (-1),
    "b": 2.004 * 10 ** (-3),
}

# リンク2のパラメータ
l2p = {
    "ome": 10.71,
    "z": 14.27 * 10 ** (-3),
    "a": 2.570 * 10 ** (-1),
    "b": 5.8 * 10 ** (-3),
}

# トルク関数のパラメータ
g1 = 2.996 * 10 ** (-2)
g2 = 6.652 * 10 ** (-2)
g3 = 5.237 * 10 ** (-2)
cs = 4.202 * 10 ** (-2)


def f(x1, x2, p, ds, dds):
    """運動方程式
    """
    dx1 = x2
    dx2 = -(
        2 * p["z"] * p["ome"] * x2 + (p["ome"] ** 2) * x1 + p["a"] * dds + p["b"] * x1 * (ds ** 2)
    )
    return np.array([dx1, dx2]) * dt


def RK4(S):
    """ルンゲクッタ法
    """
    k1 = np.empty([2, 4])
    k2 = np.empty([2, 4])
    X1 = np.zeros([2, Nrk + 1])
    X2 = np.zeros([2, Nrk + 1])

    for i in range(0, Nrk):
        x1, x2 = X1[:, i]
        x3, x4 = X2[:, i]

        k1[:, 0] = f(x1, x2, l1p, S[2 * i, 1], S[2 * i, 2])
        k2[:, 0] = f(x3, x4, l2p, S[2 * i, 1], S[2 * i, 2])

        k1[:, 1] = f(x1 + k1[0, 0] / 2, x2 + k1[1, 0] / 2, l1p, S[2 * i + 1, 1], S[2 * i + 1, 2])
        k2[:, 1] = f(x3 + k2[0, 0] / 2, x4 + k2[1, 0] / 2, l2p, S[2 * i + 1, 1], S[2 * i + 1, 2])

        k1[:, 2] = f(x1 + k1[0, 1] / 2, x2 + k1[1, 1] / 2, l1p, S[2 * i + 1, 1], S[2 * i + 1, 2])
        k2[:, 2] = f(x3 + k2[0, 1] / 2, x4 + k2[1, 1] / 2, l2p, S[2 * i + 1, 1], S[2 * i + 1, 2])

        k1[:, 3] = f(x1 + k1[0, 2], x2 + k1[1, 2], l1p, S[2 * i + 2, 1], S[2 * i + 2, 2])
        k2[:, 3] = f(x3 + k2[0, 2], x4 + k2[1, 2], l2p, S[2 * i + 2, 1], S[2 * i + 2, 2])

        X1[:, i + 1] = X1[:, i] + ((k1[:, 0] + 2 * k1[:, 1] + 2 * k1[:, 2] + k1[:, 3]) / 6.0)
        X2[:, i + 1] = X2[:, i] + ((k2[:, 0] + 2 * k2[:, 1] + 2 * k2[:, 2] + k2[:, 3]) / 6.0)

    return X1, X2


def cyc():
    """サイクロイド軌道
    """
    S = np.zeros([2 * Nrk + 1, 3])
    t = np.linspace(0.0, TE, 2 * Nte + 1)

    S[: 2 * Nte + 1, 0] = SE * (t / TE - np.sin(2 * np.pi * t / TE) / 2 / np.pi)
    S[: 2 * Nte + 1, 1] = np.gradient(S[: 2 * Nte + 1, 0], TE / (2 * Nte + 1))
    S[: 2 * Nte + 1, 2] = np.gradient(S[: 2 * Nte + 1, 1], TE / (2 * Nte + 1))
    S[2 * Nte + 1 :, 0] = SE

    return S


if __name__ == "__main__":
    start = time.time()

    S = cyc()
    X1, X2 = RK4(S)

    w1 = X1[0, :]
    w2 = X2[0, :]

    ddW1 = -(
        2 * l1p["z"] * l1p["ome"] * X1[1, :]
        + l1p["ome"] ** 2 * X1[0, :]
        + l1p["a"] * S[0 : 2 * Nrk + 1 : 2, 2]
        + l1p["b"] * X1[0, :] * S[0 : 2 * Nrk + 1 : 2, 1] ** 2
    )
    ddW2 = -(
        2 * l2p["z"] * l2p["ome"] * X2[1, :]
        + l2p["ome"] ** 2 * X2[0, :]
        + l2p["a"] * S[0 : 2 * Nrk + 1 : 2, 2]
        + l2p["b"] * X2[0, :] * S[0 : 2 * Nrk + 1 : 2, 1] ** 2
    )
    trq = g1 * S[0 : 2 * Nrk + 1 : 2, 2] + g2 * ddW1 + g3 * ddW2 + cs * S[0 : 2 * Nrk + 1 : 2, 1]

    print(f"Elapsed time: {time.time()-start}")

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
    df.to_csv("./data/dst/output.csv")

    # Plot Setting
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
