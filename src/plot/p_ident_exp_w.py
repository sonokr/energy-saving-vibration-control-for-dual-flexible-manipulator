import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit

dt = 0.002
Tend = 3.0

TE = 0.8
SE = np.pi * (2 / 4)

Nrk = round(Tend / dt)
Nte = round(TE / dt)


@njit("f8[:](f8, f8, f8, f8, f8[:])")
def f(x11, x12, ds, dds, a_):
    """運動方程式
    """
    # リンク1
    ome = 12.68 * 0.993 * np.float64(a_[0])
    z = 15.49 * 10 ** (-3) * 1.214 * np.float64(a_[1])
    a = 2.266 * 10 ** (-1) * 0.984 * np.float64(a_[2])
    b = 2.004 * 10 ** (-3) * 1.020 * np.float64(a_[3])

    # リンク2
    # ome = 10.71 * np.float64(a_[0])
    # z = 14.27 * 10.0 ** -3 * np.float64(a_[1])
    # a = 2.570 * 10.0 ** -1 * np.float64(a_[2])
    # b = 5.822 * 10.0 ** -3 * np.float64(a_[3])

    dx11 = x12
    dx12 = -(2 * z * ome * x12 + ome ** 2 * x11 + a * dds + b * x11 * ds ** 2)
    k1 = np.array([dx11, dx12]) * dt
    return k1


@njit("f8[:,:](f8[:,:], f8[:])")
def RK4(S, a):
    """ルンゲクッタ法
    """
    k1 = np.empty((2, 4), np.float64)
    X1 = np.zeros((2, Nrk + 1))

    for i in range(0, Nrk):
        x11, x12 = X1[:, i]

        k1[:, 0] = f(x11, x12, S[2 * i, 1], S[2 * i, 2], a)
        k1[:, 1] = f(x11 + k1[0, 0] / 2, x12 + k1[1, 0] / 2, S[2 * i + 1, 1], S[2 * i + 1, 2], a)
        k1[:, 2] = f(x11 + k1[0, 1] / 2, x12 + k1[1, 1] / 2, S[2 * i + 1, 1], S[2 * i + 1, 2], a)
        k1[:, 3] = f(x11 + k1[0, 2], x12 + k1[1, 2], S[2 * i + 2, 1], S[2 * i + 2, 2], a)

        X1[:, i + 1] = X1[:, i] + ((k1[:, 0] + 2 * k1[:, 1] + 2 * k1[:, 2] + k1[:, 3]) / 6.0)

    return X1


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


if __name__ == "__main__":
    v = {
        "target": "w1",
    }
    v["datadir"] = "./data/plot/p_ident_exp/"
    ###################
    # experience data #
    ###################
    df = pd.read_csv("./data/p_ident_exp/exp9_8.csv")
    if v["target"] == "w1":
        w1_exp = np.array(df["先端の変位[1mm]"])
    elif v["target"] == "w2":
        w1_exp = np.array(df["先端の変位[0.8mm]"])
    else:
        raise Exception("")

    ##############
    # simulation #
    ##############
    S = cycloid()

    X1 = RK4(S, np.array([0.993, 1.214, 0.984, 1.020]),)
    w1 = X1[0, :] * 2.7244

    ###############
    # deta output #
    ###############
    df = pd.DataFrame(
        {
            "t": np.linspace(0, Tend, Nrk + 1),
            "θ": S[0 : 2 * Nrk + 1 : 2, 0],
            "dθ": S[0 : 2 * Nrk + 1 : 2, 1],
            "ddθ": S[0 : 2 * Nrk + 1 : 2, 2],
            "w1": w1,
        }
    )

    ################
    # plot setting #
    ################
    mpl.rcParams["figure.figsize"] = [6.0, 3.0]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["mathtext.default"] = "default"

    fig, ax = plt.subplots(1)

    ax.plot(df["t"], w1_exp, label="Experiment")
    ax.plot(df["t"], df["w1"], label="Cycloidal motion")

    ax.set_ylabel(r"$\theta$ [rad]")
    ax.set_xlabel(r"$t [s]$")

    fig.patch.set_alpha(0)

    plt.legend()
    # plt.show()
    fig.savefig(v["datadir"] + v["target"] + ".png", dpi=600)
