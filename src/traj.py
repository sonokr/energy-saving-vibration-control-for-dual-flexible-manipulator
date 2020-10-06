import numpy as np
from numba import njit

from cond import *


def cycloid(a, v):
    if v["mode"] == "power":
        return power(a)
    elif v["mode"] == "gauss_n2":
        return gauss_n2(a)
    elif v["mode"] == "gauss_n4":
        return gauss_n4(a)


def power(a):
    S = np.zeros((2 * Nrk + 1, 3))
    t = np.linspace(0.0, TE, 2 * Nte + 1)

    T = -1 + 2 * t / TE
    dT = 2 / TE

    # u = t / TE + (1 - T ** 2) ** 2 * sum([a[n] * T ** n for n in range(len(a))])
    # du = np.gradient(u, TE / (2 * Nte + 1))
    # ddu = np.gradient(du, TE / (2 * Nte + 1))

    u = t / TE + (1 - T ** 2) ** 2 * sum([a[n] * T ** n for n in range(len(a))])
    du = (
        (1 / TE)
        - 4 * T * (1 - T ** 2) * sum([a[n] * T ** n for n in range(len(a))]) * dT
        + (1 - T ** 2) ** 2 * sum([n * a[n] * dT * T ** (n - 1) for n in range(1, len(a))])
    )
    ddu = (
        2
        * (-4 * T * (1 - T ** 2) * dT)
        * sum([n * a[n] * dT * T ** (n - 1) for n in range(1, len(a))])
        + sum([a[n] * T ** n for n in range(len(a))]) * 4 * ((-1 + 3 * T ** 2) * dT ** 2)
        + (1 - T ** 2) ** 2
        * 2
        * sum([3 * n * a[n + 2] * T ** (n) for n in range(len(a) - 2)])
        * dT ** 2
    )

    S[: 2 * Nte + 1, 0] = SE * (u - np.sin(2 * np.pi * t / TE) / 2 / np.pi)
    S[: 2 * Nte + 1, 1] = SE * (du - np.cos(2 * np.pi * u) * du)
    S[: 2 * Nte + 1, 2] = SE * (
        ddu - np.cos(2 * np.pi * u) * ddu + 2 * np.pi * np.sin(2 * np.pi * u) * du ** 2
    )
    S[2 * Nte + 1 :, 0] = SE

    return S


@njit("f8[:,:](f8[:])")
def gauss_n2(a):
    """サイクロイド軌道
    """
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


def gauss_n4(a):
    pass
