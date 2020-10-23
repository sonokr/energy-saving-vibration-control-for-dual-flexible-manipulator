import numpy as np
from numba import njit

from cond import *


def cycloid(a, v):
    if v["mode"] == "power":
        return power(a)
    elif v["mode"] == "gauss_n4":
        return gauss_n4(a)
    elif v["mode"] == "gauss_n6":
        return gauss_n6(a)


@njit("f8[:,:](f8[:])")
def power(a):
    a = np.array(list(a) + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    S = np.zeros((2 * Nrk + 1, 3))
    t = np.linspace(0.0, TE, 2 * Nte + 1)
    dX = 2.0 / TE
    for i in range(0, 2 * Nte + 1):
        X = -1 + 2 * t[i] / TE
        y = t[i] / TE + (1 - X ** 2) ** 2 * (
            a[0]
            + a[1] * X
            + a[2] * X ** 2
            + a[3] * X ** 3
            + a[4] * X ** 4
            + a[5] * X ** 5
            + a[6] * X ** 6
            + a[7] * X ** 7
            + a[8] * X ** 8
            + a[9] * X ** 9
            + a[10] * X ** 10
        )
        #
        dy = 1 / TE - (
            (1 - X ** 2)
            * (
                -a[1]
                + X
                * (
                    4 * a[0]
                    - 2 * a[2]
                    + X
                    * (
                        5 * a[1]
                        - 3 * a[3]
                        + X
                        * (
                            6 * a[2]
                            - 4 * a[4]
                            + X
                            * (
                                7 * a[3]
                                - 5 * a[5]
                                + X
                                * (
                                    8 * a[4]
                                    - 6 * a[6]
                                    + X
                                    * (
                                        9 * a[5]
                                        - 7 * a[7]
                                        + X
                                        * (
                                            10 * a[6]
                                            - 8 * a[8]
                                            + X
                                            * (
                                                11 * a[7]
                                                - 9 * a[9]
                                                + X
                                                * (
                                                    -10 * a[10]
                                                    + 12 * a[8]
                                                    + 13 * a[9] * X
                                                    + 14 * a[10] * X ** 2
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            * dX
        )
        #
        ddy = (
            2
            * (
                -2 * a[0]
                + a[2]
                + X
                * (
                    -6 * a[1]
                    + 3 * a[3]
                    + X
                    * (
                        6 * (a[0] - 2 * a[2] + a[4])
                        + X
                        * (
                            10 * (a[1] - 2 * a[3] + a[5])
                            + 15 * (a[2] - 2 * a[4] + a[6]) * X
                            + 21 * (a[3] - 2 * a[5] + a[7]) * X ** 2
                            + 28 * (a[4] - 2 * a[6] + a[8]) * X ** 3
                            + 36 * (a[5] - 2 * a[7] + a[9]) * X ** 4
                            + 45 * (a[10] + a[6] - 2 * a[8]) * X ** 5
                            + 55 * (a[7] - 2 * a[9]) * X ** 6
                            + 66 * (-2 * a[10] + a[8]) * X ** 7
                            + 78 * a[9] * X ** 8
                            + 91 * a[10] * X ** 9
                        )
                    )
                )
            )
            * dX ** 2
        )
        S[i, 0] = SE * (y - np.sin(2 * np.pi * y) / 2 / np.pi)
        S[i, 1] = SE * (dy - np.cos(2 * np.pi * y) * dy)
        S[i, 2] = SE * (
            ddy - np.cos(2 * np.pi * y) * ddy + 2 * np.pi * np.sin(2 * np.pi * y) * dy ** 2
        )
    S[2 * Nte + 1 :, 0] = SE

    return S


@njit("f8[:,:](f8[:])")
def gauss_n4(a):
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


@njit("f8[:,:](f8[:])")
def gauss_n6(a):
    """サイクロイド軌道
    """
    c = np.array([a[4], a[5]])
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
