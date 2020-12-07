import numpy as np
from numba import njit

from config.config_of_calc import *
from config.param_of_equation import *


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
