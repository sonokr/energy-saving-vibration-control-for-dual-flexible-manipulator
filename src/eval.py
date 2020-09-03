from scipy import integrate

from cond import *


def torque(S, X1, X2):
    """トルクを計算する
    """
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


def energy(f, x):
    """消費エネルギーを計算する
    """

    def find_inflection_point(x):
        """変曲点を見つける
        """
        inflection_point = [0]
        for i in range(1, len(x) - 1):
            if (x[i] >= x[i - 1] and x[i] >= x[i + 1]) or (x[i] <= x[i - 1] and x[i] <= x[i + 1]):
                inflection_point.append(i + 1)
        if inflection_point[-1] != len(x) - 1:
            inflection_point.append(len(x) - 1)
        return inflection_point

    p = find_inflection_point(x)
    ene = 0
    for i in range(1, len(p)):
        ene = ene + abs(integrate.simps(np.abs(f[p[i - 1] : p[i]]), x[p[i - 1] : p[i]]))
    return ene
