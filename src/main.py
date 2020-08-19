import time

import numpy as np

# 条件値（いずれ設定ファイルにまとめる）
dt = 0.002
Tend = 3.0

TE = 1.2  # 駆動時間
SE = np.pi  # 目標角

Nrk = round(Tend / dt)
Nte = round(TE / dt)

# リンク1のパラメータ
ome1 = 12.68
z1 = 15.49 * 10 ** (-3)
a1 = 2.266 * 10 ** (-3)
b1 = 2.004 * 10 ** (-3)

# リンク2のパラメータ
ome2 = 10.71
z2 = 14.27 * 10 ** (-3)
a2 = 2.570 * 10 ** (-3)
b2 = 5.822 * 10 ** (-3)

# トルク関数のパラメータ
g1 = 2.996 * 10 ** (-2)
g2 = 6.652 * 10 ** (-2)
g3 = 5.237 * 10 ** (-2)
cs = 4.202 * 10 ** (-2)


def f(x11, x12, x21, x22, ds, dds):
    dx11 = x12
    dx12 = -(2 * z1 * ome1 * x12 + ome1 ** 2 * x11 + a1 * dds * b1 * x11 * ds ** 2)
    dx21 = x22
    dx22 = -(2 * z2 * ome2 * x22 + ome2 ** 2 * x21 + a2 * dds + b2 * x21 * ds ** 2)

    k1 = np.array([dx11, dx12]) * dt
    k2 = np.array([dx21, dx22]) * dt

    return k1, k2


def RK4(S):
    k1 = np.empty([2, 4])
    k2 = np.empty([2, 4])
    X1 = np.zeros([2, Nrk + 1])
    X2 = np.zeros([2, Nrk + 1])

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


def cyc():
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

    ddW1 = (
        2 * z1 * ome1 * X1[1, :]
        + ome1 ** 2 * X1[0, :]
        + a1 * S[0 : 2 * Nrk + 1 : 2, 2]
        + b1 * X1[0, :] * S[0 : 2 * Nrk + 1 : 2, 1] ** 2
    )
    ddW2 = (
        2 * z2 * ome2 * X2[1, :]
        + ome2 ** 2 * X2[0, :]
        + a2 * S[0 : 2 * Nrk + 1 : 2, 2]
        + b2 * X2[0, :] * S[0 : 2 * Nrk + 1 : 2, 1] ** 2
    )
    trq = g1 * S[0 : 2 * Nrk + 1 : 2, 2] + g2 * ddW1 + g3 * ddW2 + cs * S[0 : 2 * Nrk + 1 : 2, 1]

    data = np.empty([Nrk + 1, 7])
    data[:, 0] = np.linspace(0, Tend, Nrk + 1)
    data[:, 1:4] = S[0 : 2 * Nrk + 1 : 2, :]
    data[:, 4] = trq
    data[:, 5] = w1
    data[:, 6] = w2

    outdir = "./outputs/"
    np.savetxt(outdir + "cycloid_data.csv", data, delimiter=",")
    print(f"Elapsed time: {time.time()-start}")
