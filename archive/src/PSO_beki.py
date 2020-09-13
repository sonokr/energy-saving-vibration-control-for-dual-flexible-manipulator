# -*- coding: utf-8 -*-
import time

import numpy as np
from numba import jit

# グローバル変数
dt = 0.002
Tend = 2.0
TE = 0.8
SE = np.pi / 2.0
Nrk = round(Tend / dt)
Nte = round(TE / dt)

# @jit('f8[:](f8, f8, f8, f8)', nopython=True)
def f(x1, x2, ds, dds):
    a1 = 3.311 / 10
    a2 = 2.353 / 10
    c = 8.238 / 1000
    ome = 12.52
    dx1 = x2
    dx2 = -(ome ** 2 * x1 + 2.0 * c * ome * x2 + a1 * ds * ds * x1 + a2 * dds)
    return np.array([dx1, dx2])


#
# @jit('f8[:,:](f8[:],f8[:,:])', nopython=True)
# @jit('f8[:,:](f8[:])')
def tra(a, S):
    t = np.linspace(0.0, TE, 2 * Nte + 1)
    dX = 2.0 / TE
    for i in range(0, 2 * Nte + 1):
        X = -1 + 2 * t[i] / TE
        y = t[i] / TE + (1 - X) * (1 + X) * (
            a[0] + a[1] * X + a[2] * X ** 2 + a[3] * X ** 3 + a[4] * X ** 4 + a[5] * X ** 5
        )
        dy = (
            1 / TE
            - 2
            * X
            * dX
            * (a[0] + a[1] * X + a[2] * X ** 2 + a[3] * X ** 3 + a[4] * X ** 4 + a[5] * X ** 5)
            + (1 - X)
            * (1 + X)
            * (a[1] + 2 * a[2] * X + 3 * a[3] * X ** 2 + 4 * a[4] * X ** 3 + 5 * a[5] * X ** 4)
            * dX
        )
        ddy = (
            -2
            * (
                a[0]
                - a[2]
                + 3 * (a[1] - a[3]) * X
                + 6 * (a[2] - a[4]) * X ** 2
                + 10 * (a[3] - a[5]) * X ** 3
                + 15 * a[4] * X ** 4
                + 21 * a[5] * X ** 5
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


#
# @jit('f8[:,:](f8[:,:], f8[:,:], f8[:,:])', nopython=True)
def RK4(X, k, S):
    for i in range(0, Nrk):
        x1, x2 = X[:, i]
        k[:, 0] = f(x1, x2, S[2 * i, 1], S[2 * i, 2]) * dt
        k[:, 1] = f(x1 + k[0, 0] / 2.0, x2 + k[1, 0] / 2.0, S[2 * i + 1, 1], S[2 * i + 1, 2]) * dt
        k[:, 2] = f(x1 + k[0, 1] / 2.0, x2 + k[1, 1] / 2.0, S[2 * i + 1, 1], S[2 * i + 1, 2]) * dt
        k[:, 3] = f(x1 + k[0, 2], x2 + k[1, 2], S[2 * i + 2, 1], S[2 * i + 2, 2]) * dt
        X[:, i + 1] = X[:, i] + ((k[:, 0] + 2.0 * k[:, 1] + 2.0 * k[:, 2] + k[:, 3]) / 6.0)
    return X


#
# @jit('f8[:](f8[:], f8[:], f8[:], f8[:])', nopython=True)
def torqe(x1, x2, ds, dds):
    a1 = 3.311 / 10
    a2 = 2.353 / 10
    c = 8.238 / 1000
    ome = 12.52
    b1 = 1.662 / 100
    b2 = 6.309 / 100
    cc = 0.03205
    #
    ddx = -(ome ** 2 * x1 + 2.0 * c * ome * x2 + a1 * ds * ds * x1 + a2 * dds)
    T = b1 * dds + b2 * ddx + cc * ds
    return T


#
# @jit('f8(f8[:], f8[:])', nopython=True)
def simpui(y, x):
    w = 0
    for i in range(2, Nte + 1, 2):
        h = (x[i] - x[i - 2]) / 2.0
        hpd = 1.0 / (x[i - 1] - x[i - 2])
        d = x[i - 1] - x[i - 2] - h
        hmd = 1.0 / (x[i] - x[i - 1])
        ww = (1.0 + 2.0 * d * hpd) * y[i - 2]
        ww = ww + 2.0 * h * (hpd + hmd) * y[i - 1]
        ww = ww + (1.0 - 2.0 * d * hmd) * y[i]
        w = w + ww * h
    return w / 3.0


#
def cal_val(a):
    S = np.zeros([2 * Nrk + 1, 3])
    S = tra(a, S)
    X = np.zeros([2, Nrk + 1])
    k = np.empty([2, 4])
    X = RK4(X, k, S)
    trq = torqe(X[0, :], X[1, :], S[0 : 2 * Nrk + 1 : 2, 1], S[0 : 2 * Nrk + 1 : 2, 2])
    ene = simpui(np.abs(trq[: Nte + 1]), S[: 2 * Nte + 1 : 2, 0])
    w = X[0, :] * 2.724
    data = np.empty([Nrk + 1, 6])
    data[:, 0] = np.linspace(0.0, Tend, Nrk + 1)
    data[:, 1:4] = S[0 : 2 * Nrk + 1 : 2, :]
    data[:, 4] = w
    data[:, 5] = trq
    f_val = np.sum(np.abs(trq))
    return ene, f_val, data


#
def ini_particle(N, dim, XU, XL, VU, VL):
    x = np.random.rand(dim, N)
    v = np.random.rand(dim, N)
    for i in range(0, dim):
        x[i, :] = x[i, :] * (XU[i] - XL[i]) + XL[i]
        v[i, :] = v[i, :] * (VU[i] - VL[i]) + VL[i]
    # end for
    pbest = np.ones([dim + 1, N]) * 10 ** 6
    gbest = np.ones([dim + 2]) * 10 ** 6
    return x, v, pbest, gbest


#
def up_best(N, dim, x, pbest, gbest, dataRK):
    for i in range(0, N):
        ene, f_val, data_p = cal_val(x[:, i])
        if pbest[dim, i] > f_val:
            pbest[0:dim, i] = x[:, i]
            pbest[dim, i] = f_val
            if gbest[dim] > f_val:
                gbest[0:dim] = x[:, i]
                gbest[dim] = f_val
                gbest[dim + 1] = ene
                dataRK = data_p
            # end if
        # end if
    # enf for
    return pbest, gbest, dataRK


#
def up_x(N, dim, x, v, pbest, gbest, XU, XL):
    for i in range(0, N):
        cr1 = np.random.rand(dim) * 2.05
        cr2 = np.random.rand(dim) * 2.05
        d_p = pbest[0:dim, i] - x[:, i]
        d_g = gbest[0:dim] - x[:, i]
        v[:, i] = 0.729 * (v[:, i] + cr1 * d_p + cr2 * d_g)
        x[:, i] = x[:, i] + v[:, i]
        for j in range(0, dim):
            if x[j, i] > XU[j]:
                x[j, i] = XU[j]
            elif x[j, i] < XL[j]:
                x[j, i] = XL[j]
            # enf if
        # end for
    return x, v


#
def pso(N, dim, XU, XL, VU, VL, TMAX):
    dataRK = np.zeros([Nrk + 1, 6])
    x, v, pbest, gbest = ini_particle(N, dim, XU, XL, VU, VL)
    for k in range(0, TMAX):
        pbest, gbest, dataRK = up_best(N, dim, x, pbest, gbest, dataRK)
        x, v = up_x(N, dim, x, v, pbest, gbest, XU, XL)
        print(k, gbest[dim])
    return gbest, dataRK


#
def pro():
    TMAX = 200
    N = 50
    dim = 6
    XL = np.array([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2])
    XU = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    VL = XL / 5.0
    VU = XU / 5.0
    gbest, dataRK = pso(N, dim, XU, XL, VU, VL, TMAX)
    return gbest, dataRK


if __name__ == "__main__":
    start = time.time()
    gbest, data = pro()
    # data=np.c_[np.linspace(0., Tend, Nrk+1),dataRK]
    np.savetxt("gbest-beki.csv", gbest, delimiter=",")
    np.savetxt("dataOPT-beki.csv", data, delimiter=",")
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
