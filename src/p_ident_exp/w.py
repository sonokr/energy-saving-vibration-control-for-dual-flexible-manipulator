import copy
import random
from multiprocessing import Pool

import bokeh
import numpy as np
import pandas as pd
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from numba import njit
from tqdm import tqdm

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
    ome = 10.71 * np.float64(a_[0])
    z = 14.27 * 10.0 ** -3 * np.float64(a_[1])
    a = 2.570 * 10.0 ** -1 * np.float64(a_[2])
    b = 5.822 * 10.0 ** -3 * np.float64(a_[3])

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


class PSO:
    def __init__(self, w1_exp):
        self.parti_count = 50
        self.loop = 200
        self.param_count = 4
        self.w1_exp = w1_exp

    def evaluate(self, a):
        """評価関数
        """
        # TODO
        S = cycloid()

        X1 = RK4(S, a)
        w1 = X1[0, :] * 2.7244

        return sum(np.abs(w1 - self.w1_exp))

    def init_pos(self):
        """PSOの位置を初期化
        """
        return np.array(
            [
                [
                    random.uniform(0.5, 2.0),
                    random.uniform(0.5, 2.0),
                    random.uniform(0.5, 2.0),
                    random.uniform(0.5, 2.0),
                ]
                for j in range(self.parti_count)
            ]
        )

    def update_pos(self, a, va):
        """位置をアップデート
        """
        new_a = a + va
        for i in range(len(new_a)):
            if new_a[i] > 2.0:
                new_a[i] = 2.0
            elif new_a[i] < 0.5:
                new_a[i] = 0.5
        return new_a

    def update_vel(self, a, va, p, g, w_=0.730, p_=2.05):
        """速度をアップデート
        """
        ro1 = random.uniform(0, 1)
        ro2 = random.uniform(0, 1)
        return w_ * (va + p_ * ro1 * (p - a) + p_ * ro2 * (g - a))

    def compute(self, _):
        """PSOを計算
        """
        print("Initializing variables\n")

        pos = self.init_pos()
        # TODO
        vel = np.array([[0.0 for i in range(len(pos[0]))] for j in range(self.parti_count)])

        p_best_pos = copy.deepcopy(pos)
        p_best_scores = [self.evaluate(p) for p in pos]

        best_parti = np.argmin(p_best_scores)
        print(f"best: {best_parti}")
        g_best_pos = p_best_pos[best_parti]

        print("Start calculation")
        for t in tqdm(range(self.loop)):
            for n in range(self.parti_count):
                # n番目のx, y, vx, vyを定義
                a = pos[n]
                va = vel[n]
                p = p_best_pos[n]

                # 粒子の位置の更新
                new_a = self.update_pos(a, va)
                pos[n] = new_a

                # 粒子の速度の更新
                new_va = self.update_vel(a, va, p, g_best_pos)
                vel[n] = new_va

                # 評価値を求め、パーソナルベストの更新
                score = self.evaluate(new_a)
                if score < p_best_scores[n]:
                    p_best_scores[n] = score
                    p_best_pos[n] = new_a

            # グローバルベストの更新
            best_parti = np.argmin(p_best_scores)
            g_best_pos = p_best_pos[best_parti]

            print(f"param: {g_best_pos}")
            print(f"score: {np.min(p_best_scores)}\n")

        print(f"\ng_best_pos: {g_best_pos}\n")

        return g_best_pos


if __name__ == "__main__":
    v = {
        "target": "w2",
    }
    v["datadir"] = f"./data/p_ident_exp/{v['target']}/"
    ###################
    # experience data #
    ###################
    df = pd.read_csv("./data/exp9_8.csv")
    if v["target"] == "w1":
        w1_exp = np.array(df["先端の変位[1mm]"])
    elif v["target"] == "w2":
        w1_exp = np.array(df["先端の変位[0.8mm]"])
    else:
        raise Exception("invalid target")

    #############
    # calculate #
    #############
    optim = PSO(w1_exp)
    # results = [optim.compute(0)]
    p = Pool(4)
    results = p.map(optim.compute, range(10))

    ################
    # plot setting #
    ################
    output_file(v["datadir"] + "plot.html")
    width, height = 350, 250
    figs = [
        figure(
            width=width,
            plot_height=height,
            title=f"ome: {res[0]:.3f}, z: {res[1]:.3f}, a: {res[2]:.3f}, b: {res[3]:.3f}",
        )
        for res in results
    ]
    palette = bokeh.palettes.Category10[10]

    #############################
    # process parameters result #
    #############################
    for i, res in enumerate(results):
        ##############
        # save param #
        ##############
        np.savetxt(
            v["datadir"] + f"param/{i}_param.csv", res, delimiter=",",
        )
        print(
            "Parameter saved at \n    " + v["datadir"] + f"param/{i}_param.csv", end="\n\n",
        )

        ##############
        # simulation #
        ##############
        S = cycloid()

        X1 = RK4(S, res)
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
        df.to_csv(v["datadir"] + f"output/{i}_output.csv")
        print("Output saved at \n\t" + v["datadir"] + f"output/{i}_output.csv")

        ################
        # plot setting #
        ################
        figs[i].line(df["t"], df["w1"], line_color=palette[0])
        figs[i].line(df["t"], w1_exp, line_color=palette[1])

    ##############
    # total plot #
    ##############
    fig = gridplot(
        [
            [figs[0], figs[1], figs[2], figs[3]],
            [figs[4], figs[5], figs[6], figs[7]],
            [figs[8], figs[9]],
        ]
    )
    show(fig)
