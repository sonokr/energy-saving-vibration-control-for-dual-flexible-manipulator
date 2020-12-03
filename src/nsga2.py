import time
from multiprocessing import Pool

import numpy as np
from platypus import NSGAII, Problem, Real

from cond import *
from eval import energy, torque
from rk4 import RK4
from traj import cycloid
from utils import create_dirs

nsga2_count = 0
print(f"0{nsga2_count} NSGA2 {nsga2_count}0")


def schaffer(a):
    global nsga2_count
    nsga2_count += 1
    print(f"{nsga2_count} : {a}")

    S = cycloid(np.array(a), {"mode": "power"})
    if np.abs(S[0 : 2 * Nrk + 1, 2]).max() >= 45:
        return [10 ** 6, 10 ** 6]

    X1, X2 = RK4(S)
    trq = torque(S, X1, X2)

    error = abs(X1[0, Nte + 1 :]).max() + abs(X2[0, Nte + 1 :]).max()
    ene = energy(trq, S[0 : 2 * Nrk + 1 : 2, 0])

    return [error, ene]


def run(cur_exec):
    start = time.time()

    problem = Problem(param_count, 2)  # 最適化パラメータの数, 目的関数の数
    problem.types[:] = Real(-2.0, 2.0)  # パラメータの範囲
    problem.function = schaffer
    algorithm = NSGAII(problem)
    algorithm.run(5000)  # 反復回数

    print("{:-^63}".format("-"))

    # データ整理
    # params: 係数a
    # f1s   : 残留振動 [deg]
    # f2s   : エネルギー
    params = np.empty([100, param_count])
    f1s = np.empty([100])
    f2s = np.empty([100])
    for i, solution in enumerate(algorithm.result):
        result = tuple(solution.variables + solution.objectives[:])

        params[i, :] = result[:param_count][:]
        f1s[i] = 180 * result[param_count] / np.pi
        f2s[i] = result[param_count + 1]

    # 残留振動が最小になるaの値を表示
    index = np.argmin(f1s)
    print("\n*** 残留振動が最小の時の各値 ***")
    print("残留振動[deg]\t{}".format(f1s[index]))
    print("エネルギー[J]\t{}".format(f2s[index]))
    print("係数a\t\t{}".format(params[index, :]))

    # 経過時間
    print(f"\nelapsed time: {time.time()-start}")

    datadir = f"./data/2020-12-03/te{str(TE)}/se{str(int(np.rad2deg(SE)))}/{mode}/"
    if mode == "power":
        datadir += f"{param_count}/"

    create_dirs(datadir + "param/")
    create_dirs(datadir + "data/")

    np.savetxt(
        datadir
        + f"param/{cur_exec}_param_nsga2_{mode}_te{str(TE)}_se{str(int(np.rad2deg(SE)))}.csv",
        params[index, :],
        delimiter=",",
    )

    # 係数a, 残留振動, エネルギーをCSVファイルに書き出す
    data = np.empty([100, param_count + 2])
    data[:, 0:param_count] = params
    data[:, param_count] = f1s
    data[:, param_count + 1] = f2s
    np.savetxt(
        datadir
        + f"data/{cur_exec}_data_nsga2_{mode}_te{str(TE)}_se{str(int(np.rad2deg(SE)))}.csv",
        data,
        delimiter=",",
    )


if __name__ == "__main__":
    # p = Pool(4)
    # results = p.map(run, range(4))

    print("##################")
    print("### Train Once ###")
    print("##################")
    print("TE = {}[s], SE = {}[deg]".format(str(TE), str(int(np.rad2deg(SE)))), end="\n\n")

    start = time.time()

    for i in [4, 5, 6, 7]:
        v = {
            "mode": "power",
            "TE": str(TE),
            "SE": str(int(np.rad2deg(SE))),
            "plot": plot,
            "param_count": i,
        }
        v["datadir"] = f"./data/2020-12-03/te{v['TE']}/se{v['SE']}/{v['mode']}/{i}/"
        print("mode = {}, param = {}".format(v["mode"], v["param_count"]), end="\n\n")
        p = Pool(4)
        results = p.map(run, range(4))

    v["mode"] = "gauss_n4"
    v["param_count"] = 4
    v["datadir"] = f"./data/2020-12-03/te{v['TE']}/se{v['SE']}/{v['mode']}/"
    print("mode = {}, param = {}".format(v["mode"], v["param_count"]), end="\n\n")
    p = Pool(4)
    results = p.map(run, range(4))

    v["mode"] = "gauss_n6"
    v["param_count"] = 6
    v["datadir"] = f"./data/2020-12-03/te{v['TE']}/se{v['SE']}/{v['mode']}/"
    print("mode = {}, param = {}".format(v["mode"], v["param_count"]), end="\n\n")
    p = Pool(4)
    results = p.map(run, range(4))

    print(f"Elapsed time: {time.time()-start}")
