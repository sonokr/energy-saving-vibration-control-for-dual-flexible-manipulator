import argparse
import csv
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from cond import *
from data import plot_graph
from eval import energy, torque
from pso import PSO_GAUSS4, PSO_GAUSS6, PSO_POWER
from rk4 import RK4
from traj import cycloid
from utils import create_dirs


def train(v):
    if v["mode"] == "power":
        optim = PSO_POWER(v)
    elif v["mode"] == "gauss_n4":
        optim = PSO_GAUSS4(v)
    elif v["mode"] == "gauss_n6":
        optim = PSO_GAUSS6(v)

    if exec_count <= 0:
        exit()

    p = Pool(4)
    results = p.map(optim.compute, range(exec_count))

    for i, res in enumerate(results):
        create_dirs(v["datadir"] + "param/")
        np.savetxt(
            v["datadir"] + f"param/{i}_param_pso_{v['mode']}_te{v['TE']}_se{v['SE']}.csv",
            res,
            delimiter=",",
        )
        print(
            "Parameter saved at \n    "
            + v["datadir"]
            + f"param/{i}_param_pso_{v['mode']}_te{v['TE']}_se{v['SE']}.csv!\n",
            end="\n\n",
        )


def test(v):
    """学習したパラメーターからテストを実行
    """
    energys = {}
    for i in range(exec_count):
        v["i"] = i

        param_path = v["datadir"] + f"param/{i}_param_pso_{v['mode']}_te{v['TE']}_se{v['SE']}.csv"
        with open(param_path) as file:
            reader = csv.reader(file)
            a = np.array([float(row[0]) for row in reader])

        print(f'{i}. Parameter{i} loaded from \n    "{param_path}"!')
        print(f"    param: {a}")

        S = cycloid(a, v)

        X1, X2 = RK4(S)
        w1 = X1[0, :] * 2.7244
        w2 = X2[0, :] * 2.7244

        trq = torque(S, X1, X2)

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
        create_dirs(v["datadir"] + "output/")
        df.to_csv(v["datadir"] + f"output/{i}_output_pso_{v['mode']}_te{v['TE']}_se{v['SE']}.csv")

        energys[i] = energy(df["trq"], df["θ"])
        print(f'    {i}_ene: {energy(df["trq"], df["θ"])}\n')

        plot_graph(df, v)

    print("energy")
    for k, v in sorted(energys.items(), key=lambda x: x[1]):
        print(k, v)
    print(f"mean {sum(energys.values()) / len(energys.values())}")


def run_once(args):
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
        v["datadir"] = f"data/2020-10-21/te{v['TE']}/se{v['SE']}/{v['mode']}/{i}/"
        print("mode = {}, param = {}".format(v["mode"], v["param_count"]), end="\n\n")
        train(v)

    v["mode"] = "gauss_n4"
    v["param_count"] = 4
    v["datadir"] = f"data/2020-10-21/te{v['TE']}/se{v['SE']}/{v['mode']}/"
    print("mode = {}, param = {}".format(v["mode"], v["param_count"]), end="\n\n")
    train(v)

    v["mode"] = "gauss_n6"
    v["param_count"] = 6
    v["datadir"] = f"data/2020-10-21/te{v['TE']}/se{v['SE']}/{v['mode']}/"
    print("mode = {}, param = {}".format(v["mode"], v["param_count"]), end="\n\n")
    train(v)

    print(f"Elapsed time: {time.time()-start}")


def run_test():
    """学習したパラメーターからテストを実行
    """
    a = np.array(
        [
            1.09649864791602e-01,
            8.08354152160375e-02,
            1.43306289363947e00,
            1.0090255492455e00,
            1.11802962136856e00,
        ]
    )
    print(f"param: {a}")

    v = {
        "i": 0,
        "optim": "nsga2",
        "mode": mode,
        "TE": str(TE),
        "SE": str(int(np.rad2deg(SE))),
        "plot": plot,
        "param_count": param_count,
    }

    S = cycloid(a, v)

    X1, X2 = RK4(S)
    w1 = X1[0, :] * 2.7244
    w2 = X2[0, :] * 2.7244

    trq = torque(S, X1, X2)

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

    v["datadir"] = "./data/test/"
    create_dirs("./data/test/output/")
    df.to_csv(
        f"./data/test/output/{v['i']}_output_{v['optim']}_{v['mode']}_te{v['TE']}_se{v['SE']}.csv"
    )

    plot_graph(df, v)

    print(f'ene: {energy(df["trq"], df["θ"])}\n')


def run(args):
    start = time.time()

    v = {
        "mode": mode,
        "TE": str(TE),
        "SE": str(int(np.rad2deg(SE))),
        "plot": plot,
        "param_count": param_count,
    }
    v["datadir"] = f"data/2020-10-21/te{v['TE']}/se{v['SE']}/{v['mode']}/"
    if v["mode"] == "power":
        v["datadir"] += f"{v['param_count']}/"  # パラメータの数も記録する

    if args.exec_mode == "train":
        print("#####################")
        print("#   Train Running   #")
        print("#####################")
        print("TE = {}[s], SE = {}[deg]".format(str(TE), str(int(np.rad2deg(SE)))), end="\n\n")
        train(v)
    elif args.exec_mode == "test":
        print("####################")
        print("#   Test Running   #")
        print("####################")
        print("TE = {}[s], SE = {}[deg]".format(str(TE), str(int(np.rad2deg(SE)))), end="\n\n")
        test(v)

    print(f"Elapsed time: {time.time()-start}")


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("exec_mode", help="Train, Test, Self.")
    args = parser.parse_args()

    if args.exec_mode.upper() == "TRAIN" and once:
        run_once(args)
    elif args.exec_mode.upper() == "SELF":
        run_test()
    else:
        run(args)
