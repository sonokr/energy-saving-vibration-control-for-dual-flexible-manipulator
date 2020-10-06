import argparse
import csv
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from cond import *
from data import plot_graph
from eval import energy, torque
from pso import PSO_GAUSS, PSO_POWER
from rk4 import RK4
from traj import cycloid


def train(v, process):
    if v["mode"] == "power":
        optim = PSO_POWER(v)
    elif v["mode"] == "gauss_n2":
        optim = PSO_GAUSS(v)

    if process == "single":
        results = [optim.compute(0)]
    elif process == "parallel":
        p = Pool(4)
        results = p.map(optim.compute, range(10))

    for i, res in enumerate(results):
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


def test(v, process):
    """学習したパラメーターからテストを実行
    """
    count = 1 if process == "single" else 10

    energys = {}
    for i in range(count):
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
        df.to_csv(v["datadir"] + f"output/{i}_output_pso_{v['mode']}_te{v['TE']}_se{v['SE']}.csv")

        energys[i] = energy(df["trq"], df["θ"])
        print(f'    {i}_ene: {energy(df["trq"], df["θ"])}\n')

        plot_graph(df, v)

    print("energy")
    for k, v in sorted(energys.items(), key=lambda x: x[1]):
        print(k, v)
    print(f"mean {sum(energys.values()) / len(energys.values())}")


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Train or Test.")
    parser.add_argument("process", help="Single or Parallel.")
    parser.add_argument("--plot", help="Plot graph or not.")
    args = parser.parse_args()

    v = {
        "mode": "power",
        "TE": str(TE),
        "SE": str(int(np.rad2deg(SE))),
        "isShow": bool(args.plot) if args.plot else False,
    }

    v["param_count"] = 7 if v["mode"] == "power" else 4
    # v["datadir"] = f"data/te{v['TE']}_se{v['SE']}/{v['mode']}/"
    v["datadir"] = f"data/2020-10-05/te{v['TE']}_se{v['SE']}/{v['param_count']}/"

    if args.mode == "train":
        print("#####################")
        print("#   Train Running   #")
        print("#####################")
        print("TE = {}[s], SE = {}[deg]".format(str(TE), str(int(np.rad2deg(SE)))), end="\n\n")
        train(v, args.process)
    elif args.mode == "test":
        print("####################")
        print("#   Test Running   #")
        print("####################")
        print("TE = {}[s], SE = {}[deg]".format(str(TE), str(int(np.rad2deg(SE)))), end="\n\n")
        test(v, args.process)

    print(f"Elapsed time: {time.time()-start}")
