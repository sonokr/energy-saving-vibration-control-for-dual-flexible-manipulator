import csv
import time
from multiprocessing import Pool

import pandas as pd

from cond import *
from data import plot_graph
from eval import energy, torque
from pso_gauss import PSO as PSO_GAUSS
from pso_power import PSO as PSO_POWER
from rk4 import RK4
from traj import cycloid


def train(v):
    v["datadir"] = f"data/te{v['TE']}_se{v['SE']}/{v['mode']}/"

    if v["mode"] == "power":
        optim = PSO_POWER(v)
    elif v["mode"] == "gauss_n2":
        optim = PSO_GAUSS(v)

    # 1回だけ計算
    # results = [optim.compute(0)]
    # 10回計算
    p = Pool(4)
    results = p.map(optim.compute, range(10))

    for i, res in enumerate(results):
        np.savetxt(
            v["datadir"] + f"param/{i}_param_pso_{v['mode']}_te{v['TE']}_se{v['SE']}.csv",
            res,
            delimiter=",",
        )


def test(v):
    """学習したパラメーターからテストを実行
    """
    v["datadir"] = f"data/te{v['TE']}_se{v['SE']}/{v['mode']}/"

    sum_energy = 0
    for i in range(10):
        v["i"] = i

        param_path = (
            v["datadir"] + f"param/{v['i']}_param_pso_{v['mode']}_te{v['TE']}_se{v['SE']}.csv"
        )
        with open(param_path) as file:
            reader = csv.reader(file)
            a = np.array([float(row[0]) for row in reader])
        print(f"Param: {a}")

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
        df.to_csv(
            v["datadir"] + f"output/{v['i']}_output_pso_{v['mode']}_te{v['TE']}_se{v['SE']}.csv"
        )

        sum_energy += energy(df["trq"], df["θ"])
        print(f'{i}_ene: {energy(df["trq"], df["θ"])}')

        plot_graph(df, v, isShow=True)

    print(f"Average of energy: {sum_energy / 10}")


if __name__ == "__main__":
    start = time.time()

    v = {
        "mode": "gauss_n2",
        "TE": "1.0",
        "SE": "135",
    }
    test(v)

    print(f"Elapsed time: {time.time()-start}")
