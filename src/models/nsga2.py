import time
from logging import DEBUG, Formatter, StreamHandler, getLogger

import numpy as np
from platypus import NSGAII, Problem, Real

from models.eval import energy, torque
from models.rk4 import RK4
from models.traj import cycloid
from utils.utils import create_dirs

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
plain_formatter = Formatter(
    "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
)
handler.setFormatter(plain_formatter)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


def update_param(a, mode):
    if mode == "gauss_n4":
        for i in range(int(len(a) / 2)):
            if a[i] > 0.2:
                a[i] = 0.2
            elif a[i] < -0.2:
                a[i] = -0.2
        for i in range(int(len(a) / 2), len(a)):
            if a[i] > 1:
                a[i] = 1
            elif a[i] < 0:
                a[i] = 0
    elif mode == "gauss_n6":
        for i in range(0, 2):
            if a[i] > 0.2:
                a[i] = 0.2
            elif a[i] < -0.2:
                a[i] = -0.2
        for i in range(2, 4):
            if a[i] > 1:
                a[i] = 1
            elif a[i] < 0:
                a[i] = 0
        if a[4] > -0.5:
            a[4] = -0.5
        elif a[4] < -2.0:
            a[4] = -2.0
        if a[5] > 2.0:
            a[5] = 2.0
        elif a[5] < 0.5:
            a[5] = 0.5

    return a


class NSGA2:
    def __init__(self, cfg):
        self.cfg = cfg

    def error_func1(self, a):
        a = update_param(a, self.cfg["COMM"]["MODE"])

        S = cycloid(np.array(a), self.cfg)
        if np.abs(S[0 : 2 * self.cfg["CALC"]["Nrk"] + 1, 2]).max() >= 45:
            return [10 ** 6, 10 ** 6]

        X1, X2 = RK4(S)
        trq = torque(S, X1, X2)

        error = np.amax(abs(X1[0, self.cfg["CALC"]["Nte"] + 1 :])) + np.amax(
            abs(X2[0, self.cfg["CALC"]["Nte"] + 1 :])
        )
        ene = energy(trq, S[0 : 2 * self.cfg["CALC"]["Nrk"] + 1 : 2, 0])

        return [error, ene]

    def error_func2(self, a):
        a = update_param(a, self.cfg["COMM"]["MODE"])

        S = cycloid(np.array(a), self.cfg)
        if np.abs(S[0 : 2 * self.cfg["CALC"]["Nrk"] + 1, 2]).max() >= 60:
            return [10 ** 6, 10 ** 6, 10 ** 6]

        X1, X2 = RK4(S)
        trq = torque(S, X1, X2)

        return [
            np.amax(abs(X1[0, self.cfg["CALC"]["Nte"] + 1 :])),
            np.amax(abs(X2[0, self.cfg["CALC"]["Nte"] + 1 :])),
            energy(trq, S[0 : 2 * self.cfg["CALC"]["Nrk"] + 1 : 2, 0]),
        ]

    def error_func3(self, a):
        a = update_param(a, self.cfg["COMM"]["MODE"])

        S = cycloid(np.array(a), self.cfg)
        if np.abs(S[0 : 2 * self.cfg["CALC"]["Nrk"] + 1, 2]).max() >= 45:
            return [10 ** 6, 10 ** 6]

        X1, X2 = RK4(S)

        return [
            np.amax(abs(X1[0, self.cfg["CALC"]["Nte"] + 1 :])),
            np.amax(abs(X2[0, self.cfg["CALC"]["Nte"] + 1 :])),
        ]

    def run(self, pid):
        logger.info(f"pid{pid} start training.")

        start = time.time()

        problem = Problem(
            self.cfg["COMM"]["PARAM"], self.cfg["NSGA2"]["OBJECT"]
        )  # 最適化パラメータの数, 目的関数の数
        problem.types[:] = Real(-2.0, 2.0)  # パラメータの範囲

        if self.cfg["NSGA2"]["ERROR"] == "func1":
            problem.function = self.error_func1
        elif self.cfg["NSGA2"]["ERROR"] == "func2":
            problem.function = self.error_func2
        elif self.cfg["NSGA2"]["ERROR"] == "func3":
            problem.function = self.error_func3
        else:
            raise Exception(f'Invalid NSGA2 error function {self.cfg["NSGA2"]["ERROR"]}')

        algorithm = NSGAII(problem)
        algorithm.run(self.cfg["NSGA2"]["EPOCH"])  # 反復回数

        # データ整理
        # params: 係数a
        # f1s   : 残留振動 [deg]
        # f2s   : エネルギー
        params = np.empty([100, self.cfg["COMM"]["PARAM"]])
        if self.cfg["NSGA2"]["ERROR"] == "func1":
            f1s = np.empty([100])
            f2s = np.empty([100])
            for i, solution in enumerate(algorithm.result):
                result = tuple(solution.variables + solution.objectives[:])
                params[i, :] = result[: self.cfg["COMM"]["PARAM"]][:]
                f1s[i] = 180 * result[self.cfg["COMM"]["PARAM"]] / np.pi
                f2s[i] = result[self.cfg["COMM"]["PARAM"] + 1]
            # 残留振動が最小になるaの値を表示
            index = np.argmin(f1s)
            logger.info(f"pid{pid} sum_vib = {f1s[index]:.3f}[deg], ene = {f2s[index]:.3f}[J]")
            logger.info(f"pid{pid} a = {params[index, :]}")

            create_dirs(self.cfg["DATA"]["DIR"] + "param/")
            create_dirs(self.cfg["DATA"]["DIR"] + "data/")

            np.savetxt(
                self.cfg["DATA"]["DIR"]
                + f'param/{pid}_param_nsga2_{self.cfg["COMM"]["MODE"]}_\
te{self.cfg["CALC"]["TE_str"]}_se{self.cfg["CALC"]["SE_str"]}.csv',
                params[index, :],
                delimiter=",",
            )
            logger.info(
                f"pid{pid} saved param at "
                + self.cfg["DATA"]["DIR"]
                + f'param/{pid}_param_nsga2_{self.cfg["COMM"]["MODE"]}_\
te{self.cfg["CALC"]["TE_str"]}_se{self.cfg["CALC"]["SE_str"]}.csv'
            )

            # 係数a, 残留振動, エネルギーをCSVファイルに書き出す
            data = np.empty([100, self.cfg["COMM"]["PARAM"] + self.cfg["NSGA2"]["OBJECT"]])
            data[:, 0 : self.cfg["COMM"]["PARAM"]] = params
            data[:, self.cfg["COMM"]["PARAM"]] = f1s
            data[:, self.cfg["COMM"]["PARAM"] + 1] = f2s
            np.savetxt(
                self.cfg["DATA"]["DIR"]
                + f'data/{pid}_data_nsga2_{self.cfg["COMM"]["MODE"]}_\
te{self.cfg["CALC"]["TE_str"]}_se{self.cfg["CALC"]["SE_str"]}.csv',
                data,
                delimiter=",",
            )
            logger.info(
                f"pid{pid} saved data at "
                + self.cfg["DATA"]["DIR"]
                + f'data/{pid}_data_nsga2_{self.cfg["COMM"]["MODE"]}_\
te{self.cfg["CALC"]["TE_str"]}_se{self.cfg["CALC"]["SE_str"]}.csv'
            )
        elif self.cfg["NSGA2"]["ERROR"] == "func2":
            f1s = np.empty([100])
            f2s = np.empty([100])
            f3s = np.empty([100])
            for i, solution in enumerate(algorithm.result):
                result = tuple(solution.variables + solution.objectives[:])
                params[i, :] = result[: self.cfg["COMM"]["PARAM"]][:]
                f1s[i] = 180 * result[self.cfg["COMM"]["PARAM"]] / np.pi
                f2s[i] = 180 * result[self.cfg["COMM"]["PARAM"] + 1] / np.pi
                f3s[i] = result[self.cfg["COMM"]["PARAM"] + 2]
            index = np.argmin(f1s)
            logger.info(
                f"pid{pid} vib1 = {f1s[index]:.3f}[deg], \
vib2 = {f2s[index]:.3f}[deg], ene = {f3s[index]:.3f}[J]"
            )
            logger.info(f"pid{pid} a = {params[index, :]}")

            create_dirs(self.cfg["DATA"]["DIR"] + "param/")
            create_dirs(self.cfg["DATA"]["DIR"] + "data/")

            np.savetxt(
                self.cfg["DATA"]["DIR"]
                + f'param/{pid}_param_nsga2_{self.cfg["COMM"]["MODE"]}_\
te{self.cfg["CALC"]["TE_str"]}_se{self.cfg["CALC"]["SE_str"]}.csv',
                params[index, :],
                delimiter=",",
            )
            logger.info(
                f"pid{pid} saved param at "
                + self.cfg["DATA"]["DIR"]
                + f'param/{pid}_param_nsga2_{self.cfg["COMM"]["MODE"]}_\
te{self.cfg["CALC"]["TE_str"]}_se{self.cfg["CALC"]["SE_str"]}.csv'
            )

            # 係数a, 残留振動, エネルギーをCSVファイルに書き出す
            data = np.empty([100, self.cfg["COMM"]["PARAM"] + self.cfg["NSGA2"]["OBJECT"]])
            data[:, 0 : self.cfg["COMM"]["PARAM"]] = params
            data[:, self.cfg["COMM"]["PARAM"]] = f1s
            data[:, self.cfg["COMM"]["PARAM"] + 1] = f2s
            data[:, self.cfg["COMM"]["PARAM"] + 2] = f3s
            np.savetxt(
                self.cfg["DATA"]["DIR"]
                + f'data/{pid}_data_nsga2_{self.cfg["COMM"]["MODE"]}_\
te{self.cfg["CALC"]["TE_str"]}_se{self.cfg["CALC"]["SE_str"]}.csv',
                data,
                delimiter=",",
            )
            logger.info(
                f"pid{pid} saved data at "
                + self.cfg["DATA"]["DIR"]
                + f'data/{pid}_data_nsga2_{self.cfg["COMM"]["MODE"]}_\
te{self.cfg["CALC"]["TE_str"]}_se{self.cfg["CALC"]["SE_str"]}.csv'
            )
        elif self.cfg["NSGA2"]["ERROR"] == "func3":
            f1s = np.empty([100])
            f2s = np.empty([100])
            for i, solution in enumerate(algorithm.result):
                result = tuple(solution.variables + solution.objectives[:])

                params[i, :] = result[: self.cfg["COMM"]["PARAM"]][:]
                f1s[i] = 180 * result[self.cfg["COMM"]["PARAM"]] / np.pi
                f2s[i] = 180 * result[self.cfg["COMM"]["PARAM"]] / np.pi
            # 残留振動が最小になるaの値を表示
            index = np.argmin(f1s)
            logger.info(f"pid{pid} vib1 = {f1s[index]:.3f}[deg], ene2 = {f2s[index]:.3f}[deg]")
            logger.info(f"pid{pid} a = {params[index, :]}")

            create_dirs(self.cfg["DATA"]["DIR"] + "param/")
            create_dirs(self.cfg["DATA"]["DIR"] + "data/")

            np.savetxt(
                self.cfg["DATA"]["DIR"]
                + f'param/{pid}_param_nsga2_{self.cfg["COMM"]["MODE"]}_\
te{self.cfg["CALC"]["TE_str"]}_se{self.cfg["CALC"]["SE_str"]}.csv',
                params[index, :],
                delimiter=",",
            )
            logger.info(
                f"pid{pid} saved param at "
                + self.cfg["DATA"]["DIR"]
                + f'param/{pid}_param_nsga2_{self.cfg["COMM"]["MODE"]}_\
te{self.cfg["CALC"]["TE_str"]}_se{self.cfg["CALC"]["SE_str"]}.csv'
            )

            # 係数a, 残留振動, エネルギーをCSVファイルに書き出す
            data = np.empty([100, self.cfg["COMM"]["PARAM"] + self.cfg["NSGA2"]["OBJECT"]])
            data[:, 0 : self.cfg["COMM"]["PARAM"]] = params
            data[:, self.cfg["COMM"]["PARAM"]] = f1s
            data[:, self.cfg["COMM"]["PARAM"] + 1] = f2s
            np.savetxt(
                self.cfg["DATA"]["DIR"]
                + f'data/{pid}_data_nsga2_{self.cfg["COMM"]["MODE"]}_\
te{self.cfg["CALC"]["TE_str"]}_se{self.cfg["CALC"]["SE_str"]}.csv',
                data,
                delimiter=",",
            )
            logger.info(
                f"pid{pid} saved data at "
                + self.cfg["DATA"]["DIR"]
                + f'data/{pid}_data_nsga2_{self.cfg["COMM"]["MODE"]}_\
te{self.cfg["CALC"]["TE_str"]}_se{self.cfg["CALC"]["SE_str"]}.csv'
            )
        else:
            raise Exception(f'Invalid NSGA2 error function {self.cfg["NSGA2"]["ERROR"]}')

        logger.info(f"pid{pid} finished in {int(time.time()-start)}[s]")
