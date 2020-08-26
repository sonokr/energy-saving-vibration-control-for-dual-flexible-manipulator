import copy
import random
import time

import numpy as np
from tqdm import tqdm

from condition import *
from main import RK4, cycloid, torque


class PSO:
    def evaluate(self, a):
        S = cycloid(a)
        if S[0 : 2 * Nrk + 1 : 2, 2].max() >= 50:
            return 10 ** 6

        X1, X2 = RK4(S)
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
        trq = torque(S, ddW1, ddW2)
        return np.sum(abs(trq))

    def update_pos(self, a, va):
        return a + va

    def update_vel(self, a, va, p, g, w_=0.730, p_=2.05):
        ro1 = random.uniform(0, 1)
        ro2 = random.uniform(0, 1)
        return w_ * (va + p_ * ro1 * (p - a) + p_ * ro2 * (g - a))

    def compute(self):
        print("Initializing variables")
        parti_count = 100  # 粒子の数 : 100
        a_min, a_max = -2.0, 2.0

        param_count = 5
        pos = np.array(
            [
                [random.uniform(a_min, a_max) for i in range(param_count)]
                for j in range(parti_count)
            ]
        )
        vel = np.array([[0.0 for i in range(param_count)] for j in range(parti_count)])

        p_best_pos = copy.deepcopy(pos)
        p_best_scores = [self.evaluate(p) for p in pos]

        best_parti = np.argmin(p_best_scores)
        g_best_pos = p_best_pos[best_parti]

        loop_count = 200  # 制限時間 : 200
        print("Start calculation")
        for t in range(loop_count):
            print("{}/{}".format(t + 1, loop_count))
            for n in tqdm(range(parti_count)):
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

        print(f"係数: {g_best_pos}")

        return g_best_pos


if __name__ == "__main__":
    start = time.time()

    optim = PSO()
    res = optim.compute()
    np.savetxt("./data/dst/pso_param.csv", res, delimiter=",")

    elapsed_time = time.time() - start
    print(f"Elapsed Time : {elapsed_time} [sec]")
