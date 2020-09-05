import copy
import random

import numpy as np
from tqdm import tqdm

from cond import *
from eval import torque
from rk4 import RK4
from traj import cycloid


class PSO:
    def __init__(self, v):
        self.parti_count = 50
        self.param_count = 6
        self.loop = 200
        self.v = v

    def evaluate(self, a):
        """評価関数
        """
        S = cycloid(a, self.v)
        if S[0 : 2 * Nrk + 1, 2].max() >= 50:
            return 10 ** 6

        X1, X2 = RK4(S)
        trq = torque(S, X1, X2)
        return sum(np.absolute(trq))

    def init_pos(self):
        """PSOの位置を初期化
        """
        return np.array(
            [
                [random.uniform(a_min, a_max) for i in range(self.param_count)]
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
            elif new_a[i] < -2.0:
                new_a[i] = -2.0
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
        vel = np.array([[0.0 for i in range(self.param_count)] for j in range(self.parti_count)])

        p_best_pos = copy.deepcopy(pos)
        p_best_scores = [self.evaluate(p) for p in pos]

        best_parti = np.argmin(p_best_scores)
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


class PSO_POWER(PSO):
    pass


class PSO_GAUSS(PSO):
    def __init__(self, v):
        self.parti_count = 50
        self.param_count = 4
        self.loop = 200
        self.v = v

    def init_pos(self):
        """PSOの位置を初期化
        """
        return np.array(
            [
                [
                    random.uniform(-0.2, 0.2),
                    random.uniform(-0.2, 0.2),
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                ]
                for j in range(self.parti_count)
            ]
        )

    def update_pos(self, a, va):
        """位置をアップデート
        """
        new_a = a + va
        for i in range(int(len(new_a) / 2)):
            if new_a[i] > 0.2:
                new_a[i] = 0.2
            elif new_a[i] < -0.2:
                new_a[i] = -0.2
        for i in range(int(len(new_a) / 2), len(new_a)):
            if new_a[i] > 1:
                new_a[i] = 1
            elif new_a[i] < 0:
                new_a[i] = 0

        return new_a
