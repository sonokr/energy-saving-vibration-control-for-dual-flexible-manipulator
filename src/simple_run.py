import argparse

parser = argparse.ArgumentParser()
parser.add_argument("cfg", help="Config file path")
args = parser.parse_args()

from utils.config import set_cfg

cfg = set_cfg(args.cfg)

from logging import DEBUG, Formatter, StreamHandler, getLogger

import numpy as np
import pandas as pd

from models.eval import energy, torque
from models.rk4 import RK4
from models.traj import cycloid

# from utils.data import plot_graph
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


def str2list(pstr):
    return list(map(float, pstr.split()))


def run_test(cfg):
    """パラメータから直接テストを実行
    """
    a = np.array(
        str2list(
            "6.455348882490680176e-06 2.205882734637162890e-02 1.144408684542324445e-01 -1.232142017960534019e+00 -2.722007589944505090e-01 -6.212503361141449298e-01"
        )
    )
    print(f"param: {a}")

    S = cycloid(a, cfg)

    X1, X2 = RK4(S)
    w1 = X1[0, :] * 2.7244
    w2 = X2[0, :] * 2.7244

    trq = torque(S, X1, X2)

    df = pd.DataFrame(
        {
            "t": np.linspace(0, cfg.CALC.TEND, cfg.CALC.Nrk + 1),
            "θ": S[0 : 2 * cfg.CALC.Nrk + 1 : 2, 0],
            "dθ": S[0 : 2 * cfg.CALC.Nrk + 1 : 2, 1],
            "ddθ": S[0 : 2 * cfg.CALC.Nrk + 1 : 2, 2],
            "trq": trq,
            "w1": w1,
            "w2": w2,
        }
    )

    create_dirs(cfg.DATA.DIR)
    df.to_csv(
        cfg.DATA.DIR
        + f"{0}_output_{cfg.COMM.OPTIM}_{cfg.COMM.MODE}_\
te{cfg.CALC.TE}_se{cfg.CALC.SE}.csv"
    )

    # plot_graph(df, cfg)

    print(f'ene: {energy(df["trq"], df["θ"])}\n')


if __name__ == "__main__":
    run_test(cfg)
