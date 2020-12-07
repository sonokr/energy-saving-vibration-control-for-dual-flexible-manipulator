import argparse
from logging import DEBUG, Formatter, StreamHandler, getLogger

import numpy as np
import pandas as pd

from models.eval import energy, torque
from models.rk4 import RK4
from models.traj import cycloid
from utils.config import gen_cfg, set_cfg
from utils.data import plot_graph
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


def run_test(cfg):
    """パラメータから直接テストを実行
    """
    a = np.array([0.09384623, -0.08338488, 0.14962311, 0.67922146, 1.1630475, -1.37600067])
    print(f"param: {a}")

    S = cycloid(a, cfg)

    X1, X2 = RK4(S)
    w1 = X1[0, :] * 2.7244
    w2 = X2[0, :] * 2.7244

    trq = torque(S, X1, X2)

    df = pd.DataFrame(
        {
            "t": np.linspace(0, cfg["CALC"]["TEND"], cfg["CALC"]["Nrk"] + 1),
            "θ": S[0 : 2 * cfg["CALC"]["Nrk"] + 1 : 2, 0],
            "dθ": S[0 : 2 * cfg["CALC"]["Nrk"] + 1 : 2, 1],
            "ddθ": S[0 : 2 * cfg["CALC"]["Nrk"] + 1 : 2, 2],
            "trq": trq,
            "w1": w1,
            "w2": w2,
        }
    )

    create_dirs("./data/test/output/")
    df.to_csv(
        f"./data/test/output/{0}_output_{cfg['COMM']['OPTIM']}_{cfg['COMM']['MODE']}_ \
        te{cfg['CALC']['TE']}_se{cfg['CALC']['SE']}.csv"
    )

    plot_graph(df, cfg)

    print(f'ene: {energy(df["trq"], df["θ"])}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file", help="Config file path")
    args = parser.parse_args()

    gen_cfg(args.cfg_file)
    cfg = set_cfg(args.cfg_file)
    logger.info(f'TE = {cfg["CALC"]["TE_str"]}[s], SE = {cfg["CALC"]["SE_str"]}[deg]')
    run_test(cfg)
