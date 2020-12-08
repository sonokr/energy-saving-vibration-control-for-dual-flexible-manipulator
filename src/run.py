import argparse
import atexit
from logging import DEBUG, Formatter, StreamHandler, getLogger
from multiprocessing import Pool

import numpy as np

from models.nsga2 import NSGA2
from models.pso import PSO_GAUSS4, PSO_GAUSS6, PSO_POWER
from utils.config import gen_cfg, set_cfg
from utils.line_notify import send_line_notify
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


def train(cfg):
    if cfg["COMM"]["OPTIM"] == "pso":
        if cfg["COMM"]["MODE"] == "power":
            optim = PSO_POWER(cfg)
        elif cfg["COMM"]["MODE"] == "gauss_n4":
            optim = PSO_GAUSS4(cfg)
        elif cfg["COMM"]["MODE"] == "gauss_n6":
            optim = PSO_GAUSS6(cfg)
        else:
            raise Exception(f'Invalid input function {cfg["COMM"]["MODE"]}.')

        p = Pool(2)
        results = p.map(optim.compute, range(cfg["COMM"]["EXEC"]))

        for i, res in enumerate(results):
            create_dirs(cfg["DATA"]["DIR"] + "param/")
            np.savetxt(
                cfg["DATA"]["DIR"]
                + f'param/{i}_param_{cfg["COMM"]["OPTIM"]}_{cfg["COMM"]["MODE"]}_\
te{cfg["CALC"]["TE_str"]}_se{cfg["CALC"]["SE_str"]}.csv',
                res,
                delimiter=",",
            )
            logger.info(
                "saved param at "
                + cfg["DATA"]["DIR"]
                + f'param/{i}_param_{cfg["COMM"]["OPTIM"]}_{cfg["COMM"]["MODE"]}_\
te{cfg["CALC"]["TE_str"]}_se{cfg["CALC"]["SE_str"]}.csv'
            )
    elif cfg["COMM"]["OPTIM"] == "nsga2":
        p = Pool(2)
        optim = NSGA2(cfg)
        results = p.map(optim.run, range(cfg["COMM"]["EXEC"]))
    else:
        raise Exception(f'Invalid optimizer {cfg["COMM"]["OPTIM"]}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="Config file path")
    parser.add_argument("--line", help="Line Notify")
    args = parser.parse_args()

    gen_cfg(args.cfg)
    cfg = set_cfg(args.cfg)

    if args.line:
        send_line_notify("学習開始！")
        atexit.register(send_line_notify)

    train(cfg)
