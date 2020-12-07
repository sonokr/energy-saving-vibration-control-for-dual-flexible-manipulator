import argparse
from logging import DEBUG, Formatter, StreamHandler, getLogger
from multiprocessing import Pool

import numpy as np

from models.nsga2 import NSGA2
from models.pso import PSO_GAUSS4, PSO_GAUSS6, PSO_POWER
from utils.config import gen_cfg, set_cfg
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

        p = Pool(4)
        results = p.map(optim.compute, range(cfg["COMM"]["EXEC"]))

        for i, res in enumerate(results):
            create_dirs(cfg["OUTPUT_DIR"] + "param/")
            np.savetxt(
                cfg["OUTPUT_DIR"]
                + f'param/{i}_param_{cfg["COMM"]["OPTIM"]}_{cfg["COMM"]["MODE"]}_ \
                te{cfg["CALC"]["TE_str"]}_se{cfg["CALC"]["SE_str"]}.csv',
                res,
                delimiter=",",
            )
            logger.info(
                "Parameter saved at \n    "
                + cfg["OUTPUT_DIR"]
                + f'param/{i}_param_{cfg["COMM"]["OPTIM"]}_{cfg["COMM"]["MODE"]}_ \
                te{cfg["CALC"]["TE_str"]}_se{cfg["CALC"]["SE_str"]}.csv'
            )
    elif cfg["COMM"]["OPTIM"] == "nsga2":
        optim = NSGA2(cfg)
        optim.run(0)
    else:
        raise Exception(f'Invalid optimizer {cfg["COMM"]["OPTIM"]}.')


def run(args):
    gen_cfg(args.cfg_file)
    cfg = set_cfg(args.cfg_file)
    logger.info(f'TE = {cfg["CALC"]["TE_str"]}[s], SE = {cfg["CALC"]["SE_str"]}[deg]')
    train(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file", help="Config file path")
    args = parser.parse_args()

    run(args)
