from dataclasses import dataclass
from logging import DEBUG, Formatter, StreamHandler, getLogger
from pprint import pprint

import numpy as np
import yaml

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


@dataclass
class COMM:
    OPTIM: str
    MODE: str
    PARAM: int
    EXEC: int
    PLOT: bool


@dataclass
class NSGA2:
    ERROR: str
    OBJECT: int
    EPOCH: int


@dataclass
class CALC:
    TE: float
    TE_str: str
    SE: int
    SE_str: str
    SE_rad: float
    DT: float
    TEND: float
    Nrk: float
    Nte: float


@dataclass
class DATA:
    DIR: str
    DATE: str


@dataclass
class Config:
    COMM: COMM
    NSGA2: NSGA2
    CALC: CALC
    DATA: DATA


def set_cfg_as_dict(yaml_path):
    with open(yaml_path, "r") as yml:
        cfg = yaml.load(yml, Loader=yaml.SafeLoader)

    cfg["CALC"]["TE_str"] = str(cfg["CALC"]["TE"])
    cfg["CALC"]["SE_str"] = str(cfg["CALC"]["SE"])
    cfg["CALC"]["SE_rad"] = np.deg2rad(cfg["CALC"]["SE"])

    cfg["CALC"]["Nrk"] = round(cfg["CALC"]["TEND"] / cfg["CALC"]["DT"])
    cfg["CALC"]["Nte"] = round(cfg["CALC"]["TE"] / cfg["CALC"]["DT"])

    if not cfg["DATA"]["DIR"]:
        if cfg["COMM"]["OPTIM"] == "nsga2":
            outdir = f'./data/{cfg["DATA"]["DATE"]}/te{cfg["CALC"]["TE_str"]}/se{cfg["CALC"]["SE_str"]}\
/{cfg["COMM"]["OPTIM"]}/{cfg["COMM"]["MODE"]}/{cfg["COMM"]["PARAM"]}/{cfg["NSGA2"]["ERROR"]}/'
        elif cfg["COMM"]["OPTIM"] == "pso":
            if cfg["COMM"]["MODE"] == "power":
                outdir = f'./data/{cfg["DATA"]["DATE"]}/te{cfg["CALC"]["TE_str"]}/se{cfg["CALC"]["SE_str"]}\
/{cfg["COMM"]["OPTIM"]}/{cfg["COMM"]["MODE"]}/{cfg["COMM"]["PARAM"]}/'
            elif cfg["COMM"]["MODE"] == "gauss_n4" or cfg["COMM"]["MODE"] == "gauss_n6":
                outdir = f'./data/{cfg["DATA"]["DATE"]}/te{cfg["CALC"]["TE_str"]}/se{cfg["CALC"]["SE_str"]}\
/{cfg["COMM"]["OPTIM"]}/{cfg["COMM"]["MODE"]}/'
            else:
                raise Exception("Invalid input function to set output directort.")
        else:
            raise Exception("Invalid optimizer to set output directory.")
        cfg["DATA"]["DIR"] = outdir

        logger.info("loaded config.")
        pprint(cfg)

    return cfg


def set_cfg(yaml_path):
    cfg_dict = set_cfg_as_dict(yaml_path)

    comm = COMM(**cfg_dict["COMM"])
    nsga2 = NSGA2(**cfg_dict["NSGA2"])
    calc = CALC(**cfg_dict["CALC"])
    data = DATA(**cfg_dict["DATA"])

    cfg = Config(comm, nsga2, calc, data)

    gen_cfg(cfg)

    return cfg


def gen_cfg(cfg):
    with open("src/config/config_of_calc.py", "w") as f:
        f.write(f"TE = {cfg.CALC.TE}\n")
        f.write(f"SE = {np.deg2rad(cfg.CALC.SE)}\n")
        f.write(f"dt = {cfg.CALC.DT}\n")
        f.write(f"Tend = {cfg.CALC.TEND}\n")
        f.write(f"Nrk = {round(cfg.CALC.TEND / cfg.CALC.DT)}\n")
        f.write(f"Nte = {round(cfg.CALC.TE / cfg.CALC.DT)}\n")

    logger.info("updated temporary config file.")


if __name__ == "__main__":
    cfg = set_cfg("./src/config/sample.yaml")
