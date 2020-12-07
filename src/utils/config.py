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


def set_cfg(yaml_path):
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


def gen_cfg(yaml_path):
    with open(yaml_path, "r") as yml:
        cfg = yaml.load(yml, Loader=yaml.SafeLoader)

    with open("src/config/config_of_calc.py", "w") as f:
        f.write(f'TE = {cfg["CALC"]["TE"]}\n')
        f.write(f'SE = {np.deg2rad(cfg["CALC"]["SE"])}\n')
        f.write(f'dt = {cfg["CALC"]["DT"]}\n')
        f.write(f'Tend = {cfg["CALC"]["TEND"]}\n')
        f.write(f'Nrk = {round(cfg["CALC"]["TEND"] / cfg["CALC"]["DT"])}\n')
        f.write(f'Nte = {round(cfg["CALC"]["TE"] / cfg["CALC"]["DT"])}\n')

    logger.info("updated temporary config file.")


if __name__ == "__main__":
    pprint(set_cfg("./src/config/nsga2.yaml"))
    gen_cfg("./src/config/nsga2.yaml")
