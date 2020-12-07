from pprint import pprint

import numpy as np
import yaml


def set_cfg(yaml_path):
    with open(yaml_path, "r") as yml:
        cfg = yaml.load(yml, Loader=yaml.SafeLoader)

    cfg["CALC"]["TE_str"] = str(cfg["CALC"]["TE"])
    cfg["CALC"]["SE_str"] = str(cfg["CALC"]["SE"])
    cfg["CALC"]["SE_rad"] = np.deg2rad(cfg["CALC"]["SE"])

    cfg["CALC"]["Nrk"] = round(cfg["CALC"]["TEND"] / cfg["CALC"]["DT"])
    cfg["CALC"]["Nte"] = round(cfg["CALC"]["TE"] / cfg["CALC"]["DT"])

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


if __name__ == "__main__":
    pprint(set_cfg("./src/config/nsga2.yaml"))
    gen_cfg("./src/config/nsga2.yaml")
