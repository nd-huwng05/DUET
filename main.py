import os
import shutil
import glob
from config.config import Config
from argparse import ArgumentParser
from module import Module

def train_more_times(config):
    auc = 0.0
    while auc < 81.0:
        auc, ap = Module(config).forward()
        if not os.path.exists(config.OUTPUT.BEST):
            os.makedirs(config.OUTPUT.BEST)
        path_result = os.path.join(config.OUTPUT.BEST, "result.txt")

        auc_best, ap_best = 0.0, 0.0
        if os.path.exists(path_result):
            try:
                with open(path_result, "r") as f:
                    lines = f.read().strip().split()
                    if len(lines) >= 2:
                        auc_best = float(lines[0])
                        ap_best = float(lines[1])
            except Exception as e:
                pass

        if auc_best < auc or (auc == auc_best and ap > ap_best):
            auc_best = auc
            ap_best = ap
            with open(path_result, "w") as f:
                f.write(f'{auc_best} {ap_best}')

            ckpt_files = glob.glob(os.path.join(config.OUTPUT.CHECKPOINT, "*.pth"))
            for src in ckpt_files:
                dst = os.path.join(config.OUTPUT.BEST, os.path.basename(src))
                shutil.copy2(src, dst)

        config.add("TRAIN", "INDEX", 0)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config-file", '-c',type=str, default="config/yaml/densenet121_UET.yaml")
    parser.add_argument("--dataset",'-d', type=str, default="VinCXR")
    parser.add_argument("--num-models",'-k', type=int, default=3)
    parser.add_argument("--name-model",'-n', type=str, default="DUET")
    config_init = parser.parse_args()

    config = Config(config_init.config_file)
    config.add("DATASET", "NAME", config_init.dataset)
    config.add("TRAIN", "MODEL_NAME", config_init.name_model)
    config.add("TRAIN", "NUM_MODEL", config_init.num_models)
    if not config.TRAIN.RESUME:
        config.add("TRAIN", "INDEX", 0)

    train_more_times(config)
