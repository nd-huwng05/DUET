from config.config import Config
from argparse import ArgumentParser
from module import Module

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config-file", '-c',type=str, default="config/yaml/densenet121_UET.yaml")
    parser.add_argument("--dataset",'-d', type=str, default="VinCXR")
    parser.add_argument("--num-models",'-k', type=int, default=3)
    parser.add_argument("--name-model",'-n', type=str, default="DUET")
    config_init = parser.parse_args()

    config = Config(config_init.config_file)
    config.add("DATASET", "NAME", config_init.dataset)
    config.add("TRAIN", "NAME", config_init.name_model)
    config.add("TRAIN", "NUM_MODEL", config_init.num_model)
    if not config.TRAIN.RESUME:
        config.add("TRAIN", "INDEX", 0)

    Module(config).forward()