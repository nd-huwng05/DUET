import os
import torch
import random
import numpy as np
from models.handler import handler_model


def loading_model(config, requires_grap=False):
    models = []
    for state_dict in sorted(os.listdir(config.OUTPUT.CHECKPOINT), key=lambda x: int(x.split(".")[0])):
        model = handler_model(config)
        model.load_state_dict(torch.load(os.path.join(config.OUTPUT.CHECKPOINT, state_dict), map_location=torch.device('cpu:{}'.format(config.GPUS))))
        model.eval()
        if not requires_grap:
            for param in model.parameters():
                param.requires_grad = False
        models.append(model)
        config.add("TRAIN", "INDEX", config.TRAIN.INDEX + 1)
    return models

def load_models(config, requires_grap=False):
    models = []
    config.add("TRAIN", "INDEX", 0)
    for state_dict in sorted(os.listdir(config.OUTPUT.CHECKPOINT), key=lambda x: int(x.split(".")[0])):
        model = handler_model(config)
        model.load_state_dict(torch.load(os.path.join(config.OUTPUT.CHECKPOINT, state_dict),
                                         map_location=torch.device('cuda:{}'.format(config.GPUS))))
        model.eval()
        if not requires_grap:
            for param in model.parameters():
                param.requires_grad = False
        models.append(model)
        config.add("TRAIN", "INDEX", config.TRAIN.INDEX + 1)
    return models

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True