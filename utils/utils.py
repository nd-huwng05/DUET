import os
import torch
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