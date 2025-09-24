from models.model.daeu import DAEU
from models.model.duet import DUET


def handler_model(config):
    if config.TRAIN.MODEL_NAME == "DAEU":
        model = DAEU(input_size=config.DATASET.IMAGE_SIZE,latent_size=config.TRAIN.LATENT_SIZE,
                     expansion=config.TRAIN.MULTIPLIER,layer=config.TRAIN.LAYER).to("cuda")
    elif config.TRAIN.MODEL_NAME == "DUET":
        model = DUET(image_size=config.DATASET.IMAGE_SIZE, out_channels_pre=config.TRAIN.OUT_CHANNELS_PRE,
                     pretrained_idx=config.TRAIN.INDEX, pretrained=config.TRAIN.PRETRAINED).to("cuda")
    else:
        raise NotImplementedError

    return model