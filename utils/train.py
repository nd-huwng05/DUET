from torchvision import transforms
from dataset.dataset import DatasetTemplate
from torch.utils.data import DataLoader

from models.handler import handler_model
from utils.logger import Logger
import torch
import os

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize(self.config.DATASET.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def train_ensemble(self):
        Logger.set_log_file(self.config.OUTPUT.LOGS_TRAIN)
        dataset_train = DatasetTemplate(self.config, mode='train')
        Logger.get_logger().info(f"Loading dataset_train {self.config.DATASET.NAME} includes {dataset_train.__len__()} images")
        train_loader = DataLoader(dataset_train, batch_size=self.config.TRAIN.BATCH_SIZE, shuffle=True,
                                  drop_last=True, num_workers=self.config.DATASET.NUM_WORKERS, pin_memory=True)
        dataset_val = DatasetTemplate(self.config, mode='test')
        Logger.get_logger().info(
            f"Loading dataset_test {self.config.DATASET.NAME} includes {dataset_val.__len__()} images")
        val_loader = DataLoader(dataset_val, batch_size=self.config.TRAIN.BATCH_SIZE, shuffle=False,
                                drop_last=True, num_workers=self.config.DATASET.NUM_WORKERS, pin_memory=True)

        seniors = []
        for i in range(self.config.TRAIN.INDEX):
            senior = handler_model(self.config.TRAIN.NAME)
            Logger.get_logger().info(f"Loading state dict senior {i}.....")
            senior.load_state_dict(torch.load(os.path.join(self.config.OUTPUT.CHECKPOINT, f"{i}.pth"),
                                              map_location=torch.device("cuda:{}".format(self.config.GPUS)),))
            seniors.append(senior)

        junior = handler_model(self.config.TRAIN.NAME)

        optimizer = torch.optim.AdamW(junior.parameters(), lr=self.config.TRAIN.LR, betas=(0.5,0.999),
                                      weight_decay=self.config.TRAIN.WEIGHT_DECAY)

        