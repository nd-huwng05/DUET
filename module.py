from utils.logger import Logger
from utils.train import Trainer


class Module:
    def __init__(self, config):
        self.config = config
        self.logger = Logger.init(self.config.OUTPUT.LOGS_TRAIN)
        self.train = Trainer(self.config)

    def forward(self):
        self.logger.set_log_file(self.config.OUTPUT.LOGS_TRAIN)
        self.logger.get_logger().info(f"Training started for {self.config.DATASET.NAME} dataset")
        self.train.train_ensemble()
