from utils.logger import Logger
from utils.test import InferenceGrad
from utils.train import Trainer
from utils.utils import set_seed


class Module:
    def __init__(self, config):
        self.config = config
        self.logger = Logger.init(self.config.OUTPUT.LOGS_TRAIN)
        self.trainer = Trainer(self.config)
        self.tester = InferenceGrad(self.config)

    def forward(self):
        self.logger.set_log_file(self.config.OUTPUT.LOGS_TRAIN)
        self.logger.get_logger().info(f"Training started for {self.config.DATASET.NAME} dataset")
        self.trainer.train()
        self.logger.get_logger().info("\n")
        self.logger.set_log_file(self.config.OUTPUT.LOGS_TEST)
        set_seed(self.config.TRAIN.SEED)
        auc, ap = self.tester.inference()
        return auc, ap
