import logging
import sys
from pathlib import Path
import os


class Logger:
    _instance = None
    _logger = None

    def __new__(cls, *args, **kwargs):
        raise RuntimeError("Use Logger.init(log_file) to initialize the logger.")

    @classmethod
    def init(cls, log_file: str):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._setup_logger(log_file)
        return cls._instance

    @classmethod
    def _setup_logger(cls, log_file: str):
        logger = logging.getLogger("UET_Logger")
        logger.setLevel(logging.INFO)

        if logger.handlers:
            logger.handlers.clear()

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "[%(name)s] %(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # File handler
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter(
            "[%(name)s] %(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        cls._logger = logger
        cls._log_file = log_file

    @classmethod
    def get_logger(cls):
        if cls._logger is None:
            raise RuntimeError("Logger is not initialized. Call Logger.init(log_file) first.")
        return cls._logger

    @classmethod
    def set_log_file(cls, log_file: str):
        if cls._logger is None:
            raise RuntimeError("Logger is not initialized. Call Logger.init(log_file) first.")
        cls._setup_logger(log_file)
