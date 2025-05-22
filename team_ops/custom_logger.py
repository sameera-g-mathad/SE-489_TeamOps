import logging
import sys
from pathlib import Path
from logging.config import dictConfig
from rich.logging import RichHandler


class Logger:
    """
    Custom logger class that sets up logging configuration with Rich and RotatingFileHandler.
    """

    def __init__(self):
        base_dir = Path().resolve().parent
        logs_dir = Path(base_dir, "logs")  # This will create a folder called logs
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Advanced logging configuration with Rich
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "minimal": {"format": "%(message)s"},  # Rich adds its own formatting
                "detailed": {
                    "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                    "formatter": "minimal",
                    "level": logging.INFO,
                },
                "info": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": Path(logs_dir, "info.log"),
                    "maxBytes": 10485760,  # 10 MB
                    "backupCount": 10,
                    "formatter": "detailed",
                    "level": logging.INFO,
                },
            },
            "root": {
                "handlers": ["console", "info"],
                "level": logging.DEBUG,
                "propagate": True,
            },
        }

        dictConfig(logging_config)
        self._logger = logging.getLogger(__name__)

        # Replace the standard console handler with Rich handler
        self._logger.root.handlers[0] = RichHandler(markup=True, rich_tracebacks=True)

    @property
    def logger(self):
        """
        Returns the logger instance.
        """
        return self._logger
