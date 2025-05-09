import logging
from hydra import initialize, compose


class HConfig:
    """
    This class is used to load and manage the configuration files using Hydra.
    It initializes the logger and loads the configuration file.
    The configuration file is loaded from the specified path and file name.
    This class is intended to be used as a base class for other classes that require configuration management.
    Args:
        conf_path (str): The path to the configuration file.
        conf_file (str): The name of the configuration file.
    Attributes:
        _cfg (dict): The loaded configuration file.
        _log (logging.Logger): The logger instance.
    """

    def __init__(self, conf_path: str = "conf", conf_file: str = "config"):
        # Setting up basic configuration for logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        self._log = logging.getLogger(__name__)  # Initializing a logger instance

        self._log.info("Loading config file from : %s/%s.yaml.", conf_path, conf_file)
        # Setting a path that holds all the config files
        with initialize(config_path=conf_path, version_base=None):
            # Reading and loading configuration into 'cfg' instance variable.
            self._cfg = compose(config_name=conf_file)
