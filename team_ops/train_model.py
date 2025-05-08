import logging
import pandas as pd
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


from hydra import compose, initialize


class Model:
    """Experimental"""

    def __init__(self, conf_path: str = "conf", conf_file: str = "config"):

        # Setting up basic configuration for logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        self.log = logging.getLogger(__name__)  # Initializing a logger instance

        self.log.info("Loading config file from : %s/%s.yaml", conf_path, conf_file)

        # Setting a path that holds all the config files
        with initialize(config_path=conf_path, version_base=None):
            # Reading and loading configuration into 'cfg' instance variable.
            self._cfg = compose(config_name=conf_file)

        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self._cfg["model"]["pretrained_model"]
        )
        print(type(self._tokenizer))

        self._data: pd.DataFrame = pd.read_csv(self._cfg["data"]["path"])
        self._feature: str = self._cfg["data"]["feature"]
        self._target: str = self._cfg["data"]["target"]

    def tokenize(self):
        """Experimental"""
        self._data[self._feature] = self._data[self._feature].apply(
            lambda x: self._tokenizer(x, truncation=True)
        )
        print(self._data.head(5))
