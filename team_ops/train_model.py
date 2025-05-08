import pandas as pd
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from team_ops.hydra_config import HConfig


class Model(HConfig):
    """Experimental"""

    def __init__(self, conf_path: str = "conf", conf_file: str = "config"):
        """Experimental"""
        super().__init__(conf_path, conf_file)

        self.log.info("Loading tokenizer: %s", self._cfg["model"]["pretrained_model"])
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self._cfg["model"]["pretrained_model"])
        self.log.info("%s loaded succesfully.", type(self._tokenizer).__name__)

        self.log.info("Loading dataset from %s", self._cfg["data"]["path"])
        self._data: pd.DataFrame = pd.read_csv(self._cfg["data"]["path"])
        self.log.info("Data loaded succesfully.")

        self._feature: str = self._cfg["data"]["feature"]
        self._target: str = self._cfg["data"]["target"]

    def tokenize(self):
        """Experimental"""
        self._data[self._feature] = self._data[self._feature].apply(lambda x: self._tokenizer(x, truncation=True))
        print(self._data.head(5))
