import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
import evaluate
from team_ops.hydra_config import HConfig


class Model(HConfig):
    """Experimental"""

    _accuracy: evaluate.EvaluationModule
    _dataset: DatasetDict | None = None
    _data_collator: DataCollatorWithPadding
    _model: PreTrainedModel
    _tokenizer: PreTrainedTokenizer

    def __init__(self, conf_path: str = "conf", conf_file: str = "config"):
        """Experimental"""
        super().__init__(conf_path, conf_file)

        self.log.info("Loading tokenizer: %s", self._cfg["model"]["pretrained_model"])
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._cfg["model"]["pretrained_model"]
        )
        self.log.info("%s loaded succesfully.", type(self._tokenizer).__name__)

        self._data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer)

        self._accuracy = evaluate.load("accuracy")

        self.log.info("Loading dataset from %s", self._cfg["data"]["path"])
        self._data: pd.DataFrame = pd.read_csv(self._cfg["data"]["path"])
        self.log.info("Data loaded succesfully.")

        # self._model = AutoModelForSequenceClassification.from_pretrained(
        #     self._cfg["model"]["pretrained_model"],
        #     num_labels=2,
        #     id2label=id2label,
        #     label2id=label2id,
        # )

        self._feature: str = self._cfg["data"]["feature"]
        self._target: str = self._cfg["data"]["target"]

    def preprocess(self, examples):
        """Experimental"""
        return self._tokenizer(examples[self._feature], truncation=True)

    def tokenize(self):
        """Experimental"""
        if isinstance(self._dataset, DatasetDict):
            self._dataset = self._dataset.map(self.preprocess, batched=True)

    def make_data(self):
        """
        Experiment
        """
        train_df = self._data.sample(n=self._cfg["data"]["train_ratio"])
        test_df = self._data[~self._data.index.isin(train_df.index)]

        print(train_df.shape, test_df.shape)

        self._dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df),
                "test": Dataset.from_pandas(test_df),
            }
        )
        del self._data
        self.tokenize()

    def compute_metrics(self, eval_pred):
        """Experimental"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self._accuracy.compute(predictions=predictions, references=labels)
        self._data[self._feature] = self._data[self._feature].apply(lambda x: self._tokenizer(x, truncation=True))
        print(self._data.head(5))
