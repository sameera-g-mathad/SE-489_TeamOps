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
from transformers.trainer_utils import EvalPrediction
import evaluate
from team_ops.hydra_config import HConfig


class Model(HConfig):
    """Experimental"""

    _accuracy: evaluate.EvaluationModule
    _dataset: DatasetDict
    _data_collator: DataCollatorWithPadding
    _model: PreTrainedModel
    _tokenizer: PreTrainedTokenizer

    def __init__(self, conf_path: str = "conf", conf_file: str = "config"):
        """Experimental"""
        super().__init__(conf_path, conf_file)

        self._log.info("Loading tokenizer: %s", self._cfg["model"]["pretrained_model"])
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._cfg["model"]["pretrained_model"]
        )
        self._log.info("%s loaded succesfully.", type(self._tokenizer).__name__)

        self._data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer)

        self._accuracy = evaluate.load("accuracy")

        self._log.info("Loading dataset from %s", self._cfg["data"]["path"])
        self._data: pd.DataFrame = pd.read_csv(self._cfg["data"]["path"])
        self._log.info("Data loaded succesfully.")

        _unique_labels = self._data[self._cfg["data"]["label"]].unique().tolist()
        _unique_targets = self._data[self._cfg["data"]["target"]].unique().tolist()
        _target2label = {
            target: label for label, target in zip(_unique_labels, _unique_targets)
        }
        _label2target = {
            label: target for label, target in zip(_unique_labels, _unique_targets)
        }

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._cfg["model"]["pretrained_model"],
            num_labels=len(_unique_labels),
            id2label=_target2label,
            label2id=_label2target,
        )

        if self._cfg["data"]["drop_target"]:
            self._data = self._data.drop(columns=self._cfg["data"]["target"])

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

    def compute_metrics(self, eval_pred: EvalPrediction) -> dict:
        """Experimental"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        _accuracy = self._accuracy.compute(predictions=predictions, references=labels)
        if isinstance(_accuracy, dict):
            return _accuracy
        return {"accuracy": _accuracy}

    def train(self):
        """Experimenatal"""
        train_args = self._cfg["train"]
        training_args = TrainingArguments(
            output_dir=f"{train_args['output_path']}/{train_args['model_name']}",
            learning_rate=train_args["learning_rate"],
            per_device_train_batch_size=train_args["per_device_train_batch_size"],
            per_device_eval_batch_size=train_args["per_device_eval_batch_size"],
            num_train_epochs=train_args["num_train_epochs"],
            weight_decay=train_args["weight_decay"],
            eval_strategy=train_args["eval_strategy"],
            save_strategy=train_args["save_strategy"],
            load_best_model_at_end=train_args["load_best_model_at_end"],
            push_to_hub=train_args["push_to_hub"],
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=self._dataset["train"],
            eval_dataset=self._dataset["test"],
            processing_class=self._tokenizer,
            data_collator=self._data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
