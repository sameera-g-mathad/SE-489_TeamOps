import os
import numpy as np
import pandas as pd
import torch
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
    """
    This class is used to load and train a model using the Hugging Face Transformers library.
    It initializes the model, tokenizer, and data collator.
    """

    _accuracy: evaluate.EvaluationModule
    _dataset: DatasetDict
    _data_collator: DataCollatorWithPadding
    _model: PreTrainedModel
    _tokenizer: PreTrainedTokenizer

    def __init__(self, conf_path: str = "conf", conf_file: str = "config"):
        """Experimental"""
        super().__init__(conf_path, conf_file)

        # Load the tokenizer
        self._log.info("Loading tokenizer: %s", self._cfg["model"]["pretrained_model"])
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._cfg["model"]["pretrained_model"]
        )
        self._log.info("%s loaded succesfully.", type(self._tokenizer).__name__)

        # Load the data collator
        self._log.info("Loading data collator")
        self._data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer)

        # Load the accuracy metric
        self._accuracy = evaluate.load("accuracy")

        # Load the dataset
        self._log.info("Loading dataset from %s", self._cfg["data"]["path"])
        self._data: pd.DataFrame = pd.read_csv(self._cfg["data"]["path"])
        self._log.info("Data loaded succesfully.")

        _unique_labels = self._data[self._cfg["data"]["label"]].unique().tolist()
        _unique_targets = self._data[self._cfg["data"]["target"]].unique().tolist()
        self._target2label = {
            target: label for label, target in zip(_unique_labels, _unique_targets)
        }
        self._label2target = {
            label: target for label, target in zip(_unique_labels, _unique_targets)
        }

        # Load the model
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._cfg["model"]["pretrained_model"],
            num_labels=len(_unique_labels),
            id2label=self._target2label,
            label2id=self._label2target,
        )

        # Drop the target column if specified
        if self._cfg["data"]["drop_target"]:
            self._data = self._data.drop(columns=self._cfg["data"]["target"])

        self._feature: str = self._cfg["data"]["feature"]
        self._target: str = self._cfg["data"]["target"]

    def preprocess(self, examples: pd.Series):
        """
        This function preprocesses the dataset by tokenizing the input text.
        It uses the tokenizer to convert the text into input IDs and attention masks.
        Args:
            examples (pd.Series): The input text to be tokenized.
        Returns:
            dict: A dictionary containing the input IDs and attention masks.
        """
        return self._tokenizer(examples[self._feature], truncation=True)

    def tokenize(self):
        """
        This function tokenizes the dataset using the tokenizer.
        It applies the preprocessing function to the dataset.
        """
        if isinstance(self._dataset, DatasetDict):
            self._dataset = self._dataset.map(self.preprocess, batched=True)

    def make_data(self):
        """
        This function splits the dataset into training and testing sets.
        It uses the train ratio specified in the configuration file to determine the split.
        The training set is sampled from the original dataset, and the testing set is created by removing the training samples.
        The resulting datasets are stored in a DatasetDict.
        """

        # Sample the training set
        self._log.info("Splitting dataset into train and test sets.")
        train_df = self._data.sample(n=self._cfg["data"]["train_ratio"])

        # Create the test set by removing the training samples
        self._log.info("Creating test set.")
        test_df = self._data[~self._data.index.isin(train_df.index)]

        self._log.info(train_df.shape, test_df.shape)

        # Create the DatasetDict
        self._dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df),
                "test": Dataset.from_pandas(test_df),
            }
        )
        self._log.info("Dataset created successfully.")
        del self._data, train_df, test_df

        self._log.info("Tokenizing dataset.")
        self.tokenize()

    def compute_metrics(self, eval_pred: EvalPrediction) -> dict:
        """
        This function computes the accuracy of the model on the test set.
        It uses the accuracy metric from the evaluate library.
        Args:
            eval_pred (EvalPrediction): The predictions and labels from the model.
        Returns:
            dict: A dictionary containing the accuracy of the model.
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        _accuracy = self._accuracy.compute(predictions=predictions, references=labels)
        if isinstance(_accuracy, dict):
            return _accuracy
        return {"accuracy": _accuracy}

    def train(self):
        """
        This function trains the model using the Trainer class from the Hugging Face Transformers library.
        It uses the training arguments specified in the configuration file.
        The model is trained on the training set and evaluated on the test set.
        The trained model is saved to the specified output directory.
        The output directory is created if it doesn't exist.
        """
        train_args = self._cfg["train"]
        out_dir = f"{train_args['output_path']}/{train_args['model_name']}"

        if os.path.exists(out_dir):
            self._log.info("Model already exists. Skipping training.")
            self._model = AutoModelForSequenceClassification.from_pretrained(
                out_dir,
                num_labels=len(self._target2label),
                id2label=self._target2label,
                label2id=self._label2target,
            )
            return

        self._log.info("Training model.")
        # Create the output directory if it doesn't exist
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

        # Create the Trainer
        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=self._dataset["train"],
            eval_dataset=self._dataset["test"],
            processing_class=self._tokenizer,
            data_collator=self._data_collator,
            compute_metrics=self.compute_metrics,
        )
        # Train the model
        self._log.info("Starting training.")
        trainer.train()
        self._log.info("Training completed.")
        # Save the model
        self._log.info("Saving model to %s", out_dir)
        self._model.save_pretrained(out_dir)
        self._tokenizer.save_pretrained(out_dir)
        self._log.info("Model saved successfully.")

    def predict(self, text: str) -> str:
        """
        This function makes predictions using the trained model.
        It takes a string input and returns the predicted label.
        Args:
            text (str): The input text to be predicted.
        Returns:
            str: The predicted label.
        """
        self._model.eval()
        with torch.inference_mode():
            self._log.info("Making predictions.")
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True)
            outputs = self._model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

        return self._target2label[predictions.item()]
