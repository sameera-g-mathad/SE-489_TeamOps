from datasets import (
    load_dataset,
    Dataset,
)
import duckdb
import pandas as pd
from team_ops.hydra_config import HConfig


class LLMDataset(HConfig):
    """
    This class is used to load, process, and save a dataset for LLM training.
    It inherits from HConfig to manage configuration settings.
    """

    _dataset: Dataset
    _raw: pd.DataFrame
    _processed: pd.DataFrame

    def __init__(self, conf_path: str = "conf", conf_file: str = "data"):
        """
        Initialize the LLMDataset class.
        """
        super().__init__(conf_path, conf_file)
        _dataset = load_dataset(self._cfg["data"]["repo_name"], split="train")
        if isinstance(_dataset, Dataset):
            self._dataset = _dataset

    def _load_raw(self):
        """
        Load the raw dataset into a pandas DataFrame and register it with DuckDB.
        """
        _df = self._dataset.to_pandas()
        if isinstance(_df, pd.DataFrame):
            self._raw = _df
        duckdb.register("df_raw", self._raw)

    def _save_raw(self):
        """
        Save the raw dataset to a CSV file.
        The file path and name are specified in the configuration.
        """
        self._raw.to_csv(
            self._cfg["data"]["raw_path"] + self._cfg["data"]["raw_filename"],
            index=self._cfg["data"]["index"],
        )
        self._log.info("Raw dataset saved as %s", self._cfg["data"]["raw_path"])

    def _load_processed(self):
        """
        Load the processed dataset from DuckDB using a query specified in the configuration.
        The processed dataset is stored in a pandas DataFrame.
        """
        self._processed = duckdb.query(self._cfg["data"]["query"]).to_df()

    def _save_processed(self):
        """
        Save the processed dataset to a CSV file.
        The file path and name are specified in the configuration.
        """
        self._processed.to_csv(
            self._cfg["data"]["processed_path"]
            + self._cfg["data"]["processed_filename"],
            index=self._cfg["data"]["index"],
        )
        self._log.info("Raw dataset saved as %s", self._cfg["data"]["processed_path"])

    def _target_to_int(self):
        """
        Convert the target column in the processed dataset to integer values.
        If the target column is of type 'object', it maps unique values to integers.
        The new target column is specified in the configuration.
        """
        if self._processed[self._cfg["data"]["target"]].dtype == "object":
            unique_targets = (
                self._processed[self._cfg["data"]["target"]].unique().tolist()
            )
            unique_targets.sort()
            self._processed[self._cfg["data"]["new_target_col"]] = self._processed[
                self._cfg["data"]["target"]
            ].map(lambda x: unique_targets.index(x))
            if self._cfg["data"]["drop_old_target"]:
                self._processed = self._processed.drop(
                    columns=self._cfg["data"]["target"]
                )

    def process(self):
        """
        Process the dataset by loading, saving, and transforming it.
        This method orchestrates the entire dataset processing pipeline:
        - Load the raw dataset from the Hugging Face repository.
        - Save the raw dataset to a CSV file.
        - Load the raw dataset into DuckDB.
        - Load the processed dataset using a query.
        - Convert the target column to integer values.
        - Save the processed dataset to a CSV file.
        """
        self._load_raw()
        self._save_raw()
        self._load_processed()
        self._target_to_int()
        self._save_processed()


dataset = LLMDataset()
dataset.process()
