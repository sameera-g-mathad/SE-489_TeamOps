from datasets import (
    load_dataset,
    Dataset,
)
import duckdb
import pandas as pd
from team_ops.hydra_config import HConfig


class LLMDataset(HConfig):
    """This class is used to load and process the dataset."""

    _dataset: Dataset
    _raw: pd.DataFrame
    _processed: pd.DataFrame

    def __init__(self, conf_path: str = "conf", conf_file: str = "data"):
        """
        This function initializes the class and loads the dataset.
        Args:
            conf_path (str): The path to the config file.
            conf_file (str): The name of the config file.
        """
        super().__init__(conf_path, conf_file)
        _dataset = load_dataset(self._cfg["data"]["repo_name"], split="train")
        if isinstance(_dataset, Dataset):
            self._dataset = _dataset

    def _load_raw(self):
        """
        This function loads the raw dataset from the dataset object.
        It converts the dataset to a pandas dataframe and registers it with duckdb.
        """
        self._log.info("Loading raw dataset")
        _df = self._dataset.to_pandas()
        if isinstance(_df, pd.DataFrame):
            self._raw = _df
        duckdb.register("df_raw", self._raw)

    def _save_raw(self):
        """
        This function saves the raw dataset to a csv file.
        It uses the path and filename from the config file.
        """
        self._raw.to_csv(
            self._cfg["data"]["raw_path"] + self._cfg["data"]["raw_filename"],
            index=self._cfg["data"]["index"],
        )
        self._log.info("Raw dataset saved as %s", self._cfg["data"]["raw_path"])

    def _load_processed(self):
        """
        This function loads the processed dataset from the raw dataset.
        """
        self._processed = duckdb.query(self._cfg["data"]["query"]).to_df()

    def _save_processed(self):
        """
        This function saves the processed dataset to a csv file.
        """
        self._processed.to_csv(
            self._cfg["data"]["processed_path"]
            + self._cfg["data"]["processed_filename"],
            index=self._cfg["data"]["index"],
        )
        self._log.info("Raw dataset saved as %s", self._cfg["data"]["processed_path"])

    def _target_to_int(self):
        """This function converts the target column to an integer column.
        It uses the mapping from the config file to convert the target column to an integer column.
        """
        # If the target column is an object, convert it to an integer column
        if self._processed[self._cfg["data"]["target"]].dtype == "object":
            unique_targets = (
                self._processed[self._cfg["data"]["target"]].unique().tolist()
            )
            # Sort the unique targets to ensure consistent mapping
            unique_targets.sort()

            # Create a mapping from the unique targets to integers
            self._processed[self._cfg["data"]["new_target_col"]] = self._processed[
                self._cfg["data"]["target"]
            ].map(lambda x: unique_targets.index(x))

            # Rename the new target column to the original target column
            if self._cfg["data"]["drop_old_target"]:
                self._processed = self._processed.drop(
                    columns=self._cfg["data"]["target"]
                )

    def process(self):
        """
        This function processes the dataset.
        """
        self._load_raw()
        self._save_raw()
        self._load_processed()
        self._target_to_int()
        self._save_processed()


dataset = LLMDataset()
dataset.process()
