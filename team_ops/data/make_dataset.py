from datasets import load_dataset, Dataset
import duckdb
import pandas as pd
from team_ops.hydra_config import HConfig

# Define a Dataset class that inherits from HConfig
class Dataset(HConfig):
    """Experimental"""
    _dataset: Dataset
    _raw: pd.DataFrame
    _processed: pd.DataFrame
    def __init__(self, conf_path: str = "conf", conf_file: str = "data"):
        """Initialize configuration and load the dataset from Hugging Face."""
        super().__init__(conf_path, conf_file)
        self._dataset = load_dataset(self._cfg['data']['repo_name'], split="train")
    
    def _load_raw(self):
        """Convert the loaded dataset to pandas DataFrame and register it with DuckDB."""
        self._raw = self._dataset.to_pandas()
        duckdb.register("df_raw", self._raw)
    
    def _save_raw(self):
        """Save the raw DataFrame to CSV using configured path and filename."""
        self._raw.to_csv(self._cfg['data']['raw_path'] + self._cfg['data']['raw_filename'], index=self._cfg['data']['index'])
        self._log.info("Raw dataset saved as %s", self._cfg['data']['raw_path'])

    def _load_processed(self):
        """Run a SQL query on the raw data to produce the processed DataFrame."""
        self._processed = duckdb.query(self._cfg['data']['query']).to_df()
    
    def _save_processed(self):
        """Save the processed DataFrame to CSV using configured path and filename."""
        self._processed.to_csv(self._cfg['data']['processed_path'] + self._cfg['data']['processed_filename'], index=self._cfg['data']['index'])
        self._log.info("Raw dataset saved as %s", self._cfg['data']['processed_path'])
    
    def _target_to_int(self):
        """Convert string labels in the target column to integer values."""
        if self._processed[self._cfg["data"]["target"]].dtype == 'object':
           unique_targets = self._processed[self._cfg["data"]["target"]].unique().tolist()
           unique_targets.sort()
           self._processed[self._cfg["data"]["new_target_col"]] = self._processed[self._cfg["data"]["target"]].map(lambda x: unique_targets.index(x))
           if self._cfg["data"]["drop_old_target"]:
               self._processed = self._processed.drop(columns=self._cfg["data"]["target"])

    def process(self):
        """Run the full data pipeline: load, save, process, and encode the target."""
        self._load_raw()
        self._save_raw()
        self._load_processed()
        self._target_to_int()
        self._save_processed()

dataset = Dataset()
dataset.process()

