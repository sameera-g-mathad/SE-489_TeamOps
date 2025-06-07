import pytest
import os
import pandas as pd
from utils import _RAW_PATH, _PROCESSED_PATH, _PRINT_END


def test_raw_dir_exists():
    """
    Test to check if the raw data directory exists.
    """

    print(f"Raw Path: {_RAW_PATH}", end=_PRINT_END)
    assert os.path.exists(_RAW_PATH)


def test_raw_data_files_exists():
    """
    Test to check if the raw data files exist.
    """
    _raw_file_path = os.path.join(_RAW_PATH, "raw_dataset.csv")

    print(f"Raw file Path: {_raw_file_path}", end=_PRINT_END)
    assert os.path.exists(_raw_file_path)


def test_processed_dir_exists():
    """
    Test to check if the processed data directory exists.
    """

    print(f"Processed Path: {_PROCESSED_PATH}", end=_PRINT_END)
    assert os.path.exists(_PROCESSED_PATH)


def test_processed_data_files_exists():
    """
    Test to check if the processed data files exist.
    """
    _processed_file_path = os.path.join(_PROCESSED_PATH, "processed_dataset.csv")

    print(f"Processed file Path: {_processed_file_path}", end=_PRINT_END)
    assert os.path.exists(_processed_file_path)


@pytest.fixture()
def _raw_dataset():
    """
    Function to load the raw dataset.
    """
    return pd.read_csv(os.path.join(_RAW_PATH, "raw_dataset.csv"))


@pytest.fixture()
def _processed_dataset():
    """
    Function to load the processed dataset.
    """
    return pd.read_csv(os.path.join(_PROCESSED_PATH, "processed_dataset.csv"))


def test_typeof_raw_dataset(_raw_dataset):
    """
    Test to check if the raw dataset is of type pd.DataFrame.
    """
    result = isinstance(_raw_dataset, pd.DataFrame)

    print(f"Is type of Raw Dataset: pd.DataFrame? {result}.", end=_PRINT_END)
    assert result


def test_typeof_processed_dataset(_processed_dataset):
    """
    Test to check if the processed dataset is of type pd.DataFrame.
    """
    result = isinstance(_processed_dataset, pd.DataFrame)
    print(f"Is type of Processed Dataset: pd.DataFrame? {result}.", end=_PRINT_END)
    assert result


def test_shape_processed_dataset(_processed_dataset):
    """
    Test to check if the processed dataset has the expected shape.
    We have a size of 50_000 samples filtered from the original data.
    Also, we have included an additional column for using huggingface label_2_id
    """
    _EXPECTED_ROWS = 50_000
    _EXPECTED_COLS = 3
    rows, cols = _processed_dataset.shape
    print(
        f"Shape match? Expected: ({_EXPECTED_ROWS}, {_EXPECTED_COLS}), Actual: ({rows}, {cols})",
        end=_PRINT_END,
    )
    assert rows == _EXPECTED_ROWS and cols == _EXPECTED_COLS


def test_dtypes_processed_dataset(_processed_dataset):
    """
    Test to check if the processed dataset has the expected data types.
    The expected data types are:
    - 'object' for the 'text' column
    - 'object' for the 'label' column
    - 'int64' for the 'label_id' column
    """
    _EXPECTED = ["object", "object", "int64"]
    for _index, column in enumerate(_processed_dataset.columns):
        _dtype = _processed_dataset[column].dtype
        _expected = _EXPECTED[_index]
        print(
            f"'{column}' Data type: Expected: {_expected}, Actual: {_dtype}",
        )
        assert _dtype == _expected
