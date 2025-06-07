import pytest
import os
from team_ops import Model
from utils import _PRINT_END, _SAVE_PATH, _TRAIN_PATH


@pytest.fixture
def _model():
    """
    This fixture creates an instance of the_ Model class.
    It is used to test the train method of the_ Model class.
    The Model class is expected to be defined in the team_ops package.
    """
    return Model()


def test_train_path():
    """
    This function tests if the training path exists.
    """
    print(f"Train Path: {_TRAIN_PATH}", end=_PRINT_END)
    assert os.path.exists(_TRAIN_PATH)


def test_save_path():
    """
    This function tests if the save path exists.
    """
    print(f"Save Path: {_SAVE_PATH}", end=_PRINT_END)
    assert os.path.exists(_SAVE_PATH)


def test_model_def_file():
    """
    This function tests if the model definition file exists.
    """
    model_def_file = os.path.join(_TRAIN_PATH, "model.py")
    print(f"Definition File Path: {model_def_file}", end=_PRINT_END)
    assert os.path.exists(model_def_file)


def test_train_script_path():
    """
    This function tests if the training script file exists.
    """
    train_script_path = os.path.join(_TRAIN_PATH, "train_model.py")
    print("Train Script Path: {train_script_path}", end=_PRINT_END)
    assert os.path.exists(train_script_path)


def test_method_train_present(_model):
    """
    This function tests if the train method is present in the model instance.
    """
    assert hasattr(_model, "train")


def test_train(_model):
    """
    This function tests the train method of the model instance.
    By default, if there is a model in saved path, it means that the model
    is already trained.
    Alternatively, if the model needs to be forced to train, one can
    pass the `force_train=True` argument in the config file.
    """
    try:
        _model.train()
    except Exception as e:
        pytest.fail(f"Train method failed due to following reason:\n{e}")
