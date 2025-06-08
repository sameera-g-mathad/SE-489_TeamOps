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


def test_inference_script_path():
    """
    This function tests if the inference script file exists.
    """
    train_script_path = os.path.join(_TRAIN_PATH, "predict_model.py")
    print("Train Script Path: {train_script_path}", end=_PRINT_END)
    assert os.path.exists(train_script_path)


def test_method_train_present(_model):
    """
    This function tests if the predict method is present in the model instance.
    """
    assert hasattr(_model, "predict")


def test_inference_rtype(_model):
    """
    This function tests if the predict method returns a value.
    """
    prediction = _model.predict(
        """
    In this study, we demonstrate the possibility of the implementation of universal
    Gaussian computation on a two-node cluster state ensemble.
    We consider the phase-locked sub-Poissonian lasers,
    which radiate the bright light with squeezed quadrature,
    as the resource to generate these states.
    """
    )
    print("\n")
    print(f"Prediction: {prediction}", end=_PRINT_END)
    print(
        f'Type of Prediction: Expected: "str", Actual: {type(prediction)}',
        end=_PRINT_END,
    )
    assert isinstance(prediction, str), "Prediction should be of type str"
    print("\n")


def test_inference(_model):
    """
    This function tests the predict method of the model instance.
    It checks if the method returns a non-None value.
    """
    prediction = _model.predict(
        """
    We study Hermitian metrics with a Gauduchon connection being ""K\""ahler-like"",
    namely, satisfying the same symmetries for curvature as the Levi-Civita and Chern connections.
    In particular, we investigate $6$-dimensional solvmanifolds with invariant complex structures
    with trivial canonical bundle and with invariant Hermitian metrics.
    The results for this case give evidence for two conjectures that are
    expected to hold in more generality: first, if the Strominger-Bismut connection is
    K\""ahler-like, then the metric is pluriclosed; second, if another
    Gauduchon connection, different from Chern or Strominger-Bismut, is K\""ahler-like,
    then the metric is K\""ahler. As a further motivation,
    we show that the K\""ahler-like condition for the Levi-Civita connection
    assures that the Ricci flow preserves the Hermitian condition along
    analytic solutions.
                          """
    )
    print("\n")
    print(f"Prediction: {prediction}", end=_PRINT_END)
    assert prediction is not None, "Inference failed, model returned None"
    print("Inference test passed successfully.", end=_PRINT_END)
    print("\n")
