import numpy as np
from unittest.mock import Mock
from ml.model import inference


def test_inference():
    model_mock = Mock()
    X_mock = Mock()
    pred = inference(model_mock, X_mock)
    assert pred is not None
    model_mock.predict.assert_called_with(X_mock)
