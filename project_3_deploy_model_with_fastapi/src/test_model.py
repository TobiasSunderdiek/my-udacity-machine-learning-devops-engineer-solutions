import numpy as np
from unittest.mock import Mock, patch
from ml.model import train_model, inference

@patch('ml.model.LogisticRegression')
def test_train_model(MockLogisticRegression):
    lr_mock = MockLogisticRegression.return_value
    X_mock = Mock()
    y_mock = Mock()
    model = train_model(X_mock, y_mock)
    lr_mock.fit.assert_called_with(X_mock, y_mock)
    assert model is lr_mock

def test_inference():
    model_mock = Mock()
    X_mock = Mock()
    pred = inference(model_mock, X_mock)
    assert pred is not None
    model_mock.predict.assert_called_with(X_mock)
