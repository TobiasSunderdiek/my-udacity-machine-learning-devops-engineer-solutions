import numpy as np
from unittest.mock import Mock, patch
from project_3_deploy_model_with_fastapi.src.ml.model import train_model, inference, save_model, MODEL_FILENAME, ENCODER_FILENAME, LB_FILENAME

@patch('project_3_deploy_model_with_fastapi.src.ml.model.LogisticRegression')
def test_train_model(MockLogisticRegression):
    lr_model_mock = MockLogisticRegression.return_value
    X_mock = Mock()
    y_mock = Mock()
    model = train_model(X_mock, y_mock)
    lr_model_mock.fit.assert_called_with(X_mock, y_mock)
    assert model is lr_model_mock

def test_inference():
    model_mock = Mock()
    X_mock = Mock()
    pred = inference(model_mock, X_mock)
    assert pred is not None
    model_mock.predict.assert_called_with(X_mock)

@patch('project_3_deploy_model_with_fastapi.src.ml.model.dump')
def test_save_model(dump_func_mock):
    lr_model_mock = Mock()
    encoder_mock = Mock()
    lb_mock = Mock()
    save_model(lr_model_mock, encoder_mock, lb_mock)
    dump_func_mock.assert_any_call(lr_model_mock, MODEL_FILENAME)
    dump_func_mock.assert_any_call(encoder_mock, ENCODER_FILENAME)
    # ensure last call was saving lb
    dump_func_mock.assert_called_with(lb_mock, LB_FILENAME)