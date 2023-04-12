from unittest.mock import Mock, patch
import pandas as pd
from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from project_3_deploy_model_with_fastapi.src.ml.model import train_model, \
    inference, save_model, _compute_model_metrics, MODEL_FILENAME, ENCODER_FILENAME, LB_FILENAME
from project_3_deploy_model_with_fastapi.src.ml.train_model import DATA_FILE, CAT_FEATURES
from project_3_deploy_model_with_fastapi.src.ml.data import process_data


@patch('project_3_deploy_model_with_fastapi.src.ml.model.LogisticRegression')
def test_train_model(MockLogisticRegression):
    lr_model_mock = MockLogisticRegression.return_value
    X_mock = Mock()
    y_mock = Mock()
    model = train_model(X_mock, y_mock)
    lr_model_mock.fit.assert_called_with(X_mock, y_mock)
    assert model is lr_model_mock

def test_train_model_return_type():
    data = pd.read_csv(DATA_FILE)
    train, _ = train_test_split(data, test_size=0.20,
                                   stratify=data['salary'])
    X_train, y_train, _, _ = process_data(
        train, categorical_features=CAT_FEATURES, label="salary",
        training=True
    )
    lr_model = train_model(X_train,y_train)
    assert isinstance(lr_model, LogisticRegression)


def test_inference():
    model_mock = Mock()
    X_mock = Mock()
    pred = inference(model_mock, X_mock)
    assert pred is not None
    model_mock.predict.assert_called_with(X_mock)

def test_inference_return_type():
    data = pd.read_csv(DATA_FILE)
    train, test = train_test_split(data, test_size=0.20,
                                   stratify=data['salary'])

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary",
        training=True
    )
    X_test, _, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label='salary',
        training=False, encoder=encoder, lb=lb)
    lr_model = train_model(X_train,y_train)
    pred = inference(lr_model, X_test)
    assert isinstance(pred, ndarray)


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

def test_compute_model_metrics_return_type():
    data = pd.read_csv(DATA_FILE)
    train, test = train_test_split(data, test_size=0.20,
                                   stratify=data['salary'])

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary",
        training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label='salary',
        training=False, encoder=encoder, lb=lb)
    lr_model = train_model(X_train,y_train)
    pred = inference(lr_model, X_test)
    precision, recall, fbeta = _compute_model_metrics(y_test, pred)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
