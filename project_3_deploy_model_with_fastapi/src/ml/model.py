import logging
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from joblib import dump
from .data import process_data


logging.basicConfig(level=logging.DEBUG)

SLICE_OUTPUT = 'project_3_deploy_model_with_fastapi/src/model/slice_output.txt'
MODEL_FILENAME = 'project_3_deploy_model_with_fastapi/src/model/lr_model.joblib'
ENCODER_FILENAME = 'project_3_deploy_model_with_fastapi/src/model/encoder.joblib'
LB_FILENAME = 'project_3_deploy_model_with_fastapi/src/model/lb.joblib'


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    logging.info("Training model finished")
    return lr


def calc_metrics(cat_features, model, y_test, y_pred, test_data, encoder, lb):
    """
    Calculates metrics of the model

    Inputs
    ------
    cat_features : list
        List of categorical feature names.
    model : sklearn.linear_model.LogisticRegression
        Trained model.
    y_test : np.array
        Labels.
    y_pred : np.array
        Predicted Labels.
    test_data : pd.DataFrame
        Test data.
    encoder : sklearn.preprocessing.OneHotEncoder
        Fitted encoder for values of category features.
    lb : sklearn.preprocessing.LabelBinarizer
        Fitted label binarizer.
    Returns
    -------
    None
    """
    precision, recall, fbeta = _compute_model_metrics(y_test, y_pred)
    logging.info(f"Overall metrics: Precision: {precision}, \
                 Recall: {recall}, Fbeta: {fbeta}")

    metrics = []
    for cat in cat_features:
        for cat_variation in test_data[cat].unique():
            logging.debug(f"cat {cat}, cat_variation {cat_variation}")
            slice_df = test_data[test_data[cat] == cat_variation]
            X_slice, y_slice, _, _ = process_data(
                slice_df, categorical_features=cat_features,
                label='salary', training=False, encoder=encoder, lb=lb)
            y_slice_pred = model.predict(X_slice)
            precision, recall, fbeta = _compute_model_metrics(y_slice,
                                                              y_slice_pred)
            metrics.append(f"Category feature: {cat}, Category variation: {cat_variation}, Precision: {precision}, Recall: {recall}, Fbeta: {fbeta}")
    with open(SLICE_OUTPUT, 'w') as file:
        # Credits to Ravikiran A S for
        # transfering the list to a string, see here:
        # https://www.simplilearn.com/tutorials/python-tutorial/list-to-string-in-python
        file.write('\n'.join(metrics))
    logging.info(f"Metrics written to {SLICE_OUTPUT}")


def _compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model
    using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : LogisticRegression
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    logging.info(f"Run model inference with input: {X}")
    pred = model.predict(X)
    return pred


def save_model(lr_model, encoder, lb):
    """ Save model/encoder/label binarizer to file.

    Inputs
    ------
    lr_model : sklearn.linear_model.LogisticRegression
        Trained model.
    encoder : sklearn.preprocessing.OneHotEncoder
        Fitted encoder for values of category features.
    lb : sklearn.preprocessing.LabelBinarizer
        Fitted label binarizer.
    Returns
    -------
    None
    """
    dump(lr_model, MODEL_FILENAME)
    logging.info(f"Model saved to file {MODEL_FILENAME}.")
    dump(encoder, ENCODER_FILENAME)
    logging.info(f"Encoder saved to {ENCODER_FILENAME}.")
    dump(lb, LB_FILENAME)
    logging.info(f"Label binarizer saved to {LB_FILENAME}")
