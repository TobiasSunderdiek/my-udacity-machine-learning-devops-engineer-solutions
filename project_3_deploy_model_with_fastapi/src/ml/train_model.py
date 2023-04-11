# Script to train machine learning model.
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from ..ml.data import process_data
from ..ml.model import train_model, calc_metrics, save_model


logging.basicConfig(level=logging.DEBUG)

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

DATA_FILE = 'project_3_deploy_model_with_fastapi/src/data/census.csv'


# Train and save a model.
def train_and_save_model():
    # fetch raw data
    data = pd.read_csv(DATA_FILE)

    # split data
    # Optional enhancement, use K-fold cross validation
    # instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20,
                                   stratify=data['salary'])
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary",
        training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label='salary',
        training=False, encoder=encoder, lb=lb)

    # train model
    lr_model = train_model(X_train, y_train)

    # save model and metrics
    y_pred = lr_model.predict(X_test)
    calc_metrics(CAT_FEATURES, lr_model, y_test, y_pred, test, encoder, lb)
    save_model(lr_model, encoder, lb)


if __name__ == '__main__':
    train_and_save_model()
