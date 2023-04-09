# Script to train machine learning model.
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from data import process_data
from model import train_model, overall_and_slice_metrics, save_model

logging.basicConfig(level=logging.DEBUG)
data = pd.read_csv('src/data/census.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, stratify=data['salary'])

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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=CAT_FEATURES, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=CAT_FEATURES, label='salary', training=False, encoder=encoder, lb=lb)

# Train and save a model.
def train_and_save_model():
    lr_model = train_model(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    overall_and_slice_metrics(lr_model, y_test, y_pred, test, encoder, lb)
    save_model(lr_model, encoder, lb)

if __name__ == '__main__':
    train_and_save_model()
