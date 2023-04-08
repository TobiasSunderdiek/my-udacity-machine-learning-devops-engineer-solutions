# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model
from joblib import dump

data = pd.read_csv('starter/data/census.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, stratify=data['salary'])

cat_features = [
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
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb)

# Train and save a model.
def train_and_save_model():
    lr_model = train_model(X_train, y_train)
    dump(lr_model, 'starter/model/lr_model.joblib')

if __name__ == '__main__':
    train_and_save_model()