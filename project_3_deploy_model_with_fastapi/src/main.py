import logging
from fastapi import FastAPI
from dto.person import Person
from ml import model
from joblib import load
from ml.train_model import CAT_FEATURES, MODEL_FILENAME, ENCODER_FILENAME, LB_FILENAME
from ml.data import process_data

logging.basicConfig(level=DEBUG)
app = FastAPI()
lr_model = load(MODEL_FILENAME)
encoder = load(ENCODER_FILENAME)
lb = load(LB_FILENAME)


@app.get("/")
async def root():
    return "API to predict if salary is greater or lower 50k based on census data"


@app.post("/predict_salary")
async def predict_salary(person: Person):
    input_data = process_data(person, CAT_FEATURES, label=None, training=False, encoder=encoder, lb=lb)
    pred = model.inference(lr_model, input_data)
    return pred
