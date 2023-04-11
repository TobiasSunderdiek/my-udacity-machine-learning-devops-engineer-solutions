import logging
import pandas as pd
from joblib import load
from fastapi import FastAPI
from project_3_deploy_model_with_fastapi.src.dto.person import Person
from project_3_deploy_model_with_fastapi.src.ml import model
from project_3_deploy_model_with_fastapi.src.ml.train_model import CAT_FEATURES
from project_3_deploy_model_with_fastapi.src.ml.model import MODEL_FILENAME, ENCODER_FILENAME, LB_FILENAME
from project_3_deploy_model_with_fastapi.src.ml.data import process_data

logging.basicConfig(level=logging.DEBUG)
app = FastAPI()
lr_model = load(MODEL_FILENAME)
encoder = load(ENCODER_FILENAME)
lb = load(LB_FILENAME)


@app.get("/")
async def root():
    return "Predict if salary is greater or lower 50k based on census data"


@app.post("/predict_salary")
async def predict_salary(person: Person):
    logging.info(f"API call to /predict_salary with {person}")
    # convert person to pandas dataframe, credits to https://stackoverflow.com/a/17840195
    person_df = pd.DataFrame(person.dict(by_alias=True), index=[0])
    input_data, _, _, _ = process_data(person_df, CAT_FEATURES, label=None, training=False, encoder=encoder, lb=lb)
    pred = model.inference(lr_model, input_data)
    pred_class = lb.inverse_transform(pred)[0]
    logging.info(f"Prediction: {pred} class: {pred_class}")
    resp = {'predicted_salary': pred_class}
    logging.info(f"Response with {resp}")
    return resp
