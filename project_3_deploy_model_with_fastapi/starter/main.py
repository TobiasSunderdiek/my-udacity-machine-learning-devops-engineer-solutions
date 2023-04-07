from fastapi import FastAPI
from dto.person import Person
from starter.ml import model

app = FastAPI()


@app.get("/")
def root():
    return "API to predict if salary is greater or lower 50k based on census data"


@app.post("/predict_salary")
def predict_salary(person: Person):
    pass
    #model.inference
