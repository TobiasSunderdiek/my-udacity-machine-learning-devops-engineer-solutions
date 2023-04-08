from fastapi.testclient import TestClient
from main import app
from starter.dto.person import Person

client = TestClient(app)

def test_welcome_msg():
  resp = client.get('/')
  assert resp.status_code == 200
  assert resp.json() == 'API to predict if salary is greater or lower 50k based on census data'

def test_lower_50k():
  person = Person()
  resp = client.post('/predict_salary', data=person)
  assert resp.status_code == 200
  assert resp.json() == "<=50k"

def test_greater_50k():
  person = Person(age=39,
                workclass='State-gov',
                fnlgt=77516,
                education='Bachelors',
                education_num=13,
                marital_status='Never-married',
                occupation='Adm-clerical',
                relationship='Not-in-family',
                race='White',
                sex='Male',
                capital_gain=2174,
                capital_loss=0,
                hours_per_week=40,
                native_country='United-States',
                salary='<=50K')

  resp = client.post('/predict_salary', data=person)
  assert resp.status_code == 200
  assert resp.json() == ">50k"