from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

def test_welcome_msg():
  resp = client.get('/')
  assert resp.status_code == 200
  assert resp.json() == 'API to predict if salary is greater or lower 50k based on census data'

def test_lower_50k():
  person = {'age':39,
            'workclass': 'State-gov',
            'fnlgt':77516,
            'education':'Bachelors',
            'education-num':13,
            'marital-status':'Never-married',
            'occupation':'Adm-clerical',
            'relationship':'Not-in-family',
            'race':'White',
            'sex':'Male',
            'capital-gain':2174,
            'capital-loss':0,
            'hours-per-week':40,
            'native-country':'United-States'}
  resp = client.post('/predict_salary', json=person)
  assert resp.status_code == 200
  assert resp.json() == '<=50k'

def test_greater_50k():
  person = {'age':52,
            'workclass': 'Self-emp-not-inc',
            'fnlgt':209642,
            'education':'HS-grad',
            'education-num':9,
            'marital-status':'Married-civ-spouse',
            'occupation':'Exec-managerial',
            'relationship':'Husband',
            'race':'White',
            'sex':'Male',
            'capital-gain':0,
            'capital-loss':0,
            'hours-per-week':45,
            'native-country':'United-States'}

  resp = client.post('/predict_salary', data=person)
  assert resp.status_code == 200
  assert resp.json() == '>50k'