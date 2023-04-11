from fastapi.testclient import TestClient
from project_3_deploy_model_with_fastapi.src.main import app


client = TestClient(app)


def test_welcome_msg():
    resp = client.get('/')
    assert resp.status_code == 200
    assert resp.json() == 'Predict if salary is greater or lower 50k based on census data'


def test_lower_50k():
    person = {'age': 39,
              'workclass': 'State-gov',
              'fnlgt': 77516,
              'education': 'Bachelors',
              'education-num': 13,
              'marital-status': 'Never-married',
              'occupation': 'Adm-clerical',
              'relationship': 'Not-in-family',
              'race': 'White',
              'sex': 'Male',
              'capital-gain': 2174,
              'capital-loss': 0,
              'hours-per-week': 40,
              'native-country': 'United-States'}
    resp = client.post('/predict_salary', json=person)
    assert resp.status_code == 200
    assert resp.json() == {'predicted_salary': '<=50K'}


def test_greater_50k():
    person = {'age': 31,
              'workclass': 'Private',
              'fnlgt': 45781,
              'education': 'Masters',
              'education-num': 14,
              'marital-status': 'Never-married',
              'occupation': 'Prof-specialty',
              'relationship': 'Not-in-family',
              'race': 'White',
              'sex': 'Female',
              'capital-gain': 14084,
              'capital-loss': 0,
              'hours-per-week': 50,
              'native-country': 'United-States'}

    resp = client.post('/predict_salary', json=person)
    assert resp.status_code == 200
    assert resp.json() == {'predicted_salary': '>50K'}
