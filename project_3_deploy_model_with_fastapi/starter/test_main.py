from fastapi.testclient import TestClient
from main import app
from starter.dto.person import Person

client = TestClient(app)
def test_welcome_msg():
  resp = client.get('/')
  assert resp.status_code == 200
  assert resp.json() == 'API to predict if salary is greater or lower 50k based on census data'

def test_lower_50k():
  data = Person()
  resp = client.post('/predict_salary', data=data)
  assert response.status_code == 200
  assert response.json() == "todo"