from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class Data(BaseModel):
  feature_1: float
  feature_2: str


app = FastAPI(title='TestAPI Exercise chapter 3', description='You can post features of type Data', version='0.1.0')

@app.post("/ingest")
async def ingest_data(data: Data):
  if (data.feature_1 < 0):
    raise HTTPException(status_code=400, detail='feature_1 must be non-negative')
  if (len(data.feature_2) >= 280):
    raise HTTPException(status_code=400, detail='feature_2 must be <= 280 chars')
  return data
