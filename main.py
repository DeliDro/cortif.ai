from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from fastapi import Header, HTTPException
# from random import choice

import inference

app = FastAPI()
model_logreg = inference.Inferer(inference.MODELS.LOGISTIC_REGRESSION)
model_nn = inference.Inferer(inference.MODELS.NN)
model_tree = inference.Inferer(inference.MODELS.DECISION_TREE)

class JsonInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

API_KEY = "be549856564e06e4b73cf3a5fb5f14911a3c11972a4119f5f67692455b3b86ae"

@app.post("/api/v1/predict")
async def predict(
    data: JsonInput,
    key: str = Header(None, alias="key")
) -> Dict[str, str]:
    if not key:
        raise HTTPException(status_code=401, detail="Missing API Key in headers")

    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    features = inference.InputFeatures(**data.dict())
    result = {"species": model_logreg.predict(features)}
    return result



@app.get("/")
async def root():
    return {"message": "This is a test API"}

@app.get("/health-check")
async def health_check():
    return {"message": "This is a health check endpoint"}
