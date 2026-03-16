import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

class NewsRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Fake News Detection API"}

@app.post("/predict")
def predict_news(request: NewsRequest):

    vector = vectorizer.transform([request.text])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        return {"prediction": "Fake News"}
    else:
        return {"prediction": "Real News"}