from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("../model/model.pkl")
vectorizer = joblib.load("../model/vectorizer.pkl")

class NewsRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Fake News Detection API"}

@app.post("/predict")
def predict_news(request: NewsRequest):

    text = request.text
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        result = "Fake News"
    else:
        result = "Real News"

    return {"prediction": result}