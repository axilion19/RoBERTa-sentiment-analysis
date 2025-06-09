from fastapi import FastAPI
from pydantic import BaseModel
from app import SentimentAnalyzer

app = FastAPI(title="Movie Review Sentiment Analysis API", version="1.0")

analyzer = SentimentAnalyzer()

class RequestText(BaseModel):
    text: str

@app.get('/')
def index():
    return {'message': 'Movie Review Sentiment Analysis API'}

@app.post("/predict")
def predict(req: RequestText):
    result = analyzer.predict(req.text)[0]
    return {"review": result["review"], 
            "prediction": result["label"], 
            "score": result["score"]}
