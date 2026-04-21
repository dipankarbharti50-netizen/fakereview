from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import pickle
import re
import string

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved files
model = load_model("fake_review_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Constants (same as training)
max_len = 100

# FastAPI app
app = FastAPI()

# Input schema
class ReviewInput(BaseModel):
    review: str
    rating: float

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Prediction API
@app.post("/predict")
def predict(data: ReviewInput):
    
    review = clean_text(data.review)
    
    seq = tokenizer.texts_to_sequences([review])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    
    rating_scaled = scaler.transform([[data.rating]])
    
    pred = model.predict([pad, rating_scaled])[0][0]
    
    result = "Fake Review ❌" if pred > 0.5 else "Genuine Review ✅"
    
    return {
        "prediction": result,
        "confidence": float(pred)
    }