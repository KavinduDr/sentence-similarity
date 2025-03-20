from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
from preprocess import preprocess_text, calculate_features
from typing import List

# Initialize FastAPI app
app = FastAPI()

# Set up templates
templates = Jinja2Templates(directory="templates")

# Mount static files directory if needed
#app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the trained model and scaler
model = tf.keras.models.load_model("model/model.h5", custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def process_form(
    request: Request,
    correct_answer: str = Form(...),
    keywords: str = Form(...),
    student_answer: str = Form(...)
):
    # Process the input and extract features
    keywords_list = keywords.split(",")
    features = calculate_features(correct_answer, keywords_list, student_answer)
    features_df = pd.DataFrame([features], columns=['cosine', 'jaccard', 'wmd', 'levenshtein', 'wordnet', 'bleu'])
    scaled_features = scaler.transform(features_df)

    # Predict marks
    predicted_marks = model.predict(scaled_features)[0][0]

    return templates.TemplateResponse(
        "result.html", 
        {"request": request, "marks": round(predicted_marks, 2)}
    )

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
