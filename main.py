import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": "Bearer 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"}

# Define the data structure for the request
class SimilarityRequest(BaseModel):
    model_answer: str
    student_answer: str

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": {
	"source_sentence": "That is a happy person",
	"sentences": [
		"That is a happy dog",
		"That is a very happy person",
		"Today is a sunny day"
	]
},
})

# print(output)

@app.get("/")
async def root():
    return {"message": "Welcome to the similarity calculator!"}

@app.post("/similarity")
async def calculate_similarity(request: SimilarityRequest):
    try:
        score = (
            query({
                "inputs": {
                    "source_sentence": request.model_answer,
                    "sentences": [request.student_answer]
                }
            })
        )
        return {"similarity_score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))