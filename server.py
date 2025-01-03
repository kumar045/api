# server.py
from fastapi import FastAPI, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import spacy

app = FastAPI()

# Optional: CORS settings if you plan to call this API from a different domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact origins instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load the spaCy German model (make sure you have installed it via: python -m spacy download de_core_news_sm)
nlp = spacy.load("de_core_news_sm")

class TextPayload(BaseModel):
    text: str

@app.post("/tokenize-sentences")
async def tokenize_sentences(payload: TextPayload):
    # Now you can access payload.text
    doc = nlp(payload.text)
    return {"sentences": [sent.text for sent in doc.sents]}
