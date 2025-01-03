# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import spacy

app = FastAPI()

# Configure CORS (allow all origins for simplicity)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # which origins are allowed
    allow_credentials=True,      # whether to allow credentials (cookies, etc.)
    allow_methods=["*"],         # which HTTP methods are allowed
    allow_headers=["*"]          # which HTTP headers are allowed
)

# Load spaCy's German model (e.g. "de_core_news_sm")
nlp = spacy.load("de_core_news_sm")

@app.post("/tokenize-sentences")
async def tokenize_sentences(text: str):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return {"sentences": sentences}
