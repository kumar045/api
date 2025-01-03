# server.py
from fastapi import FastAPI
import spacy

app = FastAPI()

# Use spaCy's German model (e.g. "de_core_news_sm")
nlp = spacy.load("de_core_news_sm")

@app.post("/tokenize-sentences")
async def tokenize_sentences(text: str):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return {"sentences": sentences}
