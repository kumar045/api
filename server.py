from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import spacy

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

nlp = spacy.load("de_core_news_sm")

@app.post("/tokenize-sentences")
async def tokenize_sentences(text: str = Form(...)):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return {"sentences": sentences}
