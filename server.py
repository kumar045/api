# server.py
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from typing import Optional
import asyncio
import openai
import spacy
import os

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

# Get API keys from environment variables
google_api_key = os.environ.get("GOOGLE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Configure API clients
if google_api_key:
    genai.configure(api_key=google_api_key)
    
if openai_api_key:
    openai.api_key = openai_api_key
    
OPENAI_SYSTEM_PROMPT = '''
Rolle
Du bist ein sorgfältiger Assistent, der medizinische Texte für 12-jährige maximal verständlich formuliert. Dein Fokus liegt auf einfacher Sprache, kurzen Sätzen und klarer Struktur.

Ersetze in folgendem Text alle Fachbegriffe akribisch genau durch leicht verständliche Begriffe und zerlege lange Wörter in kürzere:

- Medizinische Begriffe durch einfache deutsche Begriffe ersetzen: Falls möglich, verwende ein einfaches Wort („Diabetes“ → „Zuckerkrankheit“).
- Falls nicht, erkläre den Begriff in Klammern.
- Lange Wörter immer aufteilen („Bluthochdruck“ → „hoher Blutdruck“).

Beispiele:
- „Infektion der Atemwege“ → „Erkältung“
- „Brustschmerzen“ → „Schmerzen in der Brust“
- „Gebärmutterschleimhaut“ → „Die Haut in der Gebärmutter“
- „Gewichtszunahme“ → „Du wirst schwerer“
- „Leberentzündung“ → „Deine Leber ist krank“
- „Infektionsschutzmaßnahmenverordnung“ → „Regeln, die dich vor Krankheiten schützen“

Denk daran:
- Verwende als Anrede "Sie".
'''

class TextPayload(BaseModel):
    text: str

@app.post("/tokenize-sentences")
async def tokenize_sentences(payload: TextPayload):
    # Now you can access payload.text
    doc = nlp(payload.text)
    return {"sentences": [sent.text for sent in doc.sents]}

class PromptRequest(BaseModel):
    system_prompt: str

class ChainResponse(BaseModel):
    final_output: str

@app.post("/chain-models", response_model=ChainResponse)
async def chain_models(request: PromptRequest):
    try:
        # Validate API keys at runtime
        if not google_api_key or not openai_api_key:
            raise HTTPException(
                status_code=500, 
                detail="API keys not configured. Set GOOGLE_API_KEY and OPENAI_API_KEY environment variables."
            )
            
        # Step 1: Process with Gemini
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        gemini_response = await asyncio.to_thread(
            gemini_model.generate_content,
            request.system_prompt
        )
        
        gemini_output = gemini_response.text
        
        # Step 2: Pass Gemini's output to GPT-4o using synchronous client in a thread
        def call_openai():
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
                    {"role": "user", "content": gemini_output}
                ]
            )
            return response
            
        openai_response = await asyncio.to_thread(call_openai)
        
        final_output = openai_response.choices[0].message.content
        
        return ChainResponse(
            final_output=final_output
        )
        
    except Exception as e:
        # Include more detailed error information
        error_detail = str(e)
        import traceback
        error_traceback = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error: {error_detail}\n\nTraceback: {error_traceback}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
