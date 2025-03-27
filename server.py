# server.py
from fastapi import FastAPI, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
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

class TextPayload(BaseModel):
    text: str

@app.post("/tokenize-sentences")
async def tokenize_sentences(payload: TextPayload):
    # Now you can access payload.text
    doc = nlp(payload.text)
    return {"sentences": [sent.text for sent in doc.sents]}

class PromptRequest(BaseModel):
    system_prompt: str = "You are a helpful assistant."

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
        gemini_model = genai.GenerativeModel('gemini-2.0-pro')
        gemini_response = await asyncio.to_thread(
            gemini_model.generate_content,
            request.system_prompt
        )
        
        gemini_output = gemini_response.text
        
        # Step 2: Pass Gemini's output to GPT-4o
        openai_response = await openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert editor who improves and refines text."},
                {"role": "user", "content": gemini_output}
            ]
        )
        
        final_output = openai_response.choices[0].message.content
        
        return ChainResponse(
            final_output=final_output
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
