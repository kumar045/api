# server.py
from fastapi import FastAPI, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from ai_sdk.google import google
from ai_sdk.openai import openai
from ai import generateText
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

google_api_key = os.environ.get("GOOGLE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Validate API keys are available
if not google_api_key or not openai_api_key:
    print("Warning: API keys not found in environment variables")
    print("Make sure to set GOOGLE_API_KEY and OPENAI_API_KEY")

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
    gemini_output: str
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
            
        # Step 1: Process with Gemini using only system prompt
        gemini_result = await generateText({
            "model": google("gemini-2.0-pro"),
            "system": request.system_prompt
        })
        
        gemini_output = gemini_result["text"]
        
        # Step 2: Pass Gemini's output to GPT-4o
        gpt_result = await generateText({
            "model": openai("gpt-4o"),
            "prompt": gemini_output,
            "system": "You are an expert editor who improves and refines text."
        })
        
        final_output = gpt_result["text"]
        
        return ChainResponse(
            gemini_output=gemini_output,
            final_output=final_output
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
