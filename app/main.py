from ddtrace import patch_all
patch_all()

import os
import logging
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from app.llm import LLMClient

# 1. Load environment variables at the very top
load_dotenv()

# 2. Configure Datadog-friendly Logging
# This format allows Datadog to link your logs directly to your APM traces
FORMAT = '%(asctime)s %(levelname)s [%(name)s] [dd.service=%(dd.service)s dd.env=%(dd.env)s dd.version=%(dd.version)s dd.trace_id=%(dd.trace_id)s dd.span_id=%(dd.span_id)s] - %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("usher-in-api")
logger.setLevel(logging.INFO)

app = FastAPI(title="Usher-In Observability Hackathon")

# 3. Initialize the LLM Client
try:
    llm = LLMClient()
except Exception as e:
    logger.error(f"CRITICAL: LLM initialization failed: {e}")
    llm = None

@app.get("/")
def root():
    return {
        "message": "Usher-In API is live",
        "project": "llm-observability-hackathon",
        "endpoints": ["/generate", "/check-config"]
    }

@app.post("/generate")
async def generate(prompt: str):
    """
    The core endpoint. Sends a prompt to Gemini and returns 
    the response along with observability metrics.
    """
    if not llm:
        raise HTTPException(
            status_code=500, 
            detail="LLM Client not initialized. Check server logs."
        )
    
    if not prompt or len(prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    logger.info(f"Received generation request. Prompt: {prompt[:50]}...")

    try:
        # result contains {"text": ..., "latency_seconds": ..., "model": ..., "usage": ...}
        result = llm.ask_gemini(prompt)
        
        return {
            "status": "success",
            "data": {
                "response": result["text"],
                "metrics": {
                    "latency": f"{result['latency_seconds']}s",
                    "model_used": result["model"]
                }
            }
        }
    except Exception as e:
        logger.error(f"Error in /generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check-config")
def check_config():
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    return {
        "target_project": "llm-observability-hackathon",
        "env_project_id": os.getenv("PROJECT_ID"),
        "creds_path": path,
        "key_file_exists": os.path.exists(path) if path else False,
        "client_initialized": llm is not None
    }