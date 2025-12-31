import os
import time
import sys
import logging
import vertexai
from vertexai.generative_models import GenerativeModel
from ddtrace import tracer
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("usher-in-llm")

class LLMClient:
    def __init__(self):
        project_id = os.getenv("PROJECT_ID", "llm-observability-hackathon")
        location = "us-central1"

        try:
            vertexai.init(project=project_id, location=location)
            # Use the cutting-edge 2.5 Pro model
            self.model = GenerativeModel("gemini-2.5-pro")
            logger.info(f"✅ Vertex AI initialized with Gemini 2.5 PRO")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Vertex AI: {e}")
            raise e

    def ask_gemini(self, prompt: str):
        # Create a custom span for the LLM request
        with tracer.trace("llm.generate", service="usher-in-llm", resource="gemini-2.5-pro") as span:
            # Add custom tags for observability analysis
            span.set_tag("llm.model", "gemini-2.5-pro")
            span.set_tag("llm.prompt_length", len(prompt))
            
            start_time = time.perf_counter()

            try:
                response = self.model.generate_content(prompt)
                
                # Capture usage metadata (tokens)
                usage = response.usage_metadata
                span.set_tag("llm.prompt_tokens", usage.prompt_token_count)
                span.set_tag("llm.completion_tokens", usage.candidates_token_count)
                span.set_tag("llm.total_tokens", usage.total_token_count)
                
                # Record response length
                span.set_tag("llm.response_length", len(response.text))
                
            except Exception as e:
                # Capture the exception in the trace so it turns RED
                span.set_exc_info(*sys.exc_info())
                logger.error(f"LLM Generation Error: {e}")
                raise e

            end_time = time.perf_counter()
            latency = round(end_time - start_time, 3)
            
            # Tag the span with the final latency
            span.set_tag("llm.latency_seconds", latency)

            return {
                "text": response.text,
                "latency_seconds": latency,
                "model": "gemini-2.5-pro"
            }