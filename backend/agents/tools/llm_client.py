# agents/tools/llm_client.py
import httpx
import json
import logging
from typing import Dict, Any, Optional
from core.config import settings
from utils.tokens import estimate_tokens, estimate_cost

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.api_key = settings.LLM_API_KEY
        self.endpoint_base = settings.LLM_ENDPOINT_BASE
        self.model = settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.top_p = settings.LLM_TOP_P
        self.max_tokens = settings.LLM_MAX_TOKENS
        
        # Debug logging
        logger.info(f"=== LLMClient Initialization ===")
        logger.info(f"Provider: {self.provider}")
        logger.info(f"Endpoint: {self.endpoint_base}")
        logger.info(f"Model: {self.model}")
        logger.info(f"API Key length: {len(self.api_key) if self.api_key else 0}")
        logger.info(f"API Key prefix: {self.api_key[:10] if self.api_key else 'None'}...")
    
    def send(self, prompt: str, params: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Send prompt to LLM and return response"""
        
        logger.info(f"=== Sending LLM Request ===")
        logger.debug(f"Prompt: {prompt[:100]}...")
        
        # Merge params with defaults
        temperature = params.get("temperature", self.temperature)
        top_p = params.get("top_p", self.top_p)
        max_tokens = params.get("max_tokens", self.max_tokens)
        model = params.get("model", self.model)
        
        # Prepare request based on provider
        if self.provider in ["openrouter", "openai"]:
            url = f"{self.endpoint_base}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            if self.provider == "openrouter":
                headers["HTTP-Referer"] = "http://localhost:8000"
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens
            }
            
            logger.info(f"URL: {url}")
            logger.info(f"Model: {model}")
            logger.debug(f"Headers: {headers}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)[:500]}...")
        else:
            url = self.endpoint_base
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {"prompt": prompt, "params": params}
        
        try:
            logger.info("Sending HTTP request...")
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
                
                logger.info(f"Response Status: {response.status_code}")
                logger.debug(f"Response Headers: {response.headers}")
                
                if response.status_code != 200:
                    logger.error(f"Non-200 response: {response.text}")
                
                response.raise_for_status()
                data = response.json()
                logger.debug(f"Response data: {json.dumps(data, indent=2)[:500]}...")
            
            # Parse response based on provider
            if self.provider in ["openrouter", "openai"]:
                answer_text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", estimate_tokens(prompt))
                completion_tokens = usage.get("completion_tokens", estimate_tokens(answer_text))
            else:
                answer_text = data.get("text", "")
                prompt_tokens = estimate_tokens(prompt)
                completion_tokens = estimate_tokens(answer_text)
            
            total_tokens = prompt_tokens + completion_tokens
            est_cost_usd = estimate_cost(prompt_tokens, completion_tokens, model)
            
            logger.info(f"Success! Tokens: {total_tokens}, Cost: ${est_cost_usd:.4f}")
            
            return {
                "answer_text": answer_text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "est_cost_usd": est_cost_usd,
                "confidence": None,
                "llm_pred_label": None
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP Status Error: {e}")
            logger.error(f"Response Status: {e.response.status_code}")
            logger.error(f"Response Body: {e.response.text}")
            
            error_detail = "Unknown error"
            try:
                error_json = e.response.json()
                error_detail = error_json.get("error", {}).get("message", str(error_json))
            except:
                error_detail = e.response.text
            
            return {
                "answer_text": f"OpenAI API Error {e.response.status_code}: {error_detail}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "est_cost_usd": 0.0,
                "confidence": None,
                "llm_pred_label": None
            }
        except Exception as e:
            logger.error(f"Unexpected error: {type(e).__name__}: {str(e)}", exc_info=True)
            return {
                "answer_text": f"Error: {type(e).__name__}: {str(e)}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "est_cost_usd": 0.0,
                "confidence": None,
                "llm_pred_label": None
            }