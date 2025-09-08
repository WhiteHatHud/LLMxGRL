# agents/tools/llm_client.py
import httpx
import json
from typing import Dict, Any, Optional
from core.config import settings
from utils.tokens import estimate_tokens, estimate_cost

class LLMClient:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.api_key = settings.LLM_API_KEY
        self.endpoint_base = settings.LLM_ENDPOINT_BASE
        self.model = settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.top_p = settings.LLM_TOP_P
        self.max_tokens = settings.LLM_MAX_TOKENS
    
    def send(self, prompt: str, params: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Send prompt to LLM and return response"""
        
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
        else:
            # Custom provider placeholder
            url = self.endpoint_base
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {"prompt": prompt, "params": params}
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            
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
            
            return {
                "answer_text": answer_text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "est_cost_usd": est_cost_usd,
                "confidence": None,  # Most APIs don't return confidence
                "llm_pred_label": None  # Extract if needed
            }
            
        except Exception as e:
            return {
                "answer_text": f"Error: {str(e)}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "est_cost_usd": 0.0,
                "confidence": None,
                "llm_pred_label": None
            }# Provider-agnostic HTTP client for LLMs
