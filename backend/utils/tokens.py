# utils/tokens.py
def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token"""
    if not text:
        return 0
    return len(text) // 4

def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str = "llama") -> float:
    """Rough cost estimation in USD"""
    # Placeholder rates (per 1M tokens)
    rates = {
        "llama": {"prompt": 0.20, "completion": 0.25},
        "gpt": {"prompt": 3.00, "completion": 6.00},
    }
    
    base = "llama" if "llama" in model.lower() else "gpt"
    rate = rates.get(base, rates["llama"])
    
    cost = (prompt_tokens * rate["prompt"] + completion_tokens * rate["completion"]) / 1_000_000
    return round(cost, 6)# Rough token estimates
