# schemas/llm.py
from pydantic import BaseModel
from typing import Optional, Any
from schemas.base import BaseSchema

class QueryRequest(BaseModel):
    query_text: str
    top_k: int = 5
    hops: int = 2
    llm_config_id: Optional[int] = None

class QueryResponse(BaseSchema):
    # Meta
    run_id: str
    timestamp: str
    graph_id: str
    task_type: str
    query_id: str
    
    # Query
    query_text: str
    retrieved_context: str
    
    # Context stats
    tokens_in_context: int
    k: int
    hops: int
    prompt_template_id: str
    
    # LLM setup
    llm_name: str
    llm_params: dict
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    
    # Outputs
    llm_answer_text: str
    llm_pred_label: Optional[str]
    
    # Gold
    gold: Optional[str]
    
    # Performance
    is_exact_match: Optional[bool]
    accuracy: Optional[float]
    hits_at_k: Optional[float]
    macro_f1: Optional[float]
    rouge: Optional[float]
    bertscore: Optional[float]
    
    # Runtime/cost
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    est_cost_usd: Optional[float]
    
    # Confidence
    confidence: Optional[float]