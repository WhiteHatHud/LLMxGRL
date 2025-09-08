# Query router: ask LLM
# routers/query.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
import json
from datetime import datetime

from core.dependencies import get_db
from core.security import get_api_key
from models.run import Run, Query as QueryModel
from models.results import RetrievalLog, GenerationLog
from schemas.llm import QueryRequest, QueryResponse
from agents.agent_manager import AgentManager
from utils.ids import generate_query_id

router = APIRouter()

@router.post("/{run_id}", response_model=QueryResponse)
def query_llm(
    run_id: str,
    request: QueryRequest,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key)
):
    """Query the LLM with graph context"""
    
    # Get run
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Generate query ID
    query_id = generate_query_id()
    
    # Save query
    query = QueryModel(
        run_id=run_id,
        query_id=query_id,
        query_text=request.query_text
    )
    db.add(query)
    
    # Prepare RAG context
    manager = AgentManager(db)
    rag_results = manager.prepare_rag_context(
        graph_id=run.graph_id,
        query_text=request.query_text,
        top_k=request.top_k,
        hops=request.hops
    )
    
    # Save retrieval log
    retrieval_log = RetrievalLog(
        run_id=run_id,
        query_id=query_id,
        k=request.top_k,
        hops=request.hops,
        prompt_template_id="default_rag",
        tokens_in_context=rag_results["tokens_in_context"],
        context_preview=rag_results["context"][:500]
    )
    db.add(retrieval_log)
    
    # Generate response
    gen_results = manager.generate_response(
        query_text=request.query_text,
        context=rag_results["context"],
        prompt_template_id="default_rag"
    )
    
    # Save generation log
    generation_log = GenerationLog(
        run_id=run_id,
        query_id=query_id,
        llm_name="default",
        llm_params_json=json.dumps({"temperature": 0.2, "top_p": 0.9}),
        prompt_tokens=gen_results.get("prompt_tokens", 0),
        completion_tokens=gen_results.get("completion_tokens", 0),
        total_tokens=gen_results.get("total_tokens", 0),
        est_cost_usd=gen_results.get("est_cost_usd", 0.0),
        llm_answer_text=gen_results.get("answer_text", ""),
        llm_pred_label=gen_results.get("llm_pred_label"),
        confidence=gen_results.get("confidence")
    )
    db.add(generation_log)
    db.commit()
    
    # Calculate total time
    total_ms = rag_results["retrieval_ms"] + gen_results["generation_ms"]
    
    return QueryResponse(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        graph_id=run.graph_id,
        task_type=run.task_type.value,
        query_id=query_id,
        query_text=request.query_text,
        retrieved_context=rag_results["context"],
        tokens_in_context=rag_results["tokens_in_context"],
        k=request.top_k,
        hops=request.hops,
        prompt_template_id="default_rag",
        llm_name="default",
        llm_params={"temperature": 0.2, "top_p": 0.9},
        prompt_tokens=gen_results.get("prompt_tokens"),
        completion_tokens=gen_results.get("completion_tokens"),
        total_tokens=gen_results.get("total_tokens"),
        llm_answer_text=gen_results.get("answer_text", ""),
        llm_pred_label=gen_results.get("llm_pred_label"),
        gold=None,
        is_exact_match=None,
        accuracy=None,
        hits_at_k=None,
        macro_f1=None,
        rouge=None,
        bertscore=None,
        retrieval_ms=rag_results["retrieval_ms"],
        generation_ms=gen_results["generation_ms"],
        total_ms=total_ms,
        est_cost_usd=gen_results.get("est_cost_usd"),
        confidence=gen_results.get("confidence")
    )