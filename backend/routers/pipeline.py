# Pipeline router: unigraph/gnn/translator/rag
# routers/pipeline.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any

from core.dependencies import get_db
from core.security import get_api_key
from models.run import Run
from models.enums import Status
from agents.tools.unigraph_adapter import UniGraphAdapter
from agents.tools.gnn_adapter import GNNAdapter
from agents.tools.graph_translator import GraphTranslator
from agents.agent_manager import AgentManager

router = APIRouter()

@router.post("/{run_id}/unigraph")
def run_unigraph(
    run_id: str,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key)
) -> Dict[str, Any]:
    """Run UniGraph normalization"""
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    adapter = UniGraphAdapter()
    results = adapter.process_graph(db, run.graph_id, [])
    
    run.status = Status.RUNNING
    db.commit()
    
    return {
        "status": "completed",
        "stage": "unigraph",
        **results
    }

@router.post("/{run_id}/gnn-adapter")
def run_gnn_adapter(
    run_id: str,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key)
) -> Dict[str, Any]:
    """Run GNN adapter"""
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    adapter = GNNAdapter()
    results = adapter.compute_features(db, run.graph_id, run_id)
    
    return {
        "status": "completed",
        "stage": "gnn_adapter",
        **results
    }

@router.post("/{run_id}/translator")
def run_translator(
    run_id: str,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key)
) -> Dict[str, Any]:
    """Run GraphTranslator"""
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    translator = GraphTranslator()
    results = translator.create_alignment(db, run.graph_id, run_id)
    
    return {
        "status": "completed",
        "stage": "translator",
        **results
    }

@router.post("/{run_id}/rag/prepare")
def prepare_rag(
    run_id: str,
    top_k: int = 5,
    hops: int = 2,
    prompt_template_id: str = "default_rag",
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key)
) -> Dict[str, Any]:
    """Prepare RAG context"""
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    manager = AgentManager(db)
    
    # Sample query for preview
    sample_query = "What is this graph about?"
    results = manager.prepare_rag_context(
        graph_id=run.graph_id,
        query_text=sample_query,
        top_k=top_k,
        hops=hops
    )
    
    return {
        "status": "ready",
        "stage": "rag",
        "context_preview": results["context"][:500],
        "tokens_in_context": results["tokens_in_context"],
        "top_k": top_k,
        "hops": hops,
        "prompt_template_id": prompt_template_id
    }