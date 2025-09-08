# Results router: export CSV/JSONL, charts
# routers/results.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import Optional, List
import pandas as pd
import json

from core.dependencies import get_db
from core.security import get_api_key
from models.run import Query as QueryModel
from models.results import GenerationLog, RetrievalLog
from schemas.results import ChartRequest, ChartResponse
from utils.io import save_csv, save_jsonl
from utils.charts import save_chart

router = APIRouter()

@router.get("/{run_id}/table")
def get_results_table(
    run_id: str,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key)
):
    """Get results as table"""
    
    # Get all queries for run
    queries = db.query(QueryModel).filter(QueryModel.run_id == run_id).all()
    
    if not queries:
        return {"data": []}
    
    results = []
    for query in queries:
        # Get associated logs
        retrieval = db.query(RetrievalLog).filter(
            RetrievalLog.query_id == query.query_id
        ).first()
        
        generation = db.query(GenerationLog).filter(
            GenerationLog.query_id == query.query_id
        ).first()
        
        row = {
            "run_id": run_id,
            "query_id": query.query_id,
            "query_text": query.query_text,
            "k": retrieval.k if retrieval else None,
            "hops": retrieval.hops if retrieval else None,
            "tokens_in_context": retrieval.tokens_in_context if retrieval else None,
            "llm_answer": generation.llm_answer_text if generation else None,
            "total_tokens": generation.total_tokens if generation else None,
            "est_cost_usd": generation.est_cost_usd if generation else None
        }
        results.append(row)
    
    return {"data": results}

@router.get("/{run_id}/export.csv")
def export_csv(
    run_id: str,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key)
):
    """Export results as CSV"""
    
    # Get table data
    table_data = get_results_table(run_id, db)
    
    if not table_data["data"]:
        raise HTTPException(status_code=404, detail="No results found")
    
    # Create DataFrame and save
    df = pd.DataFrame(table_data["data"])
    filepath = f"data/outputs/{run_id}_results.csv"
    save_csv(df, filepath)
    
    return FileResponse(filepath, filename=f"{run_id}_results.csv")

@router.get("/{run_id}/export.jsonl")
def export_jsonl(
    run_id: str,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key)
):
    """Export results as JSONL"""
    
    # Get table data
    table_data = get_results_table(run_id, db)
    
    if not table_data["data"]:
        raise HTTPException(status_code=404, detail="No results found")
    
    # Save as JSONL
    filepath = f"data/outputs/{run_id}_results.jsonl"
    save_jsonl(table_data["data"], filepath)
    
    return FileResponse(filepath, filename=f"{run_id}_results.jsonl")

@router.post("/{run_id}/charts", response_model=ChartResponse)
def generate_charts(
    run_id: str,
    request: ChartRequest,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key)
):
    """Generate charts"""
    
    paths = []
    
    for chart_type in request.charts:
        # Generate sample data (would normally come from metrics)
        data = {
            "k_values": [1, 3, 5, 7, 10],
            "accuracies": [0.6, 0.7, 0.75, 0.78, 0.8],
            "context_sizes": [100, 200, 300, 400, 500],
            "token_counts": [150, 280, 420, 560, 700],
            "shots": [0, 1, 2, 3, 5],
            "f1_scores": [0.5, 0.65, 0.72, 0.76, 0.78]
        }
        
        path = save_chart(data, chart_type, run_id)
        paths.append(path)
    
    return ChartResponse(paths=paths)