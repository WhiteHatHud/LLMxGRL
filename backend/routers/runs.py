# routers/runs.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional

from core.dependencies import get_db
from core.security import get_api_key
from models.run import Run
from schemas.run import RunCreate, RunResponse
from utils.ids import generate_run_id

router = APIRouter()

@router.post("", response_model=RunResponse)
def create_run(
    run_data: RunCreate,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key)
):
    """Create a new experiment run"""
    run_id = generate_run_id()
    
    run = Run(
        run_id=run_id,
        graph_id=run_data.graph_id,
        task_type=run_data.task_type
    )
    
    db.add(run)
    db.commit()
    db.refresh(run)
    
    return RunResponse(
        run_id=run.run_id,
        timestamp=run.timestamp,
        graph_id=run.graph_id,
        task_type=run.task_type.value,
        status=run.status.value
    )

@router.get("/{run_id}", response_model=RunResponse)
def get_run(
    run_id: str,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key)
):
    """Get run details"""
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return RunResponse(
        run_id=run.run_id,
        timestamp=run.timestamp,
        graph_id=run.graph_id,
        task_type=run.task_type.value,
        status=run.status.value
    )# Runs router
