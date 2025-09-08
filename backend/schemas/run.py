# schemas/run.py
from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime
from schemas.base import BaseSchema

class RunCreate(BaseModel):
    graph_id: str
    task_type: Literal["QA", "SUMMARY", "NODE_CLASSIFICATION"]

class RunResponse(BaseSchema):
    run_id: str
    timestamp: datetime
    graph_id: str
    task_type: str
    status: str