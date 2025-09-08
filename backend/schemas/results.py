# schemas/results.py
from pydantic import BaseModel
from typing import List, Optional

class ChartRequest(BaseModel):
    charts: List[str]

class ChartResponse(BaseModel):
    paths: List[str]

class ExportResponse(BaseModel):
    path: str
    rows: int# Results schemas
