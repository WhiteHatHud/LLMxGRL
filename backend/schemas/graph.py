# schemas/graph.py
from pydantic import BaseModel
from typing import Optional
from schemas.base import BaseSchema

class GraphCreate(BaseModel):
    name: str
    description: Optional[str] = None

class GraphResponse(BaseSchema):
    graph_id: str
    name: str
    description: Optional[str]

class UploadResponse(BaseModel):
    graph_id: str
    nodes_count: int
    edges_count: int
    texts_count: int
    artifacts: list[str]