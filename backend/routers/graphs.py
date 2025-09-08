# routers/graphs.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import Optional, List
import pandas as pd
import os

from core.dependencies import get_db
from core.security import get_api_key
from models.graph import Graph, Node, Edge, NodeText
from schemas.graph import GraphCreate, GraphResponse, UploadResponse
from utils.ids import generate_graph_id
from utils.io import save_upload, load_csv

router = APIRouter()

@router.post("", response_model=GraphResponse)
def create_graph(
    graph_data: GraphCreate,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key)
):
    """Create a new graph"""
    graph_id = generate_graph_id()
    
    graph = Graph(
        graph_id=graph_id,
        name=graph_data.name,
        description=graph_data.description
    )
    
    db.add(graph)
    db.commit()
    db.refresh(graph)
    
    return GraphResponse(
        graph_id=graph.graph_id,
        name=graph.name,
        description=graph.description
    )

@router.post("/{graph_id}/upload", response_model=UploadResponse)
async def upload_graph_data(
    graph_id: str,
    edges_file: Optional[UploadFile] = File(None),
    nodes_file: Optional[UploadFile] = File(None),
    node_text_file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key)
):
    """Upload graph data files"""
    
    # Check graph exists
    graph = db.query(Graph).filter(Graph.graph_id == graph_id).first()
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    artifacts = []
    nodes_count = 0
    edges_count = 0
    texts_count = 0
    
    # Process edges
    if edges_file:
        filepath = f"data/uploads/{graph_id}_edges.csv"
        content = await edges_file.read()
        save_upload(content, filepath)
        artifacts.append(filepath)
        
        df = load_csv(filepath)
        for _, row in df.iterrows():
            edge = Edge(
                graph_id=graph_id,
                src=str(row.get('src', row.get('source', row[0]))),
                dst=str(row.get('dst', row.get('target', row[1]))),
                relation=str(row.get('relation', row.get('type', 'connects')))
            )
            db.add(edge)
            edges_count += 1
    
    # Process nodes
    if nodes_file:
        filepath = f"data/uploads/{graph_id}_nodes.csv"
        content = await nodes_file.read()
        save_upload(content, filepath)
        artifacts.append(filepath)
        
        df = load_csv(filepath)
        for _, row in df.iterrows():
            node = Node(
                graph_id=graph_id,
                node_id=str(row.get('node_id', row.get('id', row[0]))),
                label=str(row.get('label', row.get('type', 'node')))
            )
            db.add(node)
            nodes_count += 1
    
    # Process node texts
    if node_text_file:
        filepath = f"data/uploads/{graph_id}_node_text.csv"
        content = await node_text_file.read()
        save_upload(content, filepath)
        artifacts.append(filepath)
        
        df = load_csv(filepath)
        for _, row in df.iterrows():
            node_text = NodeText(
                graph_id=graph_id,
                node_id=str(row.get('node_id', row.get('id', row[0]))),
                text=str(row.get('text', row.get('content', row[1] if len(row) > 1 else '')))
            )
            db.add(node_text)
            texts_count += 1
    
    db.commit()
    
    return UploadResponse(
        graph_id=graph_id,
        nodes_count=nodes_count,
        edges_count=edges_count,
        texts_count=texts_count,
        artifacts=artifacts
    )# Graphs router
