# models/graph.py
from sqlalchemy import Column, Integer, String, Text
from db.database import Base

class Graph(Base):
    __tablename__ = "graphs"
    
    id = Column(Integer, primary_key=True, index=True)
    graph_id = Column(String, unique=True, index=True)
    name = Column(String)
    description = Column(Text)

class Node(Base):
    __tablename__ = "nodes"
    
    id = Column(Integer, primary_key=True, index=True)
    graph_id = Column(String, index=True)
    node_id = Column(String, index=True)
    label = Column(String)

class NodeText(Base):
    __tablename__ = "node_texts"
    
    id = Column(Integer, primary_key=True, index=True)
    graph_id = Column(String, index=True)
    node_id = Column(String, index=True)
    text = Column(Text)

class Edge(Base):
    __tablename__ = "edges"
    
    id = Column(Integer, primary_key=True, index=True)
    graph_id = Column(String, index=True)
    src = Column(String)
    dst = Column(String)
    relation = Column(String)