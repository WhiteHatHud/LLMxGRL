# models/run.py
from sqlalchemy import Column, Integer, String, DateTime, Enum as SQLEnum
from sqlalchemy.sql import func
from db.database import Base
from models.enums import TaskType, Status

class Run(Base):
    __tablename__ = "runs"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, unique=True, index=True)
    timestamp = Column(DateTime, server_default=func.now())
    graph_id = Column(String, index=True)
    task_type = Column(SQLEnum(TaskType))
    status = Column(SQLEnum(Status), default=Status.CREATED)

class Query(Base):
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, index=True)
    query_id = Column(String, unique=True, index=True)
    query_text = Column(String)

class Artifact(Base):
    __tablename__ = "artifacts"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, index=True)
    kind = Column(String)
    path = Column(String)
    meta_json = Column(String)# Run, Query, Artifact models
