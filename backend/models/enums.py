# models/enums.py
from enum import Enum

class TaskType(str, Enum):
    QA = "QA"
    SUMMARY = "SUMMARY"
    NODE_CLASSIFICATION = "NODE_CLASSIFICATION"

class Stage(str, Enum):
    UNIGRAPH = "UNIGRAPH"
    GNN_ADAPTER = "GNN_ADAPTER"
    TRANSLATOR = "TRANSLATOR"
    RAG = "RAG"
    GENERATION = "GENERATION"

class Status(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"# TaskType, Stage, Status enums
