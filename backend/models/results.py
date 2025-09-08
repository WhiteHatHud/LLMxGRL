# models/results.py
from sqlalchemy import Column, Integer, String, Float, Text, Boolean
from db.database import Base

class RetrievalLog(Base):
    __tablename__ = "retrieval_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, index=True)
    query_id = Column(String, index=True)
    k = Column(Integer)
    hops = Column(Integer)
    prompt_template_id = Column(String)
    tokens_in_context = Column(Integer)
    context_preview = Column(Text)

class GenerationLog(Base):
    __tablename__ = "generation_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, index=True)
    query_id = Column(String, index=True)
    llm_name = Column(String)
    llm_params_json = Column(Text)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    est_cost_usd = Column(Float)
    llm_answer_text = Column(Text)
    llm_pred_label = Column(String)
    confidence = Column(Float)

class Metrics(Base):
    __tablename__ = "metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, index=True)
    query_id = Column(String, index=True)
    is_exact_match = Column(Boolean)
    accuracy = Column(Float)
    hits_at_k = Column(Float)
    macro_f1 = Column(Float)
    rouge = Column(Float)
    bertscore = Column(Float)