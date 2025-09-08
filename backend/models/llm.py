# models/llm.py
from sqlalchemy import Column, Integer, String, Float, Text
from db.database import Base

class LLMConfig(Base):
    __tablename__ = "llm_configs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    provider = Column(String)
    model = Column(String)
    temperature = Column(Float)
    top_p = Column(Float)
    max_tokens = Column(Integer)

class PromptTemplate(Base):
    __tablename__ = "prompt_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    template_id = Column(String, unique=True, index=True)
    kind = Column(String)
    text = Column(Text)