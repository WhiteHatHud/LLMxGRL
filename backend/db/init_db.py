# db/init_db.py
from sqlalchemy.orm import Session
from db.database import SessionLocal
from models.llm import PromptTemplate, LLMConfig
import logging

logger = logging.getLogger(__name__)

def init_db():
    db = SessionLocal()
    try:
        # Check if already initialized
        existing = db.query(PromptTemplate).first()
        if existing:
            return
        
        # Add default prompt templates
        rag_prompt = PromptTemplate(
            template_id="default_rag",
            kind="rag",
            text="""You are an expert graph analyst. Using the CONTEXT (a serialized subgraph),
answer the QUESTION concisely with citations to node IDs when possible.
CONTEXT:
{context}
QUESTION:
{question}"""
        )
        
        qa_prompt = PromptTemplate(
            template_id="default_qa",
            kind="qa",
            text="""Given the following text-attributed graph excerpts, answer the user query.
If uncertain, answer "NA".
EXCERPTS:
{context}
QUERY:
{question}"""
        )
        
        db.add(rag_prompt)
        db.add(qa_prompt)
        
        # Add default LLM config
        default_llm = LLMConfig(
            name="default",
            provider="openrouter",
            model="meta-llama/llama-3.1-8b-instruct",
            temperature=0.2,
            top_p=0.9,
            max_tokens=512
        )
        
        db.add(default_llm)
        db.commit()
        logger.info("Database initialized with default data")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()