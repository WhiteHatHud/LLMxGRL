# agents/agent_manager.py
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
import json

from agents.tools.llm_client import LLMClient
from agents.tools.retriever_tfidf import TFIDFRetriever
from agents.tools.graph_sos import GraphSOS
from agents.tools.unigraph_adapter import UniGraphAdapter
from agents.tools.gnn_adapter import GNNAdapter
from agents.tools.graph_translator import GraphTranslator
from models.llm import PromptTemplate
from utils.timers import timer
from utils.tokens import estimate_tokens

class AgentManager:
    def __init__(self, db: Session):
        self.db = db
        self.llm_client = LLMClient()
        self.retriever = TFIDFRetriever()
        self.graph_sos = GraphSOS()
        self.unigraph = UniGraphAdapter()
        self.gnn_adapter = GNNAdapter()
        self.translator = GraphTranslator()
    
    def prepare_rag_context(
        self,
        graph_id: str,
        query_text: str,
        top_k: int = 5,
        hops: int = 2
    ) -> Dict[str, Any]:
        """Prepare RAG context from graph"""
        with timer() as t:
            # Get retrieval results
            retrieved_nodes = self.retriever.retrieve(
                db=self.db,
                graph_id=graph_id,
                query=query_text,
                top_k=top_k
            )
            
            # Expand with GraphSOS
            context = self.graph_sos.serialize(
                db=self.db,
                graph_id=graph_id,
                seed_nodes=retrieved_nodes,
                hops=hops
            )
            
            tokens = estimate_tokens(context)
        
        return {
            "context": context,
            "retrieved_nodes": retrieved_nodes,
            "tokens_in_context": tokens,
            "retrieval_ms": t["elapsed_ms"]
        }
    
    def generate_response(
        self,
        query_text: str,
        context: str,
        prompt_template_id: str = "default_rag",
        llm_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate LLM response"""
        # Get prompt template
        template = self.db.query(PromptTemplate).filter(
            PromptTemplate.template_id == prompt_template_id
        ).first()
        
        if not template:
            template_text = "{context}\n\nQuestion: {question}"
        else:
            template_text = template.text
        
        # Format prompt
        prompt = template_text.format(
            context=context,
            question=query_text
        )
        
        # Generate with LLM
        with timer() as t:
            result = self.llm_client.send(prompt, llm_config or {})
        
        return {
            **result,
            "generation_ms": t["elapsed_ms"],
            "prompt": prompt
        }# Agent manager for orchestrating agent lifecycles
