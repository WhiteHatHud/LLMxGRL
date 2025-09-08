# agents/tools/retriever_tfidf.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session
from typing import List
from models.graph import NodeText

class TFIDFRetriever:
    def __init__(self):
        self.vectorizer = None
        self.vectors = None
        self.node_ids = []
    
    def fit(self, db: Session, graph_id: str):
        """Fit TF-IDF on all node texts for a graph"""
        node_texts = db.query(NodeText).filter(
            NodeText.graph_id == graph_id
        ).all()
        
        if not node_texts:
            return
        
        texts = [nt.text for nt in node_texts]
        self.node_ids = [nt.node_id for nt in node_texts]
        
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.vectors = self.vectorizer.fit_transform(texts)
    
    def retrieve(
        self,
        db: Session,
        graph_id: str,
        query: str,
        top_k: int = 5
    ) -> List[str]:
        """Retrieve top-k most similar nodes to query"""
        
        # Fit if not already fitted
        if self.vectorizer is None or not self.node_ids:
            self.fit(db, graph_id)
        
        if self.vectorizer is None:
            return []
        
        # Transform query
        query_vec = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, self.vectors).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return node IDs
        return [self.node_ids[i] for i in top_indices]# sklearn TF-IDF retriever
