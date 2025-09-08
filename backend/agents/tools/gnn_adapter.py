# agents/tools/gnn_adapter.py
import networkx as nx
from sqlalchemy.orm import Session
from typing import Dict, Any
import json
from models.graph import Node, Edge, NodeText
from models.run import Artifact

class GNNAdapter:
    """Add structural bias via graph features"""
    
    def compute_features(
        self,
        db: Session,
        graph_id: str,
        run_id: str
    ) -> Dict[str, Any]:
        """Compute structural features for nodes"""
        
        # Build NetworkX graph
        G = nx.Graph()
        
        nodes = db.query(Node).filter(Node.graph_id == graph_id).all()
        for node in nodes:
            G.add_node(node.node_id)
        
        edges = db.query(Edge).filter(Edge.graph_id == graph_id).all()
        for edge in edges:
            G.add_edge(edge.src, edge.dst)
        
        # Compute features
        degrees = dict(G.degree())
        pagerank = nx.pagerank(G, max_iter=100) if len(G) > 0 else {}
        clustering = nx.clustering(G) if len(G) > 0 else {}
        
        # Add as text tags to nodes
        for node_id in G.nodes():
            features = []
            if node_id in degrees:
                features.append(f"degree:{degrees[node_id]}")
            if node_id in pagerank:
                features.append(f"pagerank:{pagerank[node_id]:.4f}")
            if node_id in clustering:
                features.append(f"clustering:{clustering[node_id]:.4f}")
            
            # Update NodeText with structural tags
            node_text = db.query(NodeText).filter(
                NodeText.graph_id == graph_id,
                NodeText.node_id == node_id
            ).first()
            
            if node_text:
                node_text.text = f"{node_text.text} [STRUCT: {' '.join(features)}]"
        
        db.commit()
        
        # Save features as artifact
        artifact = Artifact(
            run_id=run_id,
            kind="gnn_features",
            path=f"data/outputs/{run_id}_gnn_features.json",
            meta_json=json.dumps({
                "num_nodes": len(G.nodes()),
                "num_edges": len(G.edges()),
                "avg_degree": sum(degrees.values()) / len(degrees) if degrees else 0
            })
        )
        db.add(artifact)
        db.commit()
        
        return {
            "nodes_processed": len(G.nodes()),
            "features_added": ["degree", "pagerank", "clustering"]
        }# Structural bias adapter (stub APIs)
