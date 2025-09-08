# agents/tools/graph_sos.py
import networkx as nx
from sqlalchemy.orm import Session
from typing import List, Set
from models.graph import Node, Edge, NodeText

class GraphSOS:
    """Graph Serialization with Order Sensitivity"""
    
    def serialize(
        self,
        db: Session,
        graph_id: str,
        seed_nodes: List[str],
        hops: int = 2
    ) -> str:
        """Serialize subgraph with order-sensitive paths"""
        
        # Build NetworkX graph
        G = self._build_graph(db, graph_id)
        
        # Expand from seed nodes
        expanded_nodes = self._expand_nodes(G, seed_nodes, hops)
        
        # Get paths between nodes
        paths = self._get_paths(G, expanded_nodes)
        
        # Serialize to text
        context_parts = []
        
        # Add node texts
        node_texts = db.query(NodeText).filter(
            NodeText.graph_id == graph_id,
            NodeText.node_id.in_(expanded_nodes)
        ).all()
        
        for nt in node_texts[:10]:  # Limit for context size
            context_parts.append(f"Node {nt.node_id}: {nt.text[:200]}")
        
        # Add path information
        for path in paths[:5]:  # Limit paths
            path_str = " → ".join(path)
            context_parts.append(f"Path: {path_str}")
        
        return "\n".join(context_parts)
    
    def _build_graph(self, db: Session, graph_id: str) -> nx.Graph:
        """Build NetworkX graph from database"""
        G = nx.Graph()
        
        # Add nodes
        nodes = db.query(Node).filter(Node.graph_id == graph_id).all()
        for node in nodes:
            G.add_node(node.node_id, label=node.label)
        
        # Add edges
        edges = db.query(Edge).filter(Edge.graph_id == graph_id).all()
        for edge in edges:
            G.add_edge(edge.src, edge.dst, relation=edge.relation)
        
        return G
    
    def _expand_nodes(
        self,
        G: nx.Graph,
        seed_nodes: List[str],
        hops: int
    ) -> Set[str]:
        """Expand nodes by k hops"""
        expanded = set(seed_nodes)
        current_layer = set(seed_nodes)
        
        for _ in range(hops):
            next_layer = set()
            for node in current_layer:
                if node in G:
                    neighbors = G.neighbors(node)
                    next_layer.update(neighbors)
            expanded.update(next_layer)
            current_layer = next_layer
        
        return expanded
    
    def _get_paths(
        self,
        G: nx.Graph,
        nodes: Set[str],
        max_paths: int = 10
    ) -> List[List[str]]:
        """Get paths between nodes"""
        paths = []
        node_list = list(nodes)
        
        for i, src in enumerate(node_list):
            if i >= 5:  # Limit source nodes
                break
            for dst in node_list[i+1:i+3]:  # Limit destinations
                if src in G and dst in G:
                    try:
                        path = nx.shortest_path(G, src, dst)
                        if len(path) > 1 and len(path) <= 4:
                            paths.append(path)
                    except nx.NetworkXNoPath:
                        pass
                    
                    if len(paths) >= max_paths:
                        return paths
        
        return paths# Order-sensitive serialization + sampling
