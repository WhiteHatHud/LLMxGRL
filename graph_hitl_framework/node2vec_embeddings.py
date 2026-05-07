#!/usr/bin/env python3
"""
node2vec_embeddings.py
======================
Train Node2Vec embeddings on Spider database schema graphs.

For each database in tables.json:
  1. Build FK graph using SchemaGraph (tables = nodes, FK = edges)
  2. Run Node2Vec to learn 64-dim structural embeddings per table
  3. Compute structural features: degree, PageRank, clustering coefficient
  4. Cache everything to disk for reuse by the GRL Router

Usage
-----
python graph_hitl_framework/node2vec_embeddings.py \
    --tables StructGPT-Ollama/data/spider/spider_data/tables.json \
    --output graph_hitl_framework/cache/schema_embeddings.pkl
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np

# Resolve StructGPT-Ollama on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "StructGPT-Ollama"))
from schema_graph import SchemaGraph


# ─── Node2Vec Helpers ─────────────────────────────────────────────────────────

def _node2vec_embeddings(G: nx.Graph, dims: int, walk_length: int,
                          num_walks: int) -> dict:
    """
    Train Node2Vec on graph G and return {node: np.array(dims)}.
    Falls back to zero vectors if the graph is too small or training fails.
    """
    if len(G.nodes()) < 2:
        return {n: np.zeros(dims) for n in G.nodes()}

    try:
        from node2vec import Node2Vec
        n2v = Node2Vec(
            G,
            dimensions=dims,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=1,   # single worker avoids multiprocessing issues
            quiet=True,
        )
        model = n2v.fit(window=5, min_count=1, batch_words=4)
        return {node: model.wv[node] for node in G.nodes()}
    except Exception as e:
        print(f"    [node2vec] fallback to zeros: {e}")
        return {node: np.zeros(dims) for node in G.nodes()}


def _structural_features(G: nx.Graph) -> dict:
    """
    Compute graph-structural features per node.
    Returns {node: {"degree": int, "pagerank": float, "clustering": float,
                     "avg_shortest_path": float}}
    """
    if len(G.nodes()) == 0:
        return {}

    degrees = dict(G.degree())

    try:
        pagerank = nx.pagerank(G, max_iter=200)
    except Exception:
        uniform = 1.0 / max(len(G), 1)
        pagerank = {n: uniform for n in G.nodes()}

    clustering = nx.clustering(G)

    # Average shortest-path length from each node to all others
    avg_path: dict = {}
    for node in G.nodes():
        lengths = nx.single_source_shortest_path_length(G, node)
        others = [v for k, v in lengths.items() if k != node]
        avg_path[node] = float(np.mean(others)) if others else 0.0

    return {
        node: {
            "degree":           degrees.get(node, 0),
            "pagerank":         pagerank.get(node, 0.0),
            "clustering":       clustering.get(node, 0.0),
            "avg_shortest_path": avg_path.get(node, 0.0),
        }
        for node in G.nodes()
    }


def _schema_level_features(G: nx.Graph) -> dict:
    """Schema-level graph statistics (one dict per database)."""
    n = len(G.nodes())
    if n == 0:
        return {"num_tables": 0, "num_edges": 0,
                "avg_degree": 0.0, "diameter": 0}

    degrees = [d for _, d in G.degree()]
    try:
        diameter = nx.diameter(G) if nx.is_connected(G) else -1
    except Exception:
        diameter = -1

    return {
        "num_tables": n,
        "num_edges":  G.number_of_edges(),
        "avg_degree": float(np.mean(degrees)),
        "max_degree": int(np.max(degrees)),
        "diameter":   diameter,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_embeddings(tables_path: str, output_path: str,
                     dims: int = 64, walk_length: int = 10,
                     num_walks: int = 100) -> dict:
    """
    Build and cache Node2Vec embeddings + structural features for all
    Spider databases found in tables_path.

    Returns the full cache dict:
    {
      "embeddings":  {db_id: {table_name: np.array(dims)}},
      "node_feats":  {db_id: {table_name: {degree, pagerank, ...}}},
      "schema_feats":{db_id: {num_tables, num_edges, avg_degree, ...}},
    }
    """
    with open(tables_path) as f:
        tables_data = json.load(f)

    sg = SchemaGraph(tables_path)

    embeddings: dict  = {}
    node_feats: dict  = {}
    schema_feats: dict = {}

    print(f"Processing {len(tables_data)} databases …")
    for i, entry in enumerate(tables_data):
        db_id = entry["db_id"]

        # Build graph using existing SchemaGraph helper
        G, _ = sg._build_graph(entry)

        print(f"  [{i+1:3d}/{len(tables_data)}] {db_id:30s} "
              f"nodes={len(G.nodes()):2d}  edges={G.number_of_edges():2d}", end="")

        embeddings[db_id]   = _node2vec_embeddings(G, dims, walk_length, num_walks)
        node_feats[db_id]   = _structural_features(G)
        schema_feats[db_id] = _schema_level_features(G)

        print(f"  ✓")

    cache = {
        "embeddings":   embeddings,
        "node_feats":   node_feats,
        "schema_feats": schema_feats,
        "dims":         dims,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"\nSaved embeddings for {len(embeddings)} databases → {output_path}")
    return cache


def load_embeddings(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Node2Vec on Spider schemas")
    p.add_argument("--tables", required=True,
                   help="Path to tables.json")
    p.add_argument("--output", default="graph_hitl_framework/cache/schema_embeddings.pkl",
                   help="Output pickle path")
    p.add_argument("--dims",        type=int, default=64)
    p.add_argument("--walk_length", type=int, default=10)
    p.add_argument("--num_walks",   type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_embeddings(
        tables_path=args.tables,
        output_path=args.output,
        dims=args.dims,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
    )
