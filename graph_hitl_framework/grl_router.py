#!/usr/bin/env python3
"""
grl_router.py
=============
GRL-based query complexity router.

Uses Node2Vec schema embeddings + sentence-transformer query encoding
to predict join complexity before any inference, routing each query to
the appropriate HITL strategy:

  "simple"  (0-1 joins) → single pass
  "medium"  (2 joins)   → standard HITL (3 iterations)
  "complex" (3+ joins)  → graph-augmented HITL

Training
--------
200 Spider samples → logistic regression on graph + similarity features.
50 held-out samples → evaluation.

Usage
-----
# Train
python graph_hitl_framework/grl_router.py \
    --answer_key answer_key.json \
    --embeddings graph_hitl_framework/cache/schema_embeddings.pkl \
    --tables StructGPT-Ollama/data/spider/spider_data/tables.json \
    --output graph_hitl_framework/cache/grl_router_model.pkl

# Predict (programmatic)
from grl_router import GRLRouter
router = GRLRouter.load("graph_hitl_framework/cache/grl_router_model.pkl")
result = router.route(question="...", db_id="car_1")
# → {"route": "complex", "predicted_joins": 3, "confidence": 0.91}
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "StructGPT-Ollama"))
from schema_graph import SchemaGraph, count_joins_in_sql

AGGREGATION_KEYWORDS = {
    "average", "avg", "total", "sum", "count", "maximum", "minimum",
    "max", "min", "how many", "how much", "percent", "ratio",
}

MULTI_TABLE_KEYWORDS = {
    "who", "which", "where", "list all", "find all", "across", "between",
    "made by", "belongs to", "associated", "related",
}


# ─── Feature Extraction ───────────────────────────────────────────────────────

def _query_text_features(question: str) -> np.ndarray:
    """
    Lightweight text features that don't require a model download.
    13 features total.
    """
    q = question.lower()
    words = q.split()

    has_agg     = float(any(k in q for k in AGGREGATION_KEYWORDS))
    has_multi   = float(any(k in q for k in MULTI_TABLE_KEYWORDS))
    word_count  = len(words)
    char_count  = len(question)
    num_clauses = q.count(",") + q.count(" and ") + q.count(" or ")

    # Structural question words often correlate with join depth
    wh_words = sum(1 for w in ["what", "who", "which", "where", "when", "how"]
                   if w in words)

    return np.array([
        has_agg, has_multi,
        min(word_count / 30.0, 1.0),      # normalised word count
        min(char_count / 200.0, 1.0),     # normalised char count
        min(num_clauses / 5.0, 1.0),      # clause density
        min(wh_words / 3.0, 1.0),         # wh-word count
    ], dtype=np.float32)


def _embedding_similarity_features(question: str, db_id: str,
                                    cache: dict,
                                    encoder=None) -> np.ndarray:
    """
    Similarity features between query embedding and table embeddings.
    If encoder (sentence-transformer) is None, returns zeros.
    """
    emb_map = cache.get("embeddings", {}).get(db_id, {})
    if not emb_map or encoder is None:
        return np.zeros(4, dtype=np.float32)

    q_emb = encoder.encode([question], convert_to_numpy=True)[0]  # (384,)
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)

    sims = []
    for t_emb in emb_map.values():
        if np.any(t_emb) and t_emb.shape == q_emb.shape:
            t_norm = t_emb / (np.linalg.norm(t_emb) + 1e-9)
            sims.append(float(np.dot(q_norm, t_norm)))

    if not sims:
        return np.zeros(4, dtype=np.float32)

    return np.array([
        max(sims),
        float(np.mean(sims)),
        float(np.std(sims)),
        float(np.min(sims)),
    ], dtype=np.float32)


def _schema_graph_features(db_id: str, cache: dict) -> np.ndarray:
    """
    Schema-level graph statistics for the database.
    5 features.
    """
    sf = cache.get("schema_feats", {}).get(db_id, {})
    return np.array([
        min(sf.get("num_tables", 0) / 20.0, 1.0),
        min(sf.get("num_edges", 0)  / 30.0, 1.0),
        min(sf.get("avg_degree", 0) / 5.0,  1.0),
        min(sf.get("max_degree", 0) / 10.0, 1.0),
        1.0 if sf.get("diameter", -1) >= 3 else 0.0,  # deep graph flag
    ], dtype=np.float32)


def extract_features(question: str, db_id: str,
                     cache: dict, encoder=None) -> np.ndarray:
    """Concatenate all feature groups into one vector."""
    text_feats   = _query_text_features(question)           # 6
    sim_feats    = _embedding_similarity_features(question, db_id, cache, encoder)  # 4
    graph_feats  = _schema_graph_features(db_id, cache)     # 5
    return np.concatenate([text_feats, sim_feats, graph_feats])  # 15 total


def join_count_to_label(n_joins: int) -> int:
    """Map join count to 3-class label."""
    if n_joins <= 1:
        return 0  # simple
    if n_joins == 2:
        return 1  # medium
    return 2      # complex


LABEL_TO_ROUTE = {0: "simple", 1: "medium", 2: "complex"}


# ─── GRL Router Class ─────────────────────────────────────────────────────────

class GRLRouter:
    """
    Predicts query join complexity from graph structure + question text.
    Wraps a trained sklearn LogisticRegression classifier.
    """

    def __init__(self, model, scaler, encoder=None, cache: dict = None):
        self.model   = model
        self.scaler  = scaler
        self.encoder = encoder
        self.cache   = cache or {}

    def route(self, question: str, db_id: str) -> dict:
        """
        Returns:
          {
            "route":           "simple" | "medium" | "complex",
            "predicted_joins": int,
            "confidence":      float,
          }
        """
        feat = extract_features(question, db_id, self.cache, self.encoder)
        feat_scaled = self.scaler.transform(feat.reshape(1, -1))
        label = int(self.model.predict(feat_scaled)[0])
        proba = self.model.predict_proba(feat_scaled)[0]
        confidence = float(proba[label])
        join_map = {0: 1, 1: 2, 2: 3}   # representative join count per class
        return {
            "route":           LABEL_TO_ROUTE[label],
            "predicted_joins": join_map[label],
            "confidence":      confidence,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model":  self.model,
                "scaler": self.scaler,
            }, f)
        print(f"Router model saved → {path}")

    @classmethod
    def load(cls, model_path: str, embeddings_path: str = None,
             use_encoder: bool = True):
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        cache = {}
        if embeddings_path and Path(embeddings_path).exists():
            with open(embeddings_path, "rb") as f:
                cache = pickle.load(f)

        encoder = None
        if use_encoder:
            try:
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer("all-MiniLM-L6-v2")
                print("Sentence transformer loaded.")
            except Exception as e:
                print(f"[router] No sentence encoder: {e}. Using text features only.")

        return cls(data["model"], data["scaler"], encoder, cache)


# ─── Training ─────────────────────────────────────────────────────────────────

def train(answer_key_path: str, embeddings_path: str,
          tables_path: str, output_path: str,
          train_size: int = 200) -> GRLRouter:

    # Load data
    with open(answer_key_path) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples from answer key.")

    with open(embeddings_path, "rb") as f:
        cache = pickle.load(f)
    print(f"Loaded embeddings for {len(cache.get('embeddings', {}))} databases.")

    # Try to load sentence encoder
    encoder = None
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading sentence transformer (downloads ~80MB on first run) …")
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Sentence transformer ready.")
    except Exception as e:
        print(f"[warning] No sentence encoder available: {e}. Continuing without.")

    # Build features and labels
    X, y, ids = [], [], []
    skipped = 0
    for s in samples:
        question = s.get("question", "")
        db_id    = s.get("db_id", "")
        gold_sql = s.get("gold_sql", s.get("query", ""))

        if not question or not db_id or not gold_sql:
            skipped += 1
            continue

        n_joins = count_joins_in_sql(gold_sql)
        label   = join_count_to_label(n_joins)
        feat    = extract_features(question, db_id, cache, encoder)

        X.append(feat)
        y.append(label)
        ids.append(s.get("id", question[:40]))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Features: {X.shape}  skipped={skipped}")
    print(f"Label distribution: simple={sum(y==0)}  medium={sum(y==1)}  complex={sum(y==2)}")

    # Stratified 200/50 split
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size,
                                 random_state=42)
    train_idx, test_idx = next(sss.split(X, y))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"\nTrain: {len(X_train)}   Test: {len(X_test)}")

    # Scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Train logistic regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42,
                             class_weight="balanced")
    clf.fit(X_train_sc, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_sc)
    print("\n── GRL Router Evaluation (50 held-out samples) ──")
    print(classification_report(y_test, y_pred,
                                 target_names=["simple", "medium", "complex"]))

    router = GRLRouter(clf, scaler, encoder, cache)
    router.save(output_path)
    return router


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train GRL Router classifier")
    p.add_argument("--answer_key",  required=True)
    p.add_argument("--embeddings",  required=True)
    p.add_argument("--tables",      required=True)
    p.add_argument("--output",      default="graph_hitl_framework/cache/grl_router_model.pkl")
    p.add_argument("--train_size",  type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        answer_key_path=args.answer_key,
        embeddings_path=args.embeddings,
        tables_path=args.tables,
        output_path=args.output,
        train_size=args.train_size,
    )
