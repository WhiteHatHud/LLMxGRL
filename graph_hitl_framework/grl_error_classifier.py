#!/usr/bin/env python3
"""
grl_error_classifier.py
=======================
Classifies HITL failures as surface errors or structural errors
using graph-theoretic reasoning over the schema FK graph.

  Surface error:    Wrong token, function, alias, WHERE condition.
                    Tables are correct (or only one missing, no path impact).
                    → Standard text feedback will likely fix it.

  Structural error: Wrong join path. Missing intermediate table(s) that
                    are required graph-hops between the attempted tables.
                    → Graph-path feedback needed; text alone won't fix it.

This is rule-based (deterministic), not a trained classifier.
The graph distance is an exact computation, not a prediction.

Usage (programmatic)
--------------------
from grl_error_classifier import GRLErrorClassifier
import sys; sys.path.insert(0, "StructGPT-Ollama")
from schema_graph import SchemaGraph

sg = SchemaGraph("StructGPT-Ollama/data/spider/spider_data/tables.json")
clf = GRLErrorClassifier(sg)

result = clf.classify(
    pred_sql="SELECT AVG(mpg) FROM car_names JOIN cars_data ...",
    gold_sql="SELECT AVG(mpg) FROM car_makers JOIN model_list ...",
    db_id="car_1",
)
# → {
#     "error_type":       "structural",
#     "missing_tables":   ["model_list"],
#     "pred_tables":      ["car_names", "cars_data"],
#     "gold_tables":      ["car_makers", "model_list", "car_names", "cars_data"],
#     "max_graph_distance": 2,
#     "fk_path":          "CAR_MAKERS -[Id=Maker]-> MODEL_LIST -[...]-> ...",
#   }
"""

import sys
from pathlib import Path
from typing import Optional

import networkx as nx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "StructGPT-Ollama"))
from schema_graph import SchemaGraph, extract_tables_from_sql


# ─── Classifier ───────────────────────────────────────────────────────────────

class GRLErrorClassifier:
    """
    Rule-based surface-vs-structural error detector.
    Requires a loaded SchemaGraph for FK graph access.
    """

    # Thresholds
    STRUCTURAL_MIN_MISSING   = 1   # at least 1 table missing
    STRUCTURAL_MIN_DISTANCE  = 2   # gap of ≥2 hops in FK graph

    def __init__(self, schema_graph: SchemaGraph):
        self.sg = schema_graph

    def classify(self, pred_sql: str, gold_sql: str, db_id: str) -> dict:
        """
        Compare predicted SQL against gold SQL using the FK graph.

        Returns a dict with:
          error_type         : "structural" | "surface" | "unknown"
          missing_tables     : list of table names in gold but not pred
          extra_tables       : list of table names in pred but not gold
          pred_tables        : tables extracted from pred SQL
          gold_tables        : tables extracted from gold SQL
          max_graph_distance : max hop distance for missing intermediate tables
          fk_path            : computed FK path (if structural), else None
        """
        pred_tables_raw = extract_tables_from_sql(pred_sql)
        gold_tables_raw = extract_tables_from_sql(gold_sql)

        pred_set = set(t.lower() for t in pred_tables_raw)
        gold_set = set(t.lower() for t in gold_tables_raw)

        missing = list(gold_set - pred_set)
        extra   = list(pred_set - gold_set)

        result = {
            "error_type":         "surface",
            "missing_tables":     missing,
            "extra_tables":       extra,
            "pred_tables":        list(pred_set),
            "gold_tables":        list(gold_set),
            "max_graph_distance": 0,
            "fk_path":            None,
        }

        # No tables missing → surface error (wrong function, alias, etc.)
        if not missing:
            return result

        # Get the FK graph for this database
        entry = self.sg._db_index.get(db_id)
        if entry is None:
            result["error_type"] = "unknown"
            return result

        G, _ = self.sg._build_graph(entry)

        # For each pair of tables the model DID attempt to join,
        # check if any missing table lies on the shortest path between them.
        max_dist = 0
        is_structural = False

        for missing_table in missing:
            if missing_table not in G:
                continue
            for pred_table in pred_set:
                if pred_table not in G or pred_table == missing_table:
                    continue
                try:
                    dist = nx.shortest_path_length(G, pred_table, missing_table)
                    max_dist = max(max_dist, dist)
                    if dist >= self.STRUCTURAL_MIN_DISTANCE:
                        is_structural = True
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

        result["max_graph_distance"] = max_dist

        if is_structural or (missing and max_dist >= 1):
            result["error_type"] = "structural"
            # Compute the FK path that bridges the gap
            all_required = list(gold_set)
            fk_path = self.sg.get_fk_path(db_id, all_required)
            result["fk_path"] = fk_path

        return result

    def is_structural(self, pred_sql: str, gold_sql: str, db_id: str) -> bool:
        """Convenience method: returns True if error is structural."""
        return self.classify(pred_sql, gold_sql, db_id)["error_type"] == "structural"


# ─── Smoke Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tables_path = ROOT / "StructGPT-Ollama/data/spider/spider_data/tables.json"
    if not tables_path.exists():
        print(f"tables.json not found at {tables_path}")
        sys.exit(1)

    sg  = SchemaGraph(str(tables_path))
    clf = GRLErrorClassifier(sg)

    # car_1: classic structural failure
    pred = ("SELECT AVG(cars_data.mpg) FROM car_names "
            "JOIN cars_data ON car_names.makeid = cars_data.id "
            "WHERE car_names.make = 'usa'")
    gold = ("SELECT AVG(T4.mpg) FROM car_makers AS T1 "
            "JOIN model_list AS T2 ON T1.id = T2.maker "
            "JOIN car_names  AS T3 ON T2.model = T3.model "
            "JOIN cars_data  AS T4 ON T3.makeid = T4.id "
            "WHERE T1.country = 'usa'")

    result = clf.classify(pred, gold, "car_1")
    print("\n── car_1 structural test ──")
    for k, v in result.items():
        print(f"  {k:25s}: {v}")

    # Simple surface error: same tables, wrong function
    pred2 = "SELECT COUNT(*) FROM singer WHERE age > 25"
    gold2 = "SELECT AVG(age) FROM singer WHERE age > 25"
    result2 = clf.classify(pred2, gold2, "concert_singer")
    print("\n── concert_singer surface test ──")
    for k, v in result2.items():
        print(f"  {k:25s}: {v}")
