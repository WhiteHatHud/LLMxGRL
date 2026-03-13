"""
schema_graph.py — SchemaGraph: FK-edge path builder for Spider schemas.

Loads tables.json, builds a foreign-key graph per database, and returns
formatted FK path chains suitable for direct injection into LLM prompts.
"""

import json
import re
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import networkx as nx


class SchemaGraph:
    """
    Builds an undirected FK graph from Spider tables.json.

    Nodes  : table names (lowercase)
    Edges  : labeled with the FK column pair that connects two tables
             e.g. model_list --[Maker=Id]--> car_makers
    """

    def __init__(self, tables_json_path: str):
        with open(tables_json_path) as f:
            self._raw = json.load(f)
        self._db_index: Dict[str, dict] = {e["db_id"]: e for e in self._raw}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_fk_path(self, db_id: str, required_tables: List[str]) -> Optional[str]:
        """
        Given a list of table names required by a query, return a
        formatted FK-path string connecting them all.

        e.g. "CAR_MAKERS -[Id=Maker]-> MODEL_LIST -[Model=Model]-> CAR_NAMES -[MakeId=Id]-> CARS_DATA"

        Returns None if db_id unknown or no path found.
        """
        entry = self._db_index.get(db_id)
        if entry is None or len(required_tables) < 2:
            return None

        G, edge_labels = self._build_graph(entry)
        tables_lower = [t.lower() for t in required_tables]

        # Verify all tables exist in the graph
        tables_lower = [t for t in tables_lower if t in G]
        if len(tables_lower) < 2:
            return None

        path = self._find_chain(G, tables_lower)
        if path is None:
            return None

        return self._format_path(path, edge_labels)

    def get_full_schema_graph_text(self, db_id: str) -> Optional[str]:
        """
        Returns a complete text description of all FK edges in the schema.
        Useful as a fallback when specific tables cannot be identified.
        """
        entry = self._db_index.get(db_id)
        if entry is None:
            return None

        _, edge_labels = self._build_graph(entry)
        lines = ["Foreign key relationships:"]
        seen = set()
        for (t1, t2), (c1, c2) in edge_labels.items():
            key = tuple(sorted([(t1, c1), (t2, c2)]))
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  {t1.upper()}.{c1} = {t2.upper()}.{c2}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_graph(self, entry: dict) -> Tuple[nx.Graph, Dict]:
        """Build undirected graph + edge label map from a tables.json entry."""
        tables_orig = entry["table_names_original"]
        col_names = entry["column_names_original"]  # [(table_idx, col_name), ...]
        foreign_keys = entry["foreign_keys"]         # [(col_id, col_id), ...]

        G = nx.Graph()
        for t in tables_orig:
            G.add_node(t.lower())

        # edge_labels: (src_table, dst_table) -> (src_col, dst_col)
        # stored in both directions so lookup is O(1) regardless of traversal direction
        edge_labels: Dict[Tuple[str, str], Tuple[str, str]] = {}

        for (fk_col_id, ref_col_id) in foreign_keys:
            fk_tbl_idx, fk_col = col_names[fk_col_id]
            ref_tbl_idx, ref_col = col_names[ref_col_id]
            fk_table = tables_orig[fk_tbl_idx].lower()
            ref_table = tables_orig[ref_tbl_idx].lower()

            G.add_edge(fk_table, ref_table)
            # fk_table.fk_col references ref_table.ref_col
            edge_labels[(fk_table, ref_table)] = (fk_col, ref_col)
            edge_labels[(ref_table, fk_table)] = (ref_col, fk_col)

        return G, edge_labels

    def _find_chain(self, G: nx.Graph, tables: List[str]) -> Optional[List[str]]:
        """
        Find the shortest node sequence that visits all required tables.

        Strategy: for small table sets try all orderings; pick the one
        whose total shortest-path length is minimal.  For larger sets fall
        back to a greedy nearest-neighbour approach.
        """
        if len(tables) == 1:
            return tables

        if len(tables) <= 5:
            best_path: Optional[List[str]] = None
            best_len = float("inf")

            for perm in permutations(tables):
                path_nodes: List[str] = []
                valid = True
                for i in range(len(perm) - 1):
                    try:
                        sp = nx.shortest_path(G, perm[i], perm[i + 1])
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        valid = False
                        break
                    path_nodes.extend(sp if i == 0 else sp[1:])

                if valid and len(path_nodes) < best_len:
                    best_len = len(path_nodes)
                    best_path = path_nodes
        else:
            # Greedy nearest-neighbour
            path_nodes = [tables[0]]
            remaining = set(tables[1:])
            while remaining:
                last = path_nodes[-1]
                closest: Optional[str] = None
                closest_sp: Optional[List[str]] = None
                closest_len = float("inf")
                for t in remaining:
                    try:
                        sp = nx.shortest_path(G, last, t)
                        if len(sp) < closest_len:
                            closest_len = len(sp)
                            closest = t
                            closest_sp = sp
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
                if closest is None:
                    break
                path_nodes.extend(closest_sp[1:])  # type: ignore[index]
                remaining.discard(closest)
            best_path = path_nodes

        if best_path is None:
            return None

        # Deduplicate while preserving order
        seen: set = set()
        deduped: List[str] = []
        for node in best_path:
            if node not in seen:
                seen.add(node)
                deduped.append(node)
        return deduped

    def _format_path(self, path: List[str], edge_labels: Dict) -> str:
        """
        Format a node path as a FK join chain string.

        Example:
          CAR_MAKERS -[Id=Maker]-> MODEL_LIST -[Model=Model]-> CAR_NAMES
        """
        parts = [path[0].upper()]
        for i in range(1, len(path)):
            prev, curr = path[i - 1], path[i]
            label = edge_labels.get((prev, curr))
            if label:
                src_col, dst_col = label
                parts.append(f"-[{src_col}={dst_col}]-> {curr.upper()}")
            else:
                parts.append(f"-> {curr.upper()}")
        return " ".join(parts)


# ------------------------------------------------------------------
# SQL utility helpers (used by the re-run script)
# ------------------------------------------------------------------

def extract_tables_from_sql(sql: str) -> List[str]:
    """Extract table names referenced in a SQL query (FROM / JOIN clauses)."""
    tables: List[str] = []
    pattern = r"\b(?:FROM|JOIN)\s+([`\"\[]?[a-zA-Z_][a-zA-Z0-9_]*[`\"\]]?)"
    for match in re.finditer(pattern, sql, re.IGNORECASE):
        table = match.group(1).strip("`\"[]")
        tables.append(table)
    return tables


def count_joins_in_sql(sql: str) -> int:
    return len(re.findall(r"\bJOIN\b", sql, re.IGNORECASE))


def clean_sql_prediction(prediction: str) -> str:
    """Strip markdown fences and normalise whitespace."""
    sql = re.sub(r"```\s*(?:sqlite|sql)?\s*", "", prediction)
    sql = re.sub(r"```", "", sql)
    sql = sql.replace("\n", " ")
    sql = " ".join(sql.split())
    sql = sql.rstrip(";").strip()
    if not sql.upper().startswith("SELECT"):
        sql = "SELECT " + sql
    sql = re.sub(r"\bSELECT\s+SELECT\b", "SELECT", sql, flags=re.IGNORECASE)
    return sql.strip()


# ------------------------------------------------------------------
# Quick smoke-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    tables_path = "data/spider/spider_data/tables.json"
    sg = SchemaGraph(tables_path)

    # Reproduce the car_1 path that Qwen keeps getting wrong
    path = sg.get_fk_path("car_1", ["car_makers", "model_list", "car_names", "cars_data"])
    print("car_1 path:", path)

    path2 = sg.get_fk_path("car_1", ["car_makers", "cars_data"])
    print("car_1 end-to-end:", path2)

    print()
    print(sg.get_full_schema_graph_text("car_1"))
