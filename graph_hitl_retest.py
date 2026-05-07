#!/usr/bin/env python3
"""
Graph-Augmented HITL Retest
============================
Re-runs the HITL loop on only the FAILED samples from a previous hitl_results.json,
with an explicit schema graph injected into the first message (Option A).

Three graph serialization modes (--mode):
  graph_sos   — FK paths between all table pairs  (GraphSOS approach)
  gnn         — Structural features per table     (GNNAdapter approach)
  translator  — Template edge list                (GraphTranslator approach)

Usage
-----
  python graph_hitl_retest.py --mode graph_sos
  python graph_hitl_retest.py --mode gnn
  python graph_hitl_retest.py --mode translator

  # custom input/output
  python graph_hitl_retest.py --mode graph_sos \
    --hitl_results hitl_results.json \
    --output graph_hitl_results_graph_sos.json
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import time
from datetime import timedelta
from typing import Any

import networkx as nx
import requests

# ─── Configuration (mirror hitl_sql_testbed.py) ───────────────────────────────

LMSTUDIO_BASE = "http://localhost:1234/v1"

AI_AGENT_MODEL    = "qwen/qwen2.5-coder-14b"
HUMAN_PROXY_MODEL = "deepseek/deepseek-r1-0528-qwen3-8b"

HUMAN_PROXY_PROVIDER = "lmstudio"   # "lmstudio" | "anthropic"
ANTHROPIC_MODEL      = "claude-3-5-sonnet-20241022"

MAX_ITERATIONS  = 3
REQUEST_TIMEOUT = 180

DB_ROOT = "StructGPT-Ollama/data/spider/spider_data/database"

# ─── System Prompts ───────────────────────────────────────────────────────────

AI_AGENT_SYSTEM = """You are a Graph-Aware SQL Assistant. You translate natural language into SQL queries based on a structured database schema.

Your Strategy:
1. Analyze: First, explain your reasoning path through the database structure (e.g., 'I need to link Table A to Table B via key X').
2. Clarify: If the user's request is ambiguous or missing key information, ask for clarification before writing any code.
3. Refine: If the user provides feedback on your SQL, analyze the error and generate a refined version.

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{"thought": "...", "clarification_needed": true, "question": "...", "sql": ""}

OR

{"thought": "...", "clarification_needed": false, "question": "", "sql": "SELECT ..."}"""

HUMAN_PROXY_SYSTEM_TEMPLATE = """You are an Oracle User Simulator. Your goal is to help an AI assistant correctly interpret a user's intent based on a provided Answer Key.

Knowledge Base (Answer Key Entry):
{answer_key_data}

Instructions:
- If the AI asks a clarifying question, answer it concisely using only the information in the Answer Key.
- If the AI provides an answer (SQL), compare its output to the 'Gold Standard' in your Answer Key.
- If the AI is wrong, provide specific, actionable feedback pointing out the error (e.g., 'You joined the wrong tables' or 'The WHERE clause should filter for age < 30').
- Do not give away the final answer immediately; guide the AI to reason it out.
- Keep your response to 2-3 sentences maximum."""

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Schema Graph Builder ─────────────────────────────────────────────────────

def build_schema_graph(db_path: str) -> nx.DiGraph:
    """Build a directed graph of the schema FK relationships from the SQLite file.
    Nodes = table names. Edges = foreign key links with from_col / to_col attributes."""
    G = nx.DiGraph()
    if not os.path.exists(db_path):
        return G
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    tables = [row[0] for row in cursor.fetchall()]
    for table in tables:
        G.add_node(table)
        try:
            cursor.execute(f"PRAGMA foreign_key_list('{table}')")
            for row in cursor.fetchall():
                # row: (id, seq, ref_table, from_col, to_col, ...)
                ref_table = row[2]
                from_col  = row[3]
                to_col    = row[4]
                G.add_node(ref_table)
                G.add_edge(table, ref_table, from_col=from_col, to_col=to_col)
        except Exception:
            pass
    conn.close()
    return G


# ─── Graph Serialization: MODE 1 — GraphSOS ───────────────────────────────────
# Finds shortest FK paths between every pair of tables and serializes them.
# Maps to agents/tools/graph_sos.py (order-sensitive path serialization).

def serialize_graph_sos(G: nx.DiGraph) -> str:
    """GraphSOS approach: emit all unique FK traversal paths up to length 4."""
    if len(G) == 0:
        return "Schema Graph: (no tables found)"

    lines = ["Schema Graph — FK Paths (GraphSOS):"]
    undirected = G.to_undirected(as_view=True)
    seen: set[tuple] = set()

    for src in sorted(G.nodes()):
        for dst in sorted(G.nodes()):
            if src >= dst:
                continue
            try:
                for path in nx.all_simple_paths(undirected, src, dst, cutoff=4):
                    key = tuple(path)
                    if key in seen:
                        continue
                    seen.add(key)
                    # Build annotated path string with FK column labels
                    parts = [path[0]]
                    for i in range(1, len(path)):
                        prev, curr = path[i - 1], path[i]
                        if G.has_edge(prev, curr):
                            ed = G[prev][curr]
                            parts.append(f"-[{ed['from_col']}={ed['to_col']}]-> {curr}")
                        elif G.has_edge(curr, prev):
                            ed = G[curr][prev]
                            parts.append(f"<-[{ed['to_col']}={ed['from_col']}]- {curr}")
                        else:
                            parts.append(f"-- {curr}")
                    lines.append("  " + " ".join(parts))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

    if len(lines) == 1:
        lines.append("  (no FK relationships found)")
    return "\n".join(lines)


# ─── Graph Serialization: MODE 2 — GNN ────────────────────────────────────────
# Computes degree, PageRank, clustering per table and adds structural tags.
# Maps to agents/tools/gnn_adapter.py (structural bias via graph features).

def serialize_gnn(G: nx.DiGraph) -> str:
    """GNNAdapter approach: annotate each table with structural graph features."""
    if len(G) == 0:
        return "Schema Graph: (no tables found)"

    undirected = G.to_undirected()
    degrees  = dict(G.degree())
    # pagerank requires scipy; fall back to normalized degree centrality
    try:
        pagerank = nx.pagerank(G, max_iter=100) if len(G) > 1 else {n: 1.0 for n in G}
    except Exception:
        pagerank = nx.degree_centrality(G)
    clustering = nx.clustering(undirected) if len(G) > 1 else {n: 0.0 for n in G}

    lines = ["Schema Graph — Structural Features (GNN):"]
    for table in sorted(G.nodes()):
        deg = degrees.get(table, 0)
        pr  = pagerank.get(table, 0.0)
        cl  = clustering.get(table, 0.0)

        # Structural role tags
        tags = []
        if deg >= 3:
            tags.append("hub")
        elif deg == 0:
            tags.append("standalone")
        if pr > 0.2:
            tags.append("high-importance")

        tag_str = f", {','.join(tags)}" if tags else ""
        lines.append(f"  {table} [degree:{deg}, pagerank:{pr:.3f}, clustering:{cl:.3f}{tag_str}]")

        # Outgoing FK edges
        for _, dst, data in G.out_edges(table, data=True):
            lines.append(f"    FK OUT: {table}.{data['from_col']} -> {dst}.{data['to_col']}")
        # Incoming FK edges
        for src, _, data in G.in_edges(table, data=True):
            lines.append(f"    FK IN:  {src}.{data['from_col']} -> {table}.{data['to_col']}")

    return "\n".join(lines)


# ─── Graph Serialization: MODE 3 — Translator ─────────────────────────────────
# Template-based edge list. Maps to agents/tools/graph_translator.py.
# Formats each FK edge as "{src} --[from_col=to_col]--> {dst}".

def serialize_translator(G: nx.DiGraph) -> str:
    """GraphTranslator approach: emit every FK edge using a fixed template."""
    if len(G) == 0:
        return "Schema Graph: (no tables found)"

    lines = ["Schema Graph — FK Relationships (Translator):"]
    edges = sorted(G.edges(data=True), key=lambda e: (e[0], e[1]))
    for src, dst, data in edges:
        lines.append(f"  {src} --[{data['from_col']}={data['to_col']}]--> {dst}")

    if len(lines) == 1:
        lines.append("  (no FK relationships defined in schema)")
    return "\n".join(lines)


def get_graph_context(mode: str, db_path: str) -> str:
    """Build and serialize the schema graph according to the chosen mode."""
    G = build_schema_graph(db_path)
    if mode == "graph_sos":
        return serialize_graph_sos(G)
    if mode == "gnn":
        return serialize_gnn(G)
    if mode == "translator":
        return serialize_translator(G)
    raise ValueError(f"Unknown mode: {mode!r}")


# ─── LLM Clients (identical to hitl_sql_testbed.py) ──────────────────────────

class AIAgent:
    """Stateful multi-turn session."""

    def __init__(self):
        self.history: list[dict] = []

    def reset(self):
        self.history = []

    def send(self, user_message: str) -> dict[str, Any]:
        self.history.append({"role": "user", "content": user_message})
        raw = _call_lmstudio(
            model=AI_AGENT_MODEL,
            system=AI_AGENT_SYSTEM,
            messages=self.history,
            temperature=0.1,
            max_tokens=512,
        )
        self.history.append({"role": "assistant", "content": raw})
        return _parse_agent_json(raw)


class HumanProxy:
    """Stateless Oracle simulator."""

    def respond(self, ai_message: str, oracle_ctx: dict) -> str:
        system = HUMAN_PROXY_SYSTEM_TEMPLATE.format(
            answer_key_data=json.dumps(oracle_ctx, indent=2)
        )
        if HUMAN_PROXY_PROVIDER == "anthropic":
            return _call_anthropic(ANTHROPIC_MODEL, system, ai_message)
        return _call_lmstudio(
            model=HUMAN_PROXY_MODEL,
            system=system,
            messages=[{"role": "user", "content": ai_message}],
            temperature=0.3,
            max_tokens=256,
        )


def _call_lmstudio(model: str, system: str, messages: list[dict],
                   temperature: float = 0.1, max_tokens: int = 512) -> str:
    payload = [{"role": "system", "content": system}] + messages
    try:
        resp = requests.post(
            f"{LMSTUDIO_BASE}/chat/completions",
            json={"model": model, "messages": payload,
                  "temperature": temperature, "max_tokens": max_tokens, "stream": False},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot reach LM Studio at http://localhost:1234 — is it running?")


def _call_anthropic(model: str, system: str, user_message: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic && export ANTHROPIC_API_KEY=sk-ant-...")
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=model, max_tokens=256, system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return msg.content[0].text.strip()


def _parse_agent_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    stripped = text.strip()
    if re.match(r"(?i)^(SELECT|WITH|INSERT|UPDATE|DELETE)\b", stripped):
        return {"thought": "", "clarification_needed": False, "question": "", "sql": stripped}
    return {"thought": text, "clarification_needed": True, "question": stripped, "sql": ""}


def _extract_sql(raw: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else raw.strip()


# ─── Execution & Matching (identical to hitl_sql_testbed.py) ──────────────────

def execute_sql(sql: str, db_path: str) -> tuple[list, str]:
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors="replace")
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return [list(r) for r in rows], ""
    except Exception as e:
        return [], str(e)


def normalize(val: Any) -> str:
    return str(val).strip().lower()


def results_match(pred: list, gold: list) -> bool:
    def sort_key(row):
        return [normalize(v) for v in row]
    try:
        pred_s = sorted([list(r) for r in pred], key=sort_key)
        gold_s = sorted([list(r) for r in gold], key=sort_key)
        if len(pred_s) != len(gold_s):
            return False
        for pr, gr in zip(pred_s, gold_s):
            if len(pr) != len(gr):
                return False
            if any(normalize(p) != normalize(g) for p, g in zip(pr, gr)):
                return False
        return True
    except Exception:
        return False


def get_schema_from_db(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
    schemas = [row[0] for row in cursor.fetchall() if row[0]]
    conn.close()
    return "\n\n".join(schemas)


# ─── Graph-Augmented HITL Loop ────────────────────────────────────────────────

def run_graph_hitl_loop(entry: dict, agent: AIAgent, proxy: HumanProxy,
                         mode: str) -> dict:
    """HITL loop identical to hitl_sql_testbed.py with one change:
    the schema graph context is injected into the first user message."""
    question     = entry["question"]
    db_id        = entry["db_id"]
    schema       = entry.get("schema", "")
    gold_sql     = entry["gold_sql"]
    gold_results = entry["gold_results"]
    db_path      = entry.get("db_path", "")

    agent.reset()
    logger.info(f"  Q: {question}")
    logger.info(f"  Gold: {gold_sql}")

    # ── Build graph context and inject into first message (Option A) ──────────
    graph_context = get_graph_context(mode, db_path) if db_path else ""

    current_message = (
        f"Database: {db_id}\n\n"
        f"Schema:\n{schema}\n\n"
        f"{graph_context}\n\n"
        f"User question: {question}"
    )

    final_sql  = ""
    passed     = False
    iterations = 0
    turns: list[dict] = []

    for iteration in range(MAX_ITERATIONS):
        iterations += 1

        # ── AI_Agent ─────────────────────────────────────────────────────────
        agent_resp = agent.send(current_message)
        turn: dict[str, Any] = {"iteration": iteration + 1, "agent_response": agent_resp}

        # ── Clarification branch ─────────────────────────────────────────────
        if agent_resp.get("clarification_needed"):
            clarification_q = (agent_resp.get("question") or
                               agent_resp.get("thought", "Please clarify.")).strip()
            logger.info(f"  [iter {iteration+1}] AI asks: {clarification_q[:80]}")
            oracle_ctx = {"question": question, "gold_sql": gold_sql,
                          "gold_results": gold_results[:5]}
            proxy_reply = proxy.respond(clarification_q, oracle_ctx)
            logger.info(f"  [iter {iteration+1}] Proxy: {proxy_reply[:80]}")
            turn["clarification_question"] = clarification_q
            turn["proxy_reply"]            = proxy_reply
            turns.append(turn)
            current_message = proxy_reply
            continue

        # ── SQL branch ───────────────────────────────────────────────────────
        predicted_sql = _extract_sql(agent_resp.get("sql", "").strip())
        if not predicted_sql:
            logger.warning(f"  [iter {iteration+1}] Empty SQL — stopping")
            turn["error"] = "empty_sql"
            turns.append(turn)
            break

        final_sql = predicted_sql
        logger.info(f"  [iter {iteration+1}] SQL: {predicted_sql[:100]}")

        if db_path and os.path.exists(db_path):
            pred_results, exec_err = execute_sql(predicted_sql, db_path)

            if exec_err:
                turn["exec_error"] = exec_err
                oracle_ctx = {"question": question, "gold_sql": gold_sql,
                              "gold_results": gold_results[:5], "execution_error": exec_err}
                proxy_feedback = proxy.respond(
                    f"Your SQL raised an execution error: {exec_err}\nAnalyze and fix it.",
                    oracle_ctx,
                )
            elif results_match(pred_results, gold_results):
                logger.info(f"  [iter {iteration+1}] PASS ✓")
                turn["result"]       = "PASS"
                turn["pred_results"] = pred_results[:5]
                turns.append(turn)
                passed = True
                break
            else:
                logger.info(
                    f"  [iter {iteration+1}] FAIL  pred={pred_results[:2]}  gold={gold_results[:2]}"
                )
                turn["result"]       = "FAIL"
                turn["pred_results"] = pred_results[:5]
                oracle_ctx = {"question": question, "gold_sql": gold_sql,
                              "gold_results": gold_results[:5],
                              "predicted_results": pred_results[:5]}
                proxy_feedback = proxy.respond(
                    f"I ran your SQL and got:\n{pred_results[:5]}\n"
                    f"This does not match the expected output.",
                    oracle_ctx,
                )
        else:
            if predicted_sql.strip().lower() == gold_sql.strip().lower():
                passed = True
                turn["result"] = "PASS (string match)"
                turns.append(turn)
                break
            oracle_ctx = {"question": question, "gold_sql": gold_sql}
            proxy_feedback = proxy.respond(f"Your SQL is:\n{predicted_sql}", oracle_ctx)
            turn["result"] = "FAIL (string match)"

        logger.info(f"  [iter {iteration+1}] Feedback: {proxy_feedback[:80]}")
        turn["proxy_feedback"] = proxy_feedback
        turns.append(turn)

        if iteration < MAX_ITERATIONS - 1:
            current_message = proxy_feedback

    return {
        "id":         entry["id"],
        "question":   question,
        "db_id":      db_id,
        "gold_sql":   gold_sql,
        "final_sql":  final_sql,
        "passed":     passed,
        "iterations": iterations,
        "turns":      turns,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True,
                   choices=["graph_sos", "gnn", "translator"],
                   help="Graph serialization approach to use")
    p.add_argument("--hitl_results", default="hitl_results.json",
                   help="Previous HITL results file (source of failed samples)")
    p.add_argument("--answer_key", default="answer_key.json",
                   help="Answer key JSON (for schema / gold data)")
    p.add_argument("--db_root", default=DB_ROOT,
                   help="Root dir containing SQLite databases")
    p.add_argument("--output", default=None,
                   help="Output file (default: graph_hitl_results_{mode}.json)")
    return p.parse_args()


def main():
    args  = parse_args()
    output_path = args.output or f"graph_hitl_results_{args.mode}.json"

    # ── Load the 38 failed samples from the previous run ─────────────────────
    with open(args.hitl_results) as f:
        prev = json.load(f)
    failed_samples = [r for r in prev["results"] if not r["passed"]]
    logger.info(f"Loaded {len(failed_samples)} failed samples from {args.hitl_results}")

    # ── Attach schema + db_path from answer_key (schema already in entry) ────
    with open(args.answer_key) as f:
        answer_key = json.load(f)
    ak_by_id = {str(e["id"]): e for e in answer_key}

    samples = []
    for r in failed_samples:
        sid   = str(r["id"])
        entry = ak_by_id.get(sid, {})
        db_id = r["db_id"]
        db_path = os.path.join(args.db_root, db_id, f"{db_id}.sqlite")
        samples.append({
            "id":           r["id"],
            "question":     r["question"],
            "db_id":        db_id,
            "gold_sql":     r["gold_sql"],
            "gold_results": entry.get("gold_results", []),
            "schema":       entry.get("schema", get_schema_from_db(db_path)
                                      if os.path.exists(db_path) else ""),
            "db_path":      db_path,
        })

    total   = len(samples)
    agent   = AIAgent()
    proxy   = HumanProxy()
    results = []
    start   = time.time()

    logger.info(f"Starting graph-augmented retest — mode={args.mode!r}  samples={total}")

    for i, entry in enumerate(samples):
        logger.info(f"\n[{i+1}/{total}]  {entry['id']}  —  {entry['question']}")
        try:
            result = run_graph_hitl_loop(entry, agent, proxy, mode=args.mode)
        except Exception as exc:
            logger.error(f"Error on {entry['id']}: {exc}")
            result = {
                "id": entry["id"], "question": entry["question"],
                "db_id": entry["db_id"], "gold_sql": entry["gold_sql"],
                "final_sql": "", "passed": False, "iterations": 0,
                "turns": [], "error": str(exc),
            }
        results.append(result)

        # Checkpoint after every sample
        with open(output_path, "w") as f:
            json.dump({"mode": args.mode, "results": results}, f, indent=2)

        # Progress line
        done   = i + 1
        passed = sum(1 for r in results if r["passed"])
        elapsed = time.time() - start
        rate   = elapsed / done
        eta    = str(timedelta(seconds=int(rate * (total - done))))
        print(f"  [{done:>2}/{total}]  acc={100*passed/done:5.1f}%  "
              f"pass={passed}  ETA={eta}  ({elapsed/60:.1f} min elapsed)", flush=True)

    # ── Final Report ──────────────────────────────────────────────────────────
    elapsed_total = time.time() - start
    passed_ids    = [r["id"] for r in results if r["passed"]]
    failed_ids    = [r["id"] for r in results if not r["passed"]]

    import re as _re
    def join_count(sql):
        return len(_re.findall(r'\bJOIN\b', sql, _re.IGNORECASE))

    fixed_by_joins = {}
    for r in results:
        if r["passed"]:
            jc = join_count(r["gold_sql"])
            fixed_by_joins[jc] = fixed_by_joins.get(jc, 0) + 1

    fixed_at_iter = {}
    for r in results:
        if r["passed"]:
            it = str(r["iterations"])
            fixed_at_iter[it] = fixed_at_iter.get(it, 0) + 1

    summary = {
        "mode":           args.mode,
        "total_retested": total,
        "passed":         len(passed_ids),
        "failed":         len(failed_ids),
        "accuracy":       len(passed_ids) / total if total else 0,
        "baseline_on_these_38": 0.0,   # all 38 failed previously
        "delta":          len(passed_ids) / total if total else 0,
        "fixed_at_iteration": fixed_at_iter,
        "fixed_by_join_depth": fixed_by_joins,
        "newly_passing_ids":   passed_ids,
        "still_failing_ids":   failed_ids,
        "elapsed_seconds":     int(elapsed_total),
        "elapsed_human":       str(timedelta(seconds=int(elapsed_total))),
    }

    with open(output_path, "w") as f:
        json.dump({"mode": args.mode, "summary": summary, "results": results}, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Mode:              {args.mode}")
    print(f"  Samples retested:  {total}")
    print(f"  Passed:            {len(passed_ids)} / {total}  ({100*summary['accuracy']:.1f}%)")
    print(f"  Baseline on these: 0 / {total}  (0.0%)")
    print(f"  Delta:             +{100*summary['accuracy']:.1f} pp")
    print(f"  Fixed at iter:     {fixed_at_iter}")
    print(f"  Fixed by joins:    {fixed_by_joins}")
    print(f"  Runtime:           {summary['elapsed_human']}")
    print(f"  Results saved →    {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
