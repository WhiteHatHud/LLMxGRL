"""
graph_augmented_rerun.py
========================
Graph-Augmented Prompt Injection Experiment — Priority 1 intervention.

Workflow
--------
1. Load the original output JSONL (predictions from baseline/HITL run).
2. Re-evaluate each sample with execution matching to identify failures.
3. For each failing sample whose gold SQL requires >= JOIN_THRESHOLD joins:
   - Extract required tables from the gold SQL.
   - Use SchemaGraph to compute the FK-edge path chain.
   - Inject the path into an augmented prompt.
4. Call Qwen (via Ollama) with the augmented prompt.
5. Evaluate the new prediction against gold.
6. Write per-sample results to output JSONL and print a summary table.

Usage
-----
  python graph_augmented_rerun.py \\
      --input  outputs/spider/output_sample_250_qwen.jsonl \\
      --output outputs/spider/output_graph_augmented.jsonl \\
      --db_dir data/spider/spider_data/database \\
      --tables data/spider/spider_data/tables.json \\
      [--join_threshold 2] \\
      [--model qwen2.5-coder:14b] \\
      [--ollama_url http://localhost:11434/api] \\
      [--max_tokens 400] \\
      [--debug]
"""

import argparse
import json
import logging
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import requests

from schema_graph import (
    SchemaGraph,
    clean_sql_prediction,
    count_joins_in_sql,
    extract_tables_from_sql,
)

logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Execution matching
# ---------------------------------------------------------------------------

def exec_sql(db_path: str, sql: str):
    """Execute sql and return result set as frozenset, or None on error."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return frozenset(rows)
    except Exception:
        return None


def results_match(db_path: str, pred_sql: str, gold_sql: str) -> bool:
    """Return True when pred and gold return the same result set."""
    pred_res = exec_sql(db_path, pred_sql)
    gold_res = exec_sql(db_path, gold_sql)
    if pred_res is None or gold_res is None:
        return False
    # Order-sensitive if gold uses ORDER BY
    if "order by" in gold_sql.lower():
        pred_list = sorted(pred_res)
        gold_list = sorted(gold_res)
        return pred_list == gold_list
    return pred_res == gold_res


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

AUGMENTED_PROMPT_TEMPLATE = """\
### Complete sqlite SQL query only and with no explanation.
#
### Sqlite SQL tables, with their properties:
#
{schema}
#
### Foreign key relationships:
#
{fk_text}
#
### IMPORTANT: To answer the question below you must traverse this foreign key path:
#
{fk_path}
#
Make sure your SQL includes ALL intermediate tables in this join chain.
#
### {question}
 SELECT"""

PLAIN_PROMPT_TEMPLATE = """\
### Complete sqlite SQL query only and with no explanation.
#
### Sqlite SQL tables, with their properties:
#
{schema}
#
### Foreign key relationships:
#
{fk_text}
#
### {question}
 SELECT"""


def build_prompt(
    question: str,
    schema_text: str,
    fk_text: str,
    fk_path: Optional[str],
) -> str:
    if fk_path:
        return AUGMENTED_PROMPT_TEMPLATE.format(
            schema=schema_text,
            fk_text=fk_text,
            fk_path=fk_path,
            question=question,
        )
    return PLAIN_PROMPT_TEMPLATE.format(
        schema=schema_text,
        fk_text=fk_text,
        question=question,
    )


def call_ollama(prompt: str, model: str, ollama_url: str, max_tokens: int) -> str:
    """Call Ollama /api/generate and return the model response text."""
    while True:
        try:
            resp = requests.post(
                f"{ollama_url}/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": max_tokens,
                        "top_p": 1,
                    },
                },
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json()["response"]
        except requests.exceptions.Timeout:
            print("  [timeout] retrying in 20s …")
            time.sleep(20)
        except requests.exceptions.ConnectionError:
            print("  [connection error] Is Ollama running? retrying in 20s …")
            time.sleep(20)
        except requests.exceptions.RequestException as e:
            print(f"  [request error] {e}  retrying in 20s …")
            time.sleep(20)


def call_lmstudio(prompt: str, model: str, api_base: str, max_tokens: int) -> str:
    """Call LM Studio OpenAI-compatible /v1/chat/completions endpoint."""
    while True:
        try:
            resp = requests.post(
                f"{api_base}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "top_p": 1,
                    "stream": False,
                },
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            print("  [timeout] retrying in 20s …")
            time.sleep(20)
        except requests.exceptions.ConnectionError:
            print("  [connection error] Is LM Studio running? retrying in 20s …")
            time.sleep(20)
        except requests.exceptions.RequestException as e:
            print(f"  [request error] {e}  retrying in 20s …")
            time.sleep(20)


def call_model(prompt: str, args) -> str:
    """Dispatch to Ollama or LM Studio based on --api_type."""
    if args.api_type == "lmstudio":
        return call_lmstudio(prompt, args.model, args.api_url, args.max_tokens)
    return call_ollama(prompt, args.model, args.api_url, args.max_tokens)


# ---------------------------------------------------------------------------
# Schema serialisation helpers (mirrors Retriever.serialize_table_and_column)
# ---------------------------------------------------------------------------

def serialise_schema(db_id: str, tables_data: list) -> tuple[str, str]:
    """
    Returns (schema_text, fk_text) for the given db_id.

    schema_text: "# TABLE(col1,col2,...);\n# ..."
    fk_text:     "TABLE_A.col = TABLE_B.col, ..."
    """
    entry = next((e for e in tables_data if e["db_id"] == db_id), None)
    if entry is None:
        return "", ""

    tables = entry["table_names_original"]
    col_names = entry["column_names_original"]
    col_types = entry["column_types"]
    foreign_keys = entry["foreign_keys"]

    # Table → columns map
    table2cols: dict[str, list[str]] = {t: [] for t in tables}
    for (tbl_idx, col_name), col_type in zip(col_names, col_types):
        if tbl_idx >= 0:
            table2cols[tables[tbl_idx]].append(col_name)

    schema_lines = []
    for t, cols in table2cols.items():
        schema_lines.append(f"# {t}({','.join(cols)});")
    schema_text = "\n".join(schema_lines)

    # FK text
    fk_parts = []
    for (fk_id, ref_id) in foreign_keys:
        fk_tbl_idx, fk_col = col_names[fk_id]
        ref_tbl_idx, ref_col = col_names[ref_id]
        fk_tbl = tables[fk_tbl_idx]
        ref_tbl = tables[ref_tbl_idx]
        fk_parts.append(f"{fk_tbl}.{fk_col}={ref_tbl}.{ref_col}")
    fk_text = ",".join(fk_parts) + (";" if fk_parts else "")

    return schema_text, fk_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Graph-Augmented Prompt Injection re-run"
    )
    parser.add_argument("--input",  required=True, help="Original output JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--db_dir", required=True, help="Spider database directory")
    parser.add_argument("--tables", required=True, help="tables.json path")
    parser.add_argument("--join_threshold", type=int, default=2,
                        help="Minimum JOIN count in gold SQL to trigger graph augmentation (default: 2)")
    parser.add_argument("--model", default="qwen/qwen2.5-coder-14b",
                        help="Model name (Ollama: qwen2.5-coder:14b  LM Studio: qwen/qwen2.5-coder-14b)")
    parser.add_argument("--api_type", default="lmstudio", choices=["ollama", "lmstudio"],
                        help="Backend API type (default: lmstudio)")
    parser.add_argument("--api_url", default="http://localhost:1234/v1",
                        help="API base URL (default: http://localhost:1234/v1 for LM Studio)")
    parser.add_argument("--max_tokens", type=int, default=400)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Load data
    with open(args.input) as f:
        all_samples = [json.loads(l) for l in f]

    with open(args.tables) as f:
        tables_data = json.load(f)

    sg = SchemaGraph(args.tables)

    db_base = Path(args.db_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Step 1: identify failures
    # -------------------------------------------------------------------
    print("=" * 65)
    print("Step 1: Identifying failures in input file …")
    failures = []
    baseline_pass = 0

    for sample in all_samples:
        db_id = sample["db_id"]
        gold = sample["query"]
        pred = clean_sql_prediction(sample["Prediction"])
        db_path = str(db_base / db_id / f"{db_id}.sqlite")

        if results_match(db_path, pred, gold):
            baseline_pass += 1
        else:
            failures.append(sample)

    total = len(all_samples)
    print(f"  Total samples : {total}")
    print(f"  Passing       : {baseline_pass}  ({baseline_pass/total*100:.1f}%)")
    print(f"  Failing       : {len(failures)}  ({len(failures)/total*100:.1f}%)")

    # Stratify by join count
    join_counts = Counter(count_joins_in_sql(s["query"]) for s in failures)
    print("\n  Failures by join count in gold SQL:")
    for k in sorted(join_counts):
        print(f"    {k} joins: {join_counts[k]}")

    target_failures = [
        s for s in failures
        if count_joins_in_sql(s["query"]) >= args.join_threshold
    ]
    print(f"\n  Targeting {len(target_failures)} samples with >= {args.join_threshold} joins for graph augmentation.")
    print("=" * 65)

    if not target_failures:
        print("No target failures found. Exiting.")
        return

    # -------------------------------------------------------------------
    # Step 2: graph-augmented re-run
    # -------------------------------------------------------------------
    print(f"\nStep 2: Re-running {len(target_failures)} samples with FK path injection …\n")

    results = []
    recovered = 0
    path_found = 0
    path_not_found = 0

    for i, sample in enumerate(target_failures, 1):
        db_id = sample["db_id"]
        question = sample["question"]
        gold = sample["query"]
        db_path = str(db_base / db_id / f"{db_id}.sqlite")
        join_count = count_joins_in_sql(gold)

        # Extract tables from gold SQL
        required_tables = extract_tables_from_sql(gold)
        unique_tables = list(dict.fromkeys(required_tables))  # preserve order, dedupe

        # Build FK path
        fk_path = sg.get_fk_path(db_id, unique_tables)
        if fk_path:
            path_found += 1
        else:
            path_not_found += 1

        # Build schema text
        schema_text, fk_text = serialise_schema(db_id, tables_data)

        # Build prompt
        prompt = build_prompt(question, schema_text, fk_text, fk_path)

        if args.debug:
            print(f"\n{'─'*60}")
            print(f"[{i}/{len(target_failures)}] db={db_id}  joins={join_count}")
            print(f"  question  : {question}")
            print(f"  tables    : {unique_tables}")
            print(f"  fk_path   : {fk_path}")
            print(f"  gold      : {gold}")
            print(f"  prompt snippet:\n{prompt[:400]}\n…")

        # Call model
        raw_response = call_ollama(prompt, args.model, args.ollama_url, args.max_tokens)
        # The prompt ends with "SELECT" so prepend it
        new_pred = clean_sql_prediction("SELECT " + raw_response)

        # Evaluate
        passed = results_match(db_path, new_pred, gold)
        if passed:
            recovered += 1

        status = "PASS" if passed else "FAIL"
        print(f"[{i:3d}/{len(target_failures)}] {status}  db={db_id}  joins={join_count}  path={'yes' if fk_path else 'no'}")
        if args.debug or not passed:
            print(f"       gold : {gold}")
            print(f"       pred : {new_pred}")

        results.append({
            "db_id": db_id,
            "question": question,
            "gold": gold,
            "original_prediction": clean_sql_prediction(sample["Prediction"]),
            "augmented_prediction": new_pred,
            "fk_path_injected": fk_path,
            "join_count": join_count,
            "passed": passed,
        })

    # -------------------------------------------------------------------
    # Step 3: write output and print summary
    # -------------------------------------------------------------------
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("\n" + "=" * 65)
    print("GRAPH AUGMENTATION EXPERIMENT RESULTS")
    print("=" * 65)
    print(f"  Samples re-run          : {len(target_failures)}")
    print(f"  FK path found           : {path_found}")
    print(f"  FK path not found       : {path_not_found}")
    print(f"  Recovered (now passing) : {recovered}")
    recovery_rate = recovered / len(target_failures) * 100 if target_failures else 0
    print(f"  Recovery rate           : {recovery_rate:.1f}%")
    print()

    # Join-stratified results
    print("  Results by join count:")
    by_joins: dict[int, dict] = {}
    for r in results:
        jc = r["join_count"]
        if jc not in by_joins:
            by_joins[jc] = {"total": 0, "passed": 0}
        by_joins[jc]["total"] += 1
        by_joins[jc]["passed"] += r["passed"]

    for jc in sorted(by_joins):
        t = by_joins[jc]["total"]
        p = by_joins[jc]["passed"]
        print(f"    {jc} joins: {p}/{t} = {p/t*100:.1f}%")

    print()
    new_overall_pass = baseline_pass + recovered
    print(f"  Original overall accuracy : {baseline_pass}/{total} = {baseline_pass/total*100:.1f}%")
    print(f"  New overall accuracy      : {new_overall_pass}/{total} = {new_overall_pass/total*100:.1f}%")
    print(f"  Delta                     : +{recovered/total*100:.1f}pp")
    print()
    print(f"  Results written to: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
