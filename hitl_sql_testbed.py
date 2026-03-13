#!/usr/bin/env python3
"""
HITL SQL Generation Testbed
============================
Human-in-the-Loop SQL generation using a Create-Verify-Refine loop.

  AI_Agent    : Qwen 2.5 Coder (LM Studio) — translates NL → SQL
  Human_Proxy : Qwen 2.5 Coder (LM Studio) OR Claude 3.5 (Anthropic) — Oracle simulator

Usage — compare against the 250-sample baseline
------------------------------------------------
# Step 1: build answer key directly from the existing baseline run
python hitl_sql_testbed.py --from_baseline StructGPT-Ollama/outputs/spider/output_sample_250_qwen.jsonl

# Step 2: resume a previous run if it was interrupted
python hitl_sql_testbed.py --from_baseline ... --resume

# Build answer key from raw Spider dev.json instead (no prior baseline needed)
python hitl_sql_testbed.py --build_key --key_limit 250

# Use Claude 3.5 as Human_Proxy
#   pip install anthropic && export ANTHROPIC_API_KEY=sk-ant-...
#   Set HUMAN_PROXY_PROVIDER = "anthropic" below
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import requests

# ─── Configuration ────────────────────────────────────────────────────────────

LMSTUDIO_BASE = "http://localhost:1234/v1"

AI_AGENT_MODEL    = "qwen/qwen2.5-coder-14b"
HUMAN_PROXY_MODEL = "qwen/qwen2.5-coder-14b"

# Switch to "anthropic" to use Claude 3.5 as Human_Proxy
# Requires: pip install anthropic && export ANTHROPIC_API_KEY=sk-ant-...
HUMAN_PROXY_PROVIDER = "lmstudio"   # "lmstudio" | "anthropic"
ANTHROPIC_MODEL      = "claude-3-5-sonnet-20241022"

MAX_ITERATIONS  = 3
REQUEST_TIMEOUT = 180   # seconds

BASELINE_ACCURACY = 0.84   # 84% — used in the final comparison report

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


# ─── LLM Clients ──────────────────────────────────────────────────────────────

class AIAgent:
    """Stateful multi-turn session — maintains conversation history across iterations."""

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
    """Stateless Oracle — each call is independent, grounded by the answer key."""

    def respond(self, ai_message: str, answer_key_entry: dict) -> str:
        display_entry = {
            k: (v[:5] if k == "gold_results" else v)
            for k, v in answer_key_entry.items()
            if k != "schema"
        }
        system = HUMAN_PROXY_SYSTEM_TEMPLATE.format(
            answer_key_data=json.dumps(display_entry, indent=2)
        )
        if HUMAN_PROXY_PROVIDER == "anthropic":
            return _call_anthropic(
                model=ANTHROPIC_MODEL,
                system=system,
                user_message=ai_message,
            )
        return _call_lmstudio(
            model=HUMAN_PROXY_MODEL,
            system=system,
            messages=[{"role": "user", "content": ai_message}],
            temperature=0.3,
            max_tokens=256,
        )


# ─── API Helpers ──────────────────────────────────────────────────────────────

def _call_lmstudio(model: str, system: str, messages: list[dict],
                   temperature: float = 0.1, max_tokens: int = 512) -> str:
    payload_messages = [{"role": "system", "content": system}] + messages
    try:
        resp = requests.post(
            f"{LMSTUDIO_BASE}/chat/completions",
            json={
                "model": model,
                "messages": payload_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot reach LM Studio at http://localhost:1234 — is it running?"
        )


def _call_anthropic(model: str, system: str, user_message: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "anthropic package not installed.\n"
            "Run:  pip install anthropic\n"
            "Then: export ANTHROPIC_API_KEY=sk-ant-..."
        )
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=model,
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return msg.content[0].text.strip()


def _parse_agent_json(text: str) -> dict[str, Any]:
    """Robustly extract the agent's JSON response, with multiple fallback strategies."""
    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. JSON inside a markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First {...} block in the text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # 4. Raw SQL — treat as a direct answer
    stripped = text.strip()
    if re.match(r"(?i)^(SELECT|WITH|INSERT|UPDATE|DELETE)\b", stripped):
        return {"thought": "", "clarification_needed": False, "question": "", "sql": stripped}

    # 5. Treat as a clarification question
    return {"thought": text, "clarification_needed": True, "question": stripped, "sql": ""}


def _extract_sql(raw: str) -> str:
    """Strip markdown code fences from a SQL string."""
    m = re.search(r"```(?:sql)?\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return raw.strip()


# ─── SQL Execution & Comparison ───────────────────────────────────────────────

def execute_sql(sql: str, db_path: str) -> tuple[list, str | None]:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = [list(row) for row in cursor.fetchall()]
        conn.close()
        return rows, None
    except sqlite3.Error as e:
        return [], str(e)


def results_match(pred: list, gold: list) -> bool:
    """Order-insensitive, case-insensitive comparison."""
    def normalize(rows):
        return sorted([sorted([str(v).strip().lower() for v in row]) for row in rows])
    return normalize(pred) == normalize(gold)


def get_schema_from_db(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    )
    schemas = [row[0] for row in cursor.fetchall() if row[0]]
    conn.close()
    return "\n\n".join(schemas)


# ─── Answer Key Builders ──────────────────────────────────────────────────────

def build_answer_key_from_baseline(
    baseline_path: str,
    db_root: str,
    output_path: str = "answer_key.json",
) -> list[dict]:
    """
    Build an answer key directly from an existing baseline JSONL run.
    Uses the pre-computed `answer_text` field as gold results — no re-execution needed.
    """
    with open(baseline_path, "r") as f:
        lines = f.readlines()

    answer_key: list[dict] = []
    skipped = 0

    for i, line in enumerate(lines):
        sample = json.loads(line)
        db_id    = sample["db_id"]
        question = sample["question"]
        gold_sql = sample["query"]

        # answer_text is already the executed gold result from the baseline run
        gold_results = sample.get("answer_text", [])

        db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            logger.warning(f"  [{i}] SQLite not found for {db_id}, skipping")
            skipped += 1
            continue

        schema = get_schema_from_db(db_path)

        answer_key.append({
            "id":              sample.get("id", f"baseline_{i:04d}"),
            "question":        question,
            "db_id":           db_id,
            "gold_sql":        gold_sql,
            "gold_results":    gold_results,
            "schema":          schema,
            "baseline_pred":   sample.get("Prediction", ""),   # keep for comparison
        })

    logger.info(f"Built answer key: {len(answer_key)} entries ({skipped} skipped)")
    with open(output_path, "w") as f:
        json.dump(answer_key, f, indent=2)
    logger.info(f"Saved → {output_path}")
    return answer_key


def build_answer_key(
    spider_dev_path: str,
    db_root: str,
    limit: int = 250,
    output_path: str = "answer_key.json",
) -> list[dict]:
    """Build an answer key from Spider dev.json by executing gold SQL queries."""
    with open(spider_dev_path, "r") as f:
        dev_data = json.load(f)

    answer_key: list[dict] = []
    skipped = 0

    for i, sample in enumerate(dev_data):
        if len(answer_key) >= limit:
            break
        db_id    = sample["db_id"]
        question = sample["question"]
        gold_sql = sample["query"]
        db_path  = os.path.join(db_root, db_id, f"{db_id}.sqlite")

        if not os.path.exists(db_path):
            skipped += 1
            continue

        gold_results, err = execute_sql(gold_sql, db_path)
        if err:
            skipped += 1
            continue

        schema = get_schema_from_db(db_path)
        answer_key.append({
            "id":           f"spider_{i:04d}",
            "question":     question,
            "db_id":        db_id,
            "gold_sql":     gold_sql,
            "gold_results": gold_results,
            "schema":       schema,
        })

    logger.info(f"Built answer key: {len(answer_key)} entries ({skipped} skipped)")
    with open(output_path, "w") as f:
        json.dump(answer_key, f, indent=2)
    logger.info(f"Saved → {output_path}")
    return answer_key


def load_answer_key(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


# ─── Checkpointing ────────────────────────────────────────────────────────────

def load_checkpoint(output_path: str) -> list[dict]:
    """Load already-completed results from a previous run."""
    if not os.path.exists(output_path):
        return []
    try:
        with open(output_path, "r") as f:
            data = json.load(f)
        completed = data.get("results", [])
        logger.info(f"Resuming from checkpoint: {len(completed)} already completed")
        return completed
    except Exception:
        return []


def save_checkpoint(output_path: str, results: list[dict], total: int,
                    config: dict) -> None:
    """Write current results to disk after every sample."""
    done    = len(results)
    passed  = sum(1 for r in results if r.get("passed"))
    avg_it  = sum(r.get("iterations", 0) for r in results) / max(done, 1)

    with open(output_path, "w") as f:
        json.dump({
            "config": config,
            "progress": {
                "completed": done,
                "total": total,
                "current_accuracy": round(passed / max(done, 1), 4),
            },
            "summary": None,   # filled at the end
            "results": results,
        }, f, indent=2)


def finalize_output(output_path: str, results: list[dict],
                    elapsed_s: float, config: dict) -> None:
    """Write the final summary."""
    total   = len(results)
    passed  = sum(1 for r in results if r.get("passed"))
    avg_it  = sum(r.get("iterations", 0) for r in results) / max(total, 1)
    hitl_acc = passed / max(total, 1)

    # Breakdown: how many fixed at each iteration
    fixed_at: dict[int, int] = {}
    for r in results:
        if r.get("passed"):
            it = r.get("iterations", 1)
            fixed_at[it] = fixed_at.get(it, 0) + 1

    summary = {
        "total":              total,
        "passed":             passed,
        "failed":             total - passed,
        "hitl_accuracy":      round(hitl_acc, 4),
        "baseline_accuracy":  BASELINE_ACCURACY,
        "delta":              round(hitl_acc - BASELINE_ACCURACY, 4),
        "avg_iterations":     round(avg_it, 2),
        "fixed_at_iteration": fixed_at,
        "elapsed_seconds":    round(elapsed_s),
        "elapsed_human":      str(timedelta(seconds=int(elapsed_s))),
    }

    with open(output_path, "w") as f:
        json.dump({"config": config, "summary": summary, "results": results}, f, indent=2)

    print(f"\n{'='*64}")
    print(f"  HITL accuracy : {passed}/{total}  ({100*hitl_acc:.1f}%)")
    print(f"  Baseline      : ({100*BASELINE_ACCURACY:.1f}%)")
    delta_sign = "+" if summary["delta"] >= 0 else ""
    print(f"  Delta         : {delta_sign}{100*summary['delta']:.1f} pp")
    print(f"  Avg iterations: {avg_it:.2f}")
    print(f"  Time elapsed  : {summary['elapsed_human']}")
    print(f"  Passes by iter: {fixed_at}")
    print(f"{'='*64}\n")
    logger.info(f"Results saved → {output_path}")


# ─── Core HITL Loop ───────────────────────────────────────────────────────────

def run_hitl_loop(entry: dict, agent: AIAgent, proxy: HumanProxy) -> dict:
    """
    Create-Verify-Refine loop for one answer key entry.

    Each iteration:
      AI_Agent responds →
        clarification_needed → Human_Proxy answers → feed back to AI
        SQL produced         → execute & compare
          PASS → done
          FAIL → Human_Proxy provides targeted feedback → next iteration
    """
    question     = entry["question"]
    db_id        = entry["db_id"]
    schema       = entry.get("schema", "")
    gold_sql     = entry["gold_sql"]
    gold_results = entry["gold_results"]
    db_path      = entry.get("db_path", "")

    agent.reset()

    logger.info(f"  Q: {question}")
    logger.info(f"  Gold: {gold_sql}")

    current_message = (
        f"Database: {db_id}\n\n"
        f"Schema:\n{schema}\n\n"
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

            oracle_ctx = {
                "question":     question,
                "gold_sql":     gold_sql,
                "gold_results": gold_results[:5],
            }
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
                oracle_ctx = {
                    "question":        question,
                    "gold_sql":        gold_sql,
                    "gold_results":    gold_results[:5],
                    "execution_error": exec_err,
                }
                proxy_feedback = proxy.respond(
                    f"Your SQL raised an execution error: {exec_err}\n"
                    f"Analyze and fix it.",
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
                oracle_ctx = {
                    "question":          question,
                    "gold_sql":          gold_sql,
                    "gold_results":      gold_results[:5],
                    "predicted_results": pred_results[:5],
                }
                proxy_feedback = proxy.respond(
                    f"I ran your SQL and got:\n{pred_results[:5]}\n"
                    f"This does not match the expected output.",
                    oracle_ctx,
                )

        else:
            # No DB available — string comparison fallback
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


# ─── Progress Printer ─────────────────────────────────────────────────────────

def print_progress(i: int, total: int, results: list[dict],
                   start_time: float) -> None:
    done    = i + 1
    passed  = sum(1 for r in results if r.get("passed"))
    elapsed = time.time() - start_time
    rate    = elapsed / done          # seconds per sample
    eta_s   = rate * (total - done)
    eta     = str(timedelta(seconds=int(eta_s)))
    acc     = 100 * passed / done

    print(
        f"  [{done:>3}/{total}]  acc={acc:5.1f}%  "
        f"pass={passed}  ETA={eta}  "
        f"({elapsed/60:.1f} min elapsed)",
        flush=True,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HITL SQL Testbed — Create-Verify-Refine loop vs 84% baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input sources (mutually exclusive)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--from_baseline", metavar="JSONL",
                     help="Build answer key from an existing baseline output JSONL "
                          "(e.g. output_sample_250_qwen.jsonl)")
    src.add_argument("--answer_key", metavar="JSON",
                     help="Use a pre-built answer key JSON")
    src.add_argument("--build_key", action="store_true",
                     help="Build answer key from raw Spider dev.json")

    p.add_argument("--db_root",
                   default="StructGPT-Ollama/data/spider/spider_data/database",
                   help="Root dir containing SQLite databases")
    p.add_argument("--output", default="hitl_results.json",
                   help="Output file (checkpointed after each sample)")
    p.add_argument("--limit", type=int, default=250,
                   help="Max samples to run (default: 250)")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-completed samples from a previous run")

    # Raw Spider key builder options
    p.add_argument("--spider_dev",
                   default="StructGPT-Ollama/data/spider/spider_data/dev.json")
    p.add_argument("--key_output", default="answer_key.json")
    p.add_argument("--key_limit", type=int, default=250)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Build / load answer key ───────────────────────────────────────────────
    if args.from_baseline:
        logger.info(f"Building answer key from baseline: {args.from_baseline}")
        answer_key = build_answer_key_from_baseline(
            baseline_path=args.from_baseline,
            db_root=args.db_root,
            output_path=args.key_output,
        )
    elif args.build_key:
        logger.info("Building answer key from Spider dev.json …")
        answer_key = build_answer_key(
            spider_dev_path=args.spider_dev,
            db_root=args.db_root,
            limit=args.key_limit,
            output_path=args.key_output,
        )
    elif args.answer_key:
        answer_key = load_answer_key(args.answer_key)
        logger.info(f"Loaded {len(answer_key)} entries from {args.answer_key}")
    else:
        logger.error(
            "Specify an input source:\n"
            "  --from_baseline <baseline.jsonl>   (recommended)\n"
            "  --answer_key <key.json>\n"
            "  --build_key"
        )
        return

    # Attach db_path + schema if missing
    for entry in answer_key:
        db_id   = entry["db_id"]
        db_path = os.path.join(args.db_root, db_id, f"{db_id}.sqlite")
        entry["db_path"] = db_path
        if "schema" not in entry and os.path.exists(db_path):
            entry["schema"] = get_schema_from_db(db_path)
        elif "schema" not in entry:
            entry["schema"] = ""

    samples = answer_key[:args.limit]
    total   = len(samples)

    # ── Resume: skip already-done samples ────────────────────────────────────
    completed_results: list[dict] = []
    completed_ids: set[str] = set()

    if args.resume:
        completed_results = load_checkpoint(args.output)
        completed_ids     = {r["id"] for r in completed_results}
        samples = [s for s in samples if s["id"] not in completed_ids]
        logger.info(
            f"Resuming: {len(completed_ids)} done, "
            f"{len(samples)} remaining out of {total} total"
        )

    # ── Init agents ───────────────────────────────────────────────────────────
    agent = AIAgent()
    proxy = HumanProxy()

    proxy_label = (
        f"{ANTHROPIC_MODEL} (Anthropic)"
        if HUMAN_PROXY_PROVIDER == "anthropic"
        else f"{HUMAN_PROXY_MODEL} (LM Studio)"
    )

    config = {
        "ai_agent":           AI_AGENT_MODEL,
        "human_proxy":        proxy_label,
        "human_proxy_source": HUMAN_PROXY_PROVIDER,
        "max_iterations":     MAX_ITERATIONS,
        "baseline_accuracy":  BASELINE_ACCURACY,
    }

    print(f"\n{'='*64}")
    print(f"  AI_Agent    : {AI_AGENT_MODEL}")
    print(f"  Human_Proxy : {proxy_label}")
    print(f"  Max iters   : {MAX_ITERATIONS}")
    print(f"  Samples     : {len(samples)}  (total target: {total})")
    print(f"  Baseline    : {100*BASELINE_ACCURACY:.0f}%  (target to beat)")
    print(f"  Output      : {args.output}")
    print(f"{'='*64}\n")

    # ── Run ───────────────────────────────────────────────────────────────────
    results: list[dict] = list(completed_results)
    start_time = time.time()

    for i, entry in enumerate(samples):
        logger.info(f"\n[{len(results)+1}/{total}]  {entry['id']}  —  {entry['question']}")
        try:
            result = run_hitl_loop(entry, agent, proxy)
        except Exception as exc:
            logger.error(f"Error on {entry['id']}: {exc}")
            result = {
                "id":       entry["id"],
                "question": entry["question"],
                "db_id":    entry.get("db_id", ""),
                "gold_sql": entry.get("gold_sql", ""),
                "final_sql": "",
                "passed":   False,
                "iterations": 0,
                "error":    str(exc),
            }

        results.append(result)
        print_progress(len(results) - 1, total, results, start_time)

        # Checkpoint after every sample
        save_checkpoint(args.output, results, total, config)

    # ── Final report ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    finalize_output(args.output, results, elapsed, config)


if __name__ == "__main__":
    main()
