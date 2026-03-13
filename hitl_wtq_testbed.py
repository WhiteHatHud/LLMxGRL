#!/usr/bin/env python3
"""
HITL WikiTableQuestions (WTQ) Testbed
======================================
Human-in-the-Loop open-ended table QA using a Create-Verify-Refine loop.

WikiTableQuestions task: given a Wikipedia table and a natural-language question,
extract or compute the correct answer (entity name, number, date, list, etc.).

Domain profile:
  - 4,344 test entries across wildly diverse Wikipedia table topics
    (sports, athletics, tennis, motorsport, politics, geography, history, science)
  - ~48% numeric answers (counting, ranking, aggregation)
  - ~2.6% multi-value answers (comma-separated in seq_out)
  - Gold answers normalised to lowercase in seq_out

  AI_Agent    : Qwen 2.5 Coder (LM Studio) — table-grounded factoid QA
  Human_Proxy : Same model — Oracle simulator pointing to specific cells/rows

Usage
-----
# First run
python hitl_wtq_testbed.py \\
  --wtq_path StructGPT-Ollama/data/wtq/wikitq_test.json \\
  --limit 250

# Resume
python hitl_wtq_testbed.py \\
  --wtq_path StructGPT-Ollama/data/wtq/wikitq_test.json \\
  --limit 250 --resume

# Use a pre-built answer key
python hitl_wtq_testbed.py --answer_key wtq_answer_key.json
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import timedelta
from typing import Any

import requests

# ─── Configuration ────────────────────────────────────────────────────────────

LMSTUDIO_BASE = "http://localhost:1234/v1"

AI_AGENT_MODEL    = "qwen/qwen2.5-coder-14b"
HUMAN_PROXY_MODEL = "qwen/qwen2.5-coder-14b"

HUMAN_PROXY_PROVIDER = "lmstudio"   # "lmstudio" | "anthropic"
ANTHROPIC_MODEL      = "claude-3-5-sonnet-20241022"

MAX_ITERATIONS  = 3
REQUEST_TIMEOUT = 180

BASELINE_ACCURACY = 0.0   # update once a no-HITL baseline run is available

# ─── System Prompts ───────────────────────────────────────────────────────────

AI_AGENT_SYSTEM = """You are a Wikipedia Table Question-Answering Expert. You answer questions by carefully reading structured tables that come from real Wikipedia articles. Topics span sports, politics, geography, history, science, and more.

Your Strategy:
1. Analyse: Identify which columns and rows are relevant. For counting questions, scan every row. For ranking questions, compare values. For "who/what/when/where" questions, locate the matching cell directly.
2. Clarify: If the question is genuinely ambiguous (e.g. the column name is unclear), ask one targeted question before answering.
3. Refine: If feedback tells you your answer is wrong, re-read the table carefully and correct your answer.

Answer rules:
- For single-value answers: give the exact value as it appears in the table (preserve original capitalisation, units, dates).
- For multi-value answers (e.g. "list all X that…"): separate values with " | " (pipe).
- For counting/numeric questions: give only the number, no units unless the question asks for units.
- Do NOT add explanation in the answer field — put reasoning in "thought" only.

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{"thought": "...", "clarification_needed": true, "question": "...", "answer": ""}

OR

{"thought": "...", "clarification_needed": false, "question": "", "answer": "Italy"}"""

HUMAN_PROXY_SYSTEM_TEMPLATE = """You are an Oracle User Simulator for a Wikipedia table QA task. Your goal is to guide an AI assistant to the correct answer using the Answer Key below.

Knowledge Base (Answer Key Entry):
{answer_key_data}

Instructions:
- If the AI asks a clarifying question, answer concisely using only the Answer Key.
- If the AI gives an answer, compare it to the gold answer in the Answer Key.
- If the AI is wrong, give specific feedback pointing to the exact row/column/cell that proves the correct answer (e.g., "Row 3 shows Italy, not Spain" or "You need to count all rows where Position = 1st").
- Do NOT give away the full answer directly — guide the AI to find it in the table.
- Keep your response to 2-3 sentences maximum."""

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── LLM Clients ──────────────────────────────────────────────────────────────

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
    """Stateless Oracle."""

    def respond(self, ai_message: str, answer_key_entry: dict) -> str:
        display_entry = {
            k: v for k, v in answer_key_entry.items()
            if k != "table"  # keep prompt small; table already given to AI
        }
        system = HUMAN_PROXY_SYSTEM_TEMPLATE.format(
            answer_key_data=json.dumps(display_entry, indent=2)
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


# ─── API Helpers ──────────────────────────────────────────────────────────────

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
    """Robustly extract the agent's JSON response."""
    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. JSON in markdown fence
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 3. First {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # 4. "Answers: X" pattern (matches the original StructGPT prompt format)
    m = re.search(r"[Aa]nswers?:\s*(.+)", text)
    if m:
        return {"thought": text, "clarification_needed": False, "question": "", "answer": m.group(1).strip()}
    # 5. Treat as clarification
    return {"thought": text, "clarification_needed": True, "question": text.strip(), "answer": ""}


# ─── Table Serialization ──────────────────────────────────────────────────────

def serialize_table(table: dict) -> str:
    """Convert header+rows to a readable string for the LLM prompt."""
    header = table.get("header", [])
    rows   = table.get("rows",   [])
    lines  = []
    for idx, row in enumerate(rows):
        pairs = [f"({col}, {val})" for col, val in zip(header, row)]
        lines.append(f"item {idx+1}: " + "; ".join(pairs))
    return "\n".join(lines)


# ─── Answer Normalisation & Matching ─────────────────────────────────────────

def _normalise(s: str) -> str:
    """Lowercase, strip, collapse whitespace, remove trailing punctuation."""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".")
    return s


def _split_multi(s: str) -> list[str]:
    """Split a multi-value answer — supports both ' | ' and ', ' separators."""
    if " | " in s:
        return [_normalise(v) for v in s.split(" | ")]
    return [_normalise(v) for v in s.split(", ")]


def answers_match(pred_raw: str, gold_seq_out: str) -> bool:
    """
    Compare predicted answer to gold seq_out.

    gold_seq_out: lowercase, comma-separated for multi-value (from wikitq_test.json)
    pred_raw:     AI answer, may use ' | ' separator for multi-value

    Matching strategy:
    1. Exact normalised match
    2. Set match (order-insensitive, for multi-value)
    3. Numeric equivalence (strip commas: 100,000 == 100000)
    """
    pred_norm = _normalise(pred_raw)
    gold_norm = _normalise(gold_seq_out)

    # 1. Exact
    if pred_norm == gold_norm:
        return True

    # 2. Multi-value set match
    pred_parts = sorted(_split_multi(pred_norm))
    gold_parts = sorted(_split_multi(gold_norm))
    if pred_parts == gold_parts:
        return True

    # 3. Numeric — strip commas and compare
    def strip_commas(v):
        return v.replace(",", "")
    if strip_commas(pred_norm) == strip_commas(gold_norm):
        return True
    if sorted(strip_commas(p) for p in pred_parts) == \
       sorted(strip_commas(g) for g in gold_parts):
        return True

    return False


# ─── Answer Key Builder ───────────────────────────────────────────────────────

def build_answer_key(
    wtq_path: str,
    limit: int = 250,
    output_path: str = "wtq_answer_key.json",
) -> list[dict]:
    """Build the answer key directly from wikitq_test.json."""
    with open(wtq_path, "r") as f:
        all_data = json.load(f)

    if isinstance(all_data, dict):
        items = list(all_data.values())
    else:
        items = all_data

    answer_key: list[dict] = []
    for i, sample in enumerate(items):
        if len(answer_key) >= limit:
            break
        answer_key.append({
            "id":           str(sample.get("id", f"wtq_{i:05d}")),
            "question":     sample.get("question", sample.get("text_in", "")),
            "gold_answers": sample["answer_text"],          # list of strings (original case)
            "gold_seq_out": sample["seq_out"].strip(),      # lowercase, comma-separated
            "table":        sample["table"],
            "table_id":     sample.get("table_id", ""),
        })

    logger.info(f"Built answer key: {len(answer_key)} entries")
    with open(output_path, "w") as f:
        json.dump(answer_key, f, indent=2)
    logger.info(f"Saved → {output_path}")
    return answer_key


def load_answer_key(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ─── Checkpointing ────────────────────────────────────────────────────────────

def load_checkpoint(output_path: str) -> list[dict]:
    if not os.path.exists(output_path):
        return []
    try:
        with open(output_path) as f:
            data = json.load(f)
        completed = data.get("results", [])
        logger.info(f"Resuming from checkpoint: {len(completed)} already completed")
        return completed
    except Exception:
        return []


def save_checkpoint(output_path: str, results: list[dict],
                    total: int, config: dict) -> None:
    done   = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    with open(output_path, "w") as f:
        json.dump({
            "config": config,
            "progress": {
                "completed": done,
                "total": total,
                "current_accuracy": round(passed / max(done, 1), 4),
            },
            "summary": None,
            "results": results,
        }, f, indent=2)


def finalize_output(output_path: str, results: list[dict],
                    elapsed_s: float, config: dict) -> None:
    total    = len(results)
    passed   = sum(1 for r in results if r.get("passed"))
    avg_it   = sum(r.get("iterations", 0) for r in results) / max(total, 1)
    hitl_acc = passed / max(total, 1)

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
    if BASELINE_ACCURACY > 0:
        delta_sign = "+" if summary["delta"] >= 0 else ""
        print(f"  Baseline      : ({100*BASELINE_ACCURACY:.1f}%)")
        print(f"  Delta         : {delta_sign}{100*summary['delta']:.1f} pp")
    print(f"  Avg iterations: {avg_it:.2f}")
    print(f"  Time elapsed  : {summary['elapsed_human']}")
    print(f"  Passes by iter: {fixed_at}")
    print(f"{'='*64}\n")
    logger.info(f"Results saved → {output_path}")


# ─── Core HITL Loop ───────────────────────────────────────────────────────────

def run_hitl_loop(entry: dict, agent: AIAgent, proxy: HumanProxy) -> dict:
    """
    Create-Verify-Refine loop for one WTQ entry.

    Each iteration:
      AI_Agent reads the table and answers the question →
        clarification_needed → Human_Proxy answers → feed back to AI
        answer produced      → compare to gold (normalised exact/set/numeric match)
          PASS → done
          FAIL → Human_Proxy provides cell-level feedback → next iteration
    """
    question     = entry["question"]
    gold_seq_out = entry["gold_seq_out"]
    gold_answers = entry["gold_answers"]
    table        = entry["table"]

    agent.reset()

    logger.info(f"  Q: {question}")
    logger.info(f"  Gold: {gold_seq_out}")

    serialized = serialize_table(table)
    current_message = (
        f"Table:\n{serialized}\n\n"
        f"Question: {question}"
    )

    final_answer = ""
    passed       = False
    iterations   = 0
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
                "gold_answers": gold_answers,
                "gold_seq_out": gold_seq_out,
            }
            proxy_reply = proxy.respond(clarification_q, oracle_ctx)
            logger.info(f"  [iter {iteration+1}] Proxy: {proxy_reply[:80]}")

            turn["clarification_question"] = clarification_q
            turn["proxy_reply"]            = proxy_reply
            turns.append(turn)
            current_message = proxy_reply
            continue

        # ── Answer branch ─────────────────────────────────────────────────────
        raw_answer = agent_resp.get("answer", "").strip()
        if not raw_answer:
            # fallback: scan thought for an "Answers:" line
            m = re.search(r"[Aa]nswers?:\s*(.+)", agent_resp.get("thought", ""))
            if m:
                raw_answer = m.group(1).strip()

        if not raw_answer:
            logger.warning(f"  [iter {iteration+1}] Empty answer — stopping")
            turn["error"] = "empty_answer"
            turns.append(turn)
            break

        final_answer = raw_answer
        logger.info(f"  [iter {iteration+1}] Answer: {raw_answer}")

        if answers_match(raw_answer, gold_seq_out):
            logger.info(f"  [iter {iteration+1}] PASS ✓")
            turn["result"] = "PASS"
            turns.append(turn)
            passed = True
            break
        else:
            logger.info(
                f"  [iter {iteration+1}] FAIL  "
                f"pred={_normalise(raw_answer)!r}  gold={gold_seq_out!r}"
            )
            turn["result"]      = "FAIL"
            turn["pred_answer"] = raw_answer

            oracle_ctx = {
                "question":     question,
                "gold_answers": gold_answers,
                "gold_seq_out": gold_seq_out,
            }
            proxy_feedback = proxy.respond(
                f"My answer was: \"{raw_answer}\". "
                f"Is that correct? If not, please point me to the specific row(s) "
                f"or column(s) in the table that I should look at.",
                oracle_ctx,
            )
            logger.info(f"  [iter {iteration+1}] Feedback: {proxy_feedback[:80]}")
            turn["proxy_feedback"] = proxy_feedback
            turns.append(turn)

            if iteration < MAX_ITERATIONS - 1:
                current_message = (
                    f"Table:\n{serialized}\n\n"
                    f"Question: {question}\n\n"
                    f"Feedback on my previous answer (\"{raw_answer}\"): {proxy_feedback}"
                )

    return {
        "id":           entry["id"],
        "question":     question,
        "gold_seq_out": gold_seq_out,
        "final_answer": final_answer,
        "passed":       passed,
        "iterations":   iterations,
        "turns":        turns,
    }


# ─── Progress Printer ─────────────────────────────────────────────────────────

def print_progress(i: int, total: int, results: list[dict],
                   start_time: float) -> None:
    done    = i + 1
    passed  = sum(1 for r in results if r.get("passed"))
    elapsed = time.time() - start_time
    rate    = elapsed / done
    eta     = str(timedelta(seconds=int(rate * (total - done))))
    print(
        f"  [{done:>3}/{total}]  acc={100*passed/done:5.1f}%  "
        f"pass={passed}  ETA={eta}  ({elapsed/60:.1f} min elapsed)",
        flush=True,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HITL WikiTableQuestions Testbed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--wtq_path", metavar="JSON",
                     help="Path to wikitq_test.json")
    src.add_argument("--answer_key", metavar="JSON",
                     help="Use a pre-built answer key JSON")

    p.add_argument("--output",     default="hitl_wtq_results.json")
    p.add_argument("--key_output", default="wtq_answer_key.json")
    p.add_argument("--limit",      type=int, default=250)
    p.add_argument("--resume",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Build / load answer key ───────────────────────────────────────────────
    if args.wtq_path:
        answer_key = build_answer_key(args.wtq_path, args.limit, args.key_output)
    elif args.answer_key:
        answer_key = load_answer_key(args.answer_key)
        logger.info(f"Loaded {len(answer_key)} entries from {args.answer_key}")
    else:
        default = "StructGPT-Ollama/data/wtq/wikitq_test.json"
        if not os.path.exists(default):
            logger.error("Specify --wtq_path or --answer_key")
            return
        answer_key = build_answer_key(default, args.limit, args.key_output)

    samples = answer_key[:args.limit]
    total   = len(samples)

    # ── Resume ────────────────────────────────────────────────────────────────
    completed_results: list[dict] = []
    completed_ids: set[str] = set()
    if args.resume:
        completed_results = load_checkpoint(args.output)
        completed_ids     = {r["id"] for r in completed_results}
        samples = [s for s in samples if s["id"] not in completed_ids]
        logger.info(
            f"Resuming: {len(completed_ids)} done, {len(samples)} remaining of {total}"
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
        "task":               "wikitableqa",
    }

    print(f"\n{'='*64}")
    print(f"  Task        : WikiTableQuestions (open-ended table QA)")
    print(f"  AI_Agent    : {AI_AGENT_MODEL}")
    print(f"  Human_Proxy : {proxy_label}")
    print(f"  Max iters   : {MAX_ITERATIONS}")
    print(f"  Samples     : {len(samples)}  (total target: {total})")
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
                "id":           entry["id"],
                "question":     entry["question"],
                "gold_seq_out": entry["gold_seq_out"],
                "final_answer": "",
                "passed":       False,
                "iterations":   0,
                "error":        str(exc),
            }

        results.append(result)
        print_progress(len(results) - 1, total, results, start_time)
        save_checkpoint(args.output, results, total, config)

    elapsed = time.time() - start_time
    finalize_output(args.output, results, elapsed, config)


if __name__ == "__main__":
    main()
