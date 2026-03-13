#!/usr/bin/env python3
"""
HITL TabFact Testbed
====================
Human-in-the-Loop fact-verification using a Create-Verify-Refine loop.

TabFact task: given a table and a natural-language statement, decide whether
the statement is ENTAILED (yes) or REFUTED (no) by the table.

  AI_Agent    : Qwen 2.5 Coder (LM Studio) — reasons over table, outputs yes/no
  Human_Proxy : Oracle simulator — corrects the AI's table reasoning

Usage
-----
# First run (reads directly from tab_fact_test.json — no prior baseline needed)
python hitl_tabfact_testbed.py \\
  --tabfact_path StructGPT-Ollama/data/tabfact/tab_fact_test.json \\
  --limit 250

# Resume a previous run
python hitl_tabfact_testbed.py \\
  --tabfact_path StructGPT-Ollama/data/tabfact/tab_fact_test.json \\
  --limit 250 --resume

# Use a pre-built answer key instead
python hitl_tabfact_testbed.py --answer_key tabfact_answer_key.json
"""

import argparse
import json
import logging
import os
import re
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
REQUEST_TIMEOUT = 180

# Tabfact published StructGPT baseline for reference (update once you have a run)
BASELINE_ACCURACY = 0.0   # set to your baseline once known

# ─── System Prompts ───────────────────────────────────────────────────────────

AI_AGENT_SYSTEM = """You are a Table Fact-Verification Assistant. You determine whether a natural language statement is supported or refuted by a given table.

Your Strategy:
1. Analyze: Read the table carefully. Identify which rows and columns are relevant to the statement. Reason step by step.
2. Clarify: If the statement is genuinely ambiguous and cannot be resolved from the table, ask a targeted question.
3. Refine: If you receive feedback that your verdict is wrong, re-examine the table and revise.

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{"thought": "...", "clarification_needed": true, "question": "...", "verdict": ""}

OR

{"thought": "...", "clarification_needed": false, "question": "", "verdict": "yes"}

The "verdict" field must be exactly "yes" (statement is entailed) or "no" (statement is refuted)."""

HUMAN_PROXY_SYSTEM_TEMPLATE = """You are an Oracle User Simulator. Your goal is to help an AI assistant correctly verify a statement against a table.

Knowledge Base (Answer Key Entry):
{answer_key_data}

Instructions:
- If the AI asks a clarifying question, answer it concisely using only the Answer Key.
- If the AI gives a verdict, compare it to the gold label in your Answer Key.
- If the AI is wrong, give specific feedback about which table values contradict the AI's reasoning (e.g., "Row 3 shows value X, not Y" or "You missed that column Z shows...").
- Do not reveal the correct verdict directly; guide the AI to reason it out.
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
            k: v for k, v in answer_key_entry.items()
            if k != "table"   # omit the full table to keep the prompt small
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
    """Robustly extract the agent's JSON response."""
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

    # 3. First {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # 4. Plain yes/no
    lower = text.strip().lower()
    if lower.startswith("yes"):
        return {"thought": text, "clarification_needed": False, "question": "", "verdict": "yes"}
    if lower.startswith("no"):
        return {"thought": text, "clarification_needed": False, "question": "", "verdict": "no"}

    # 5. Treat as a clarification question
    return {"thought": text, "clarification_needed": True, "question": text.strip(), "verdict": ""}


# ─── Table Serialization ──────────────────────────────────────────────────────

def serialize_table(table: dict) -> str:
    """Convert header+rows dict to a readable string for the LLM prompt."""
    header = table.get("header", [])
    rows   = table.get("rows", [])
    lines  = []
    for idx, row in enumerate(rows):
        pairs = [f"({col}, {val})" for col, val in zip(header, row)]
        lines.append(f"item {idx+1}: " + "; ".join(pairs))
    return "\n".join(lines)


# ─── Verdict Normalization & Comparison ──────────────────────────────────────

def normalize_verdict(raw: str) -> str:
    """Map raw LLM output to 'entailed', 'refuted', or 'unknown'."""
    lower = raw.lower()
    has_yes = "yes" in lower
    has_no  = "no"  in lower
    if has_yes and has_no:
        return "unknown"
    if has_yes:
        return "entailed"
    if has_no:
        return "refuted"
    return "unknown"


def verdict_matches(pred_raw: str, gold_seq_out: str) -> bool:
    """
    gold_seq_out is the raw `seq_out` field from tab_fact_test.json.
    Typical values: 'yes'/'no' OR 'entailed'/'refuted' depending on dataset version.
    We normalize both sides before comparing.
    """
    pred_norm = normalize_verdict(pred_raw)
    gold_lower = gold_seq_out.strip().lower()
    # handle both label conventions
    if gold_lower in ("yes", "entailed"):
        gold_norm = "entailed"
    elif gold_lower in ("no", "refuted"):
        gold_norm = "refuted"
    else:
        gold_norm = gold_lower
    return pred_norm == gold_norm


# ─── Answer Key Builder ───────────────────────────────────────────────────────

def build_answer_key_from_tabfact(
    tabfact_path: str,
    limit: int = 250,
    output_path: str = "tabfact_answer_key.json",
) -> list[dict]:
    """
    Build the answer key directly from tab_fact_test.json.
    No SQL execution needed — the gold label is `seq_out` in the file.
    """
    with open(tabfact_path, "r") as f:
        all_data = json.load(f)

    # tab_fact_test.json can be a list or a dict keyed by id
    if isinstance(all_data, dict):
        items = list(all_data.values())
    else:
        items = all_data

    answer_key: list[dict] = []
    for i, sample in enumerate(items):
        if len(answer_key) >= limit:
            break
        answer_key.append({
            "id":         str(sample.get("id", f"tabfact_{i:05d}")),
            "statement":  sample.get("statement", sample.get("question", "")),
            "gold_label": sample["seq_out"].strip().lower(),   # "yes"/"no" or "entailed"/"refuted"
            "table":      sample["table"],
        })

    logger.info(f"Built answer key: {len(answer_key)} entries")
    with open(output_path, "w") as f:
        json.dump(answer_key, f, indent=2)
    logger.info(f"Saved → {output_path}")
    return answer_key


def load_answer_key(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


# ─── Checkpointing ────────────────────────────────────────────────────────────

def load_checkpoint(output_path: str) -> list[dict]:
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
    Create-Verify-Refine loop for one TabFact entry.

    Each iteration:
      AI_Agent reasons over the table →
        clarification_needed → Human_Proxy answers → feed back to AI
        verdict produced     → compare to gold label
          PASS → done
          FAIL → Human_Proxy provides targeted feedback → next iteration
    """
    statement  = entry["statement"]
    gold_label = entry["gold_label"]
    table      = entry["table"]

    agent.reset()

    logger.info(f"  Statement: {statement}")
    logger.info(f"  Gold label: {gold_label}")

    serialized = serialize_table(table)
    current_message = (
        f"Table:\n{serialized}\n\n"
        f"Statement: {statement}"
    )

    final_verdict = ""
    passed        = False
    iterations    = 0
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
                "statement":  statement,
                "gold_label": gold_label,
            }
            proxy_reply = proxy.respond(clarification_q, oracle_ctx)
            logger.info(f"  [iter {iteration+1}] Proxy: {proxy_reply[:80]}")

            turn["clarification_question"] = clarification_q
            turn["proxy_reply"]            = proxy_reply
            turns.append(turn)
            current_message = proxy_reply
            continue

        # ── Verdict branch ────────────────────────────────────────────────────
        raw_verdict = agent_resp.get("verdict", "").strip()
        if not raw_verdict:
            # fall back: check thought for yes/no
            raw_verdict = agent_resp.get("thought", "")

        if not raw_verdict:
            logger.warning(f"  [iter {iteration+1}] Empty verdict — stopping")
            turn["error"] = "empty_verdict"
            turns.append(turn)
            break

        final_verdict = raw_verdict
        logger.info(f"  [iter {iteration+1}] Verdict: {raw_verdict}")

        if verdict_matches(raw_verdict, gold_label):
            logger.info(f"  [iter {iteration+1}] PASS ✓")
            turn["result"] = "PASS"
            turns.append(turn)
            passed = True
            break
        else:
            normalized = normalize_verdict(raw_verdict)
            logger.info(f"  [iter {iteration+1}] FAIL  pred={normalized}  gold={gold_label}")
            turn["result"]          = "FAIL"
            turn["pred_normalized"] = normalized

            oracle_ctx = {
                "statement":  statement,
                "gold_label": gold_label,
            }
            proxy_feedback = proxy.respond(
                f"I evaluated the statement as '{normalized}'. "
                f"Is that correct? If not, which part of the table did I misread?",
                oracle_ctx,
            )
            logger.info(f"  [iter {iteration+1}] Feedback: {proxy_feedback[:80]}")
            turn["proxy_feedback"] = proxy_feedback
            turns.append(turn)

            if iteration < MAX_ITERATIONS - 1:
                current_message = (
                    f"Table:\n{serialized}\n\n"
                    f"Statement: {statement}\n\n"
                    f"Feedback: {proxy_feedback}"
                )

    return {
        "id":             entry["id"],
        "statement":      statement,
        "gold_label":     gold_label,
        "final_verdict":  final_verdict,
        "passed":         passed,
        "iterations":     iterations,
        "turns":          turns,
    }


# ─── Progress Printer ─────────────────────────────────────────────────────────

def print_progress(i: int, total: int, results: list[dict],
                   start_time: float) -> None:
    done    = i + 1
    passed  = sum(1 for r in results if r.get("passed"))
    elapsed = time.time() - start_time
    rate    = elapsed / done
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
        description="HITL TabFact Testbed — Create-Verify-Refine loop for table fact verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = p.add_mutually_exclusive_group()
    src.add_argument("--tabfact_path", metavar="JSON",
                     help="Path to tab_fact_test.json "
                          "(default: StructGPT-Ollama/data/tabfact/tab_fact_test.json)")
    src.add_argument("--answer_key", metavar="JSON",
                     help="Use a pre-built answer key JSON")

    p.add_argument("--output",  default="hitl_tabfact_results.json",
                   help="Output file (checkpointed after each sample)")
    p.add_argument("--key_output", default="tabfact_answer_key.json",
                   help="Where to save the built answer key")
    p.add_argument("--limit",   type=int, default=250,
                   help="Max samples to run (default: 250)")
    p.add_argument("--resume",  action="store_true",
                   help="Skip already-completed samples from a previous run")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Build / load answer key ───────────────────────────────────────────────
    if args.tabfact_path:
        tabfact_path = args.tabfact_path
        logger.info(f"Building answer key from: {tabfact_path}")
        answer_key = build_answer_key_from_tabfact(
            tabfact_path=tabfact_path,
            limit=args.limit,
            output_path=args.key_output,
        )
    elif args.answer_key:
        answer_key = load_answer_key(args.answer_key)
        logger.info(f"Loaded {len(answer_key)} entries from {args.answer_key}")
    else:
        default = "StructGPT-Ollama/data/tabfact/tab_fact_test.json"
        if not os.path.exists(default):
            logger.error(
                "Specify an input source:\n"
                "  --tabfact_path <tab_fact_test.json>   (recommended)\n"
                "  --answer_key <key.json>"
            )
            return
        logger.info(f"Using default path: {default}")
        answer_key = build_answer_key_from_tabfact(
            tabfact_path=default,
            limit=args.limit,
            output_path=args.key_output,
        )

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
        "task":               "tabfact",
    }

    print(f"\n{'='*64}")
    print(f"  Task        : TabFact (table fact verification)")
    print(f"  AI_Agent    : {AI_AGENT_MODEL}")
    print(f"  Human_Proxy : {proxy_label}")
    print(f"  Max iters   : {MAX_ITERATIONS}")
    print(f"  Samples     : {len(samples)}  (total target: {total})")
    if BASELINE_ACCURACY > 0:
        print(f"  Baseline    : {100*BASELINE_ACCURACY:.0f}%  (target to beat)")
    print(f"  Output      : {args.output}")
    print(f"{'='*64}\n")

    # ── Run ───────────────────────────────────────────────────────────────────
    results: list[dict] = list(completed_results)
    start_time = time.time()

    for i, entry in enumerate(samples):
        logger.info(f"\n[{len(results)+1}/{total}]  {entry['id']}  —  {entry['statement']}")
        try:
            result = run_hitl_loop(entry, agent, proxy)
        except Exception as exc:
            logger.error(f"Error on {entry['id']}: {exc}")
            result = {
                "id":            entry["id"],
                "statement":     entry["statement"],
                "gold_label":    entry["gold_label"],
                "final_verdict": "",
                "passed":        False,
                "iterations":    0,
                "error":         str(exc),
            }

        results.append(result)
        print_progress(len(results) - 1, total, results, start_time)
        save_checkpoint(args.output, results, total, config)

    elapsed = time.time() - start_time
    finalize_output(args.output, results, elapsed, config)


if __name__ == "__main__":
    main()
