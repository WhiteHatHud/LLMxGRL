#!/usr/bin/env python3
"""
Single-Pass Baseline
====================
Clean no-HITL baseline for TabFact and WTQ.
Uses identical model, prompts, table serialization, and answer matching
as the HITL testbeds — the ONLY difference is zero feedback iterations.

Usage
-----
python single_pass_baseline.py --dataset tabfact \\
  --answer_key tabfact_answer_key.json \\
  --output baseline_tabfact_results.json \\
  --limit 250

python single_pass_baseline.py --dataset wtq \\
  --answer_key wtq_answer_key.json \\
  --output baseline_wtq_results.json \\
  --limit 250
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

# ─── Configuration (identical to HITL testbeds) ───────────────────────────────

LMSTUDIO_BASE  = "http://localhost:1234/v1"
AI_AGENT_MODEL = "qwen/qwen2.5-coder-14b"
REQUEST_TIMEOUT = 180

# ─── System Prompts (identical to HITL testbeds) ──────────────────────────────

TABFACT_SYSTEM = """You are a Table Fact-Verification Assistant. You determine whether a natural language statement is supported or refuted by a given table.

Your Strategy:
1. Analyze: Read the table carefully. Identify which rows and columns are relevant to the statement. Reason step by step.
2. Clarify: If the statement is genuinely ambiguous and cannot be resolved from the table, ask a targeted question.
3. Refine: If you receive feedback that your verdict is wrong, re-examine the table and revise.

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{"thought": "...", "clarification_needed": true, "question": "...", "verdict": ""}

OR

{"thought": "...", "clarification_needed": false, "question": "", "verdict": "yes"}

The "verdict" field must be exactly "yes" (statement is entailed) or "no" (statement is refuted)."""

WTQ_SYSTEM = """You are a Wikipedia Table Question-Answering Expert. You answer questions by carefully reading structured tables that come from real Wikipedia articles. Topics span sports, politics, geography, history, science, and more.

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── API ──────────────────────────────────────────────────────────────────────

def call_lmstudio(system: str, user_message: str) -> str:
    try:
        resp = requests.post(
            f"{LMSTUDIO_BASE}/chat/completions",
            json={
                "model": AI_AGENT_MODEL,
                "messages": [
                    {"role": "system",  "content": system},
                    {"role": "user",    "content": user_message},
                ],
                "temperature": 0.1,
                "max_tokens": 512,
                "stream": False,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot reach LM Studio at http://localhost:1234")


def parse_json(text: str) -> dict[str, Any]:
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
    return {"thought": text, "verdict": "", "answer": ""}


# ─── Table Serialization (identical to HITL testbeds) ─────────────────────────

def serialize_table(table: dict) -> str:
    header = table.get("header", [])
    rows   = table.get("rows",   [])
    lines  = []
    for idx, row in enumerate(rows):
        pairs = [f"({col}, {val})" for col, val in zip(header, row)]
        lines.append(f"item {idx+1}: " + "; ".join(pairs))
    return "\n".join(lines)


# ─── Answer Matching (identical to HITL testbeds) ─────────────────────────────

def normalize_verdict(raw: str) -> str:
    lower = raw.lower()
    if "yes" in lower and "no" in lower:
        return "unknown"
    if "yes" in lower:
        return "entailed"
    if "no" in lower:
        return "refuted"
    return "unknown"


def verdict_matches(pred_raw: str, gold_seq_out: str) -> bool:
    pred_norm = normalize_verdict(pred_raw)
    gold_lower = gold_seq_out.strip().lower()
    gold_norm = "entailed" if gold_lower in ("yes", "entailed") else \
                "refuted"  if gold_lower in ("no",  "refuted")  else gold_lower
    return pred_norm == gold_norm


def _normalise(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.rstrip(".")


def _split_multi(s: str) -> list[str]:
    if " | " in s:
        return [_normalise(v) for v in s.split(" | ")]
    return [_normalise(v) for v in s.split(", ")]


def answers_match(pred_raw: str, gold_seq_out: str) -> bool:
    pred_norm = _normalise(pred_raw)
    gold_norm = _normalise(gold_seq_out)
    if pred_norm == gold_norm:
        return True
    pred_parts = sorted(_split_multi(pred_norm))
    gold_parts = sorted(_split_multi(gold_norm))
    if pred_parts == gold_parts:
        return True
    def strip_commas(v):
        return v.replace(",", "")
    if strip_commas(pred_norm) == strip_commas(gold_norm):
        return True
    if sorted(strip_commas(p) for p in pred_parts) == \
       sorted(strip_commas(g) for g in gold_parts):
        return True
    return False


# ─── Checkpointing ────────────────────────────────────────────────────────────

def load_checkpoint(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        completed = data.get("results", [])
        logger.info(f"Resuming: {len(completed)} already done")
        return completed
    except Exception:
        return []


def save_checkpoint(path: str, results: list[dict], total: int, config: dict) -> None:
    done   = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    with open(path, "w") as f:
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


def finalize(path: str, results: list[dict], elapsed_s: float, config: dict) -> None:
    total    = len(results)
    passed   = sum(1 for r in results if r.get("passed"))
    accuracy = passed / max(total, 1)

    summary = {
        "total":           total,
        "passed":          passed,
        "failed":          total - passed,
        "accuracy":        round(accuracy, 4),
        "elapsed_seconds": round(elapsed_s),
        "elapsed_human":   str(timedelta(seconds=int(elapsed_s))),
    }

    with open(path, "w") as f:
        json.dump({"config": config, "summary": summary, "results": results}, f, indent=2)

    print(f"\n{'='*64}")
    print(f"  Dataset    : {config['dataset']}")
    print(f"  Model      : {AI_AGENT_MODEL}")
    print(f"  Accuracy   : {passed}/{total}  ({100*accuracy:.1f}%)")
    print(f"  Time       : {summary['elapsed_human']}")
    print(f"{'='*64}\n")
    logger.info(f"Saved → {path}")


# ─── Single-pass inference ────────────────────────────────────────────────────

def run_tabfact_sample(entry: dict) -> dict:
    serialized = serialize_table(entry["table"])
    prompt = f"Table:\n{serialized}\n\nStatement: {entry['statement']}"
    try:
        raw  = call_lmstudio(TABFACT_SYSTEM, prompt)
        resp = parse_json(raw)
        verdict = resp.get("verdict", "").strip()
        if not verdict:
            verdict = resp.get("thought", "")
        passed = verdict_matches(verdict, entry["gold_label"])
        return {
            "id":         entry["id"],
            "statement":  entry["statement"],
            "gold_label": entry["gold_label"],
            "prediction": verdict,
            "normalized": normalize_verdict(verdict),
            "passed":     passed,
        }
    except Exception as e:
        return {
            "id":         entry["id"],
            "statement":  entry["statement"],
            "gold_label": entry["gold_label"],
            "prediction": "",
            "passed":     False,
            "error":      str(e),
        }


def run_wtq_sample(entry: dict) -> dict:
    serialized = serialize_table(entry["table"])
    prompt = f"Table:\n{serialized}\n\nQuestion: {entry['question']}"
    try:
        raw    = call_lmstudio(WTQ_SYSTEM, prompt)
        resp   = parse_json(raw)
        answer = resp.get("answer", "").strip()
        if not answer:
            m = re.search(r"[Aa]nswers?:\s*(.+)", resp.get("thought", ""))
            if m:
                answer = m.group(1).strip()
        passed = answers_match(answer, entry["gold_seq_out"]) if answer else False
        return {
            "id":           entry["id"],
            "question":     entry["question"],
            "gold_seq_out": entry["gold_seq_out"],
            "prediction":   answer,
            "passed":       passed,
        }
    except Exception as e:
        return {
            "id":           entry["id"],
            "question":     entry["question"],
            "gold_seq_out": entry["gold_seq_out"],
            "prediction":   "",
            "passed":       False,
            "error":        str(e),
        }


# ─── Progress ─────────────────────────────────────────────────────────────────

def print_progress(done: int, total: int, passed: int, start_time: float) -> None:
    elapsed = time.time() - start_time
    eta     = str(timedelta(seconds=int((elapsed / done) * (total - done))))
    print(
        f"  [{done:>3}/{total}]  acc={100*passed/done:5.1f}%  "
        f"pass={passed}  ETA={eta}  ({elapsed/60:.1f} min elapsed)",
        flush=True,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-pass baseline (no HITL)")
    p.add_argument("--dataset",    required=True, choices=["tabfact", "wtq"])
    p.add_argument("--answer_key", required=True, help="Pre-built answer key JSON")
    p.add_argument("--output",     required=True, help="Output results JSON")
    p.add_argument("--limit",      type=int, default=250)
    p.add_argument("--resume",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.answer_key) as f:
        answer_key = json.load(f)
    samples = answer_key[:args.limit]
    total   = len(samples)
    logger.info(f"Loaded {total} samples from {args.answer_key}")

    # Resume
    completed_results: list[dict] = []
    completed_ids: set[str] = set()
    if args.resume:
        completed_results = load_checkpoint(args.output)
        completed_ids     = {r["id"] for r in completed_results}
        samples = [s for s in samples if s["id"] not in completed_ids]
        logger.info(f"Resuming: {len(completed_ids)} done, {len(samples)} remaining")

    config = {
        "dataset":    args.dataset,
        "model":      AI_AGENT_MODEL,
        "mode":       "single_pass",
        "iterations": 1,
    }

    print(f"\n{'='*64}")
    print(f"  Mode       : Single-pass (NO HITL — ablation baseline)")
    print(f"  Dataset    : {args.dataset.upper()}")
    print(f"  Model      : {AI_AGENT_MODEL}")
    print(f"  Samples    : {len(samples)}  (target: {total})")
    print(f"  Output     : {args.output}")
    print(f"{'='*64}\n")

    results    = list(completed_results)
    start_time = time.time()

    run_fn = run_tabfact_sample if args.dataset == "tabfact" else run_wtq_sample

    for entry in samples:
        logger.info(f"[{len(results)+1}/{total}]  {entry['id']}")
        result = run_fn(entry)
        results.append(result)
        passed = sum(1 for r in results if r.get("passed"))
        print_progress(len(results), total, passed, start_time)
        save_checkpoint(args.output, results, total, config)

    finalize(args.output, results, time.time() - start_time, config)


if __name__ == "__main__":
    main()
