#!/usr/bin/env python3
"""
hitl_full_1034.py
=================
Full 1034-sample Spider dev evaluation with the Qwen 2.5-based HITL pipeline.

Builds an answer key from Spider dev.json (executing gold SQL to get gold results),
then runs the Create-Verify-Refine loop with per-sample timing and checkpointing.

Usage
-----
# First run — builds key + runs HITL from scratch
python hitl_full_1034.py --build_key

# Resume after any interruption (safe to re-run repeatedly)
python hitl_full_1034.py --resume

# Build key only (no HITL yet)
python hitl_full_1034.py --build_key --key_only

Time estimate
-------------
~35 s / sample  ×  1034 samples  ≈  10 hours total
Runs overnight; Ctrl-C at any point, resume with --resume.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from hitl_sql_testbed import (
    AIAgent,
    HumanProxy,
    run_hitl_loop,
    build_answer_key,
    load_answer_key,
    get_schema_from_db,
    AI_AGENT_MODEL,
    HUMAN_PROXY_MODEL,
    HUMAN_PROXY_PROVIDER,
    ANTHROPIC_MODEL,
    MAX_ITERATIONS,
)

# ─── Paths ────────────────────────────────────────────────────────────────────

SPIDER_DEV     = "StructGPT-Ollama/data/spider/spider_data/dev.json"
DB_ROOT        = "StructGPT-Ollama/data/spider/spider_data/database"
ANSWER_KEY_OUT = "answer_key_1034.json"
RESULTS_OUT    = "hitl_results_1034.json"
LOG_FILE       = "hitl_full_1034.log"

# Empirical estimate: ablation hard samples = ~45 s, easier samples faster.
# Weighted average over the full dev set ≈ 35 s/sample.
SECS_PER_SAMPLE_ESTIMATE = 35

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def load_checkpoint(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        done = data.get("results", [])
        logger.info(f"Checkpoint found: {len(done)} samples already completed")
        return done
    except Exception as exc:
        logger.warning(f"Could not read checkpoint ({exc}) — starting fresh")
        return []


def save_checkpoint(path: str, results: list[dict], total: int, config: dict) -> None:
    done   = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    with open(path, "w") as f:
        json.dump(
            {
                "config":   config,
                "progress": {
                    "completed":        done,
                    "total":            total,
                    "current_accuracy": round(passed / max(done, 1), 4),
                },
                "summary": None,   # filled at the end
                "results":  results,
            },
            f,
            indent=2,
        )


def finalize(path: str, results: list[dict], elapsed_s: float, config: dict) -> None:
    total  = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    failed = total - passed
    acc    = passed / max(total, 1)
    avg_it = sum(r.get("iterations", 0) for r in results) / max(total, 1)

    fixed_at: dict[int, int] = {}
    for r in results:
        if r.get("passed"):
            it = r.get("iterations", 1)
            fixed_at[it] = fixed_at.get(it, 0) + 1

    sample_times = [r["elapsed_seconds"] for r in results if "elapsed_seconds" in r]
    avg_s  = sum(sample_times) / max(len(sample_times), 1)
    min_s  = min(sample_times, default=0)
    max_s  = max(sample_times, default=0)

    # Error rate by type
    errors = [r.get("error") for r in results if r.get("error")]

    summary = {
        "total":                   total,
        "passed":                  passed,
        "failed":                  failed,
        "accuracy":                round(acc, 4),
        "avg_iterations":          round(avg_it, 2),
        "fixed_at_iteration":      fixed_at,
        "elapsed_seconds":         round(elapsed_s),
        "elapsed_human":           str(timedelta(seconds=int(elapsed_s))),
        "avg_seconds_per_sample":  round(avg_s, 1),
        "min_seconds_per_sample":  round(min_s, 1),
        "max_seconds_per_sample":  round(max_s, 1),
        "error_count":             len(errors),
    }

    with open(path, "w") as f:
        json.dump({"config": config, "summary": summary, "results": results}, f, indent=2)

    print(f"\n{'='*64}")
    print(f"  FINAL RESULTS — 1034-sample HITL (Qwen 2.5 Coder 14B)")
    print(f"{'='*64}")
    print(f"  Samples   : {total}")
    print(f"  Passed    : {passed}  ({100*acc:.1f}%)")
    print(f"  Failed    : {failed}")
    print(f"  Avg iters : {avg_it:.2f}")
    print(f"  Time/smpl : avg={avg_s:.1f}s  min={min_s:.1f}s  max={max_s:.1f}s")
    print(f"  Total time: {summary['elapsed_human']}")
    print(f"  Fixed@iter: {fixed_at}")
    if errors:
        print(f"  Errors    : {len(errors)} samples errored (see log)")
    print(f"{'='*64}\n")
    logger.info(f"Final results saved → {path}")


# ─── Progress display ─────────────────────────────────────────────────────────

def print_progress(done: int, total: int, results: list[dict],
                   start_time: float, last_s: float) -> None:
    passed  = sum(1 for r in results if r.get("passed"))
    elapsed = time.time() - start_time
    rate    = elapsed / max(done, 1)          # seconds per sample (running avg)
    eta_s   = rate * (total - done)
    print(
        f"  [{done:>4}/{total}]  "
        f"acc={100*passed/max(done,1):5.1f}%  "
        f"pass={passed:<4}  "
        f"last={last_s:5.1f}s  "
        f"avg={rate:5.1f}s  "
        f"ETA={str(timedelta(seconds=int(eta_s)))}  "
        f"({elapsed/3600:.2f}h elapsed)",
        flush=True,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full 1034-sample HITL evaluation — Qwen 2.5 Coder 14B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--build_key", action="store_true",
                   help="Build answer key from Spider dev.json (run once)")
    p.add_argument("--key_only", action="store_true",
                   help="Exit after building the answer key (no HITL)")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-completed samples from checkpoint")
    p.add_argument("--answer_key", default=ANSWER_KEY_OUT,
                   help=f"Answer key path (default: {ANSWER_KEY_OUT})")
    p.add_argument("--output", default=RESULTS_OUT,
                   help=f"Results output path (default: {RESULTS_OUT})")
    p.add_argument("--db_root", default=DB_ROOT,
                   help="Root directory containing Spider SQLite databases")
    p.add_argument("--spider_dev", default=SPIDER_DEV,
                   help="Path to Spider dev.json")
    p.add_argument("--limit", type=int, default=1034,
                   help="Max samples (default: 1034 = full dev set)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Build / load answer key ───────────────────────────────────────────────
    if args.build_key or not os.path.exists(args.answer_key):
        logger.info("Building answer key from Spider dev.json (executes gold SQL) …")
        answer_key = build_answer_key(
            spider_dev_path=args.spider_dev,
            db_root=args.db_root,
            limit=args.limit,
            output_path=args.answer_key,
        )
        logger.info(f"Answer key built: {len(answer_key)} entries → {args.answer_key}")
    else:
        answer_key = load_answer_key(args.answer_key)
        logger.info(f"Loaded {len(answer_key)} entries from {args.answer_key}")

    if args.key_only:
        print(f"Key-only mode: answer key saved to {args.answer_key}")
        return

    # Attach db_path and schema to each entry
    for entry in answer_key:
        db_id   = entry["db_id"]
        db_path = os.path.join(args.db_root, db_id, f"{db_id}.sqlite")
        entry["db_path"] = db_path
        if "schema" not in entry:
            entry["schema"] = get_schema_from_db(db_path) if os.path.exists(db_path) else ""

    samples = answer_key[:args.limit]
    total   = len(samples)

    # ── Resume: skip completed samples ───────────────────────────────────────
    results: list[dict] = []
    completed_ids: set[str] = set()

    if args.resume:
        results = load_checkpoint(args.output)
        completed_ids = {r["id"] for r in results}
        samples = [s for s in samples if s["id"] not in completed_ids]
        logger.info(
            f"Resume: {len(completed_ids)} done, {len(samples)} remaining of {total}"
        )

    remaining = len(samples)

    proxy_label = (
        f"{ANTHROPIC_MODEL} (Anthropic)"
        if HUMAN_PROXY_PROVIDER == "anthropic"
        else f"{HUMAN_PROXY_MODEL} (LM Studio)"
    )
    config = {
        "ai_agent":       AI_AGENT_MODEL,
        "human_proxy":    proxy_label,
        "max_iterations": MAX_ITERATIONS,
        "total_samples":  total,
    }

    est_total   = remaining * SECS_PER_SAMPLE_ESTIMATE
    est_human   = str(timedelta(seconds=est_total))

    print(f"\n{'='*64}")
    print(f"  Full 1034-sample HITL — Qwen 2.5 Coder 14B")
    print(f"{'='*64}")
    print(f"  AI Agent    : {AI_AGENT_MODEL}")
    print(f"  Human Proxy : {proxy_label}")
    print(f"  Max iters   : {MAX_ITERATIONS}")
    print(f"  Total target: {total}")
    print(f"  Remaining   : {remaining}")
    print(f"  Output      : {args.output}")
    print(f"  Log file    : {LOG_FILE}")
    print(f"  Est. time   : ~{est_human}  (@{SECS_PER_SAMPLE_ESTIMATE}s/sample)")
    print(f"  Ctrl-C safe : yes — resume with --resume")
    print(f"{'='*64}\n")

    if remaining == 0:
        logger.info("All samples already completed. Nothing to run.")
        # Still finalize to print the summary.
        finalize(args.output, results, 0.0, config)
        return

    agent = AIAgent()
    proxy = HumanProxy()
    start_time = time.time()

    for entry in samples:
        sample_start = time.time()
        idx = len(results) + 1
        logger.info(
            f"\n[{idx}/{total}]  {entry['id']}  db={entry['db_id']}  "
            f"Q: {entry['question'][:90]}"
        )

        try:
            result = run_hitl_loop(entry, agent, proxy)
        except KeyboardInterrupt:
            logger.info("Interrupted by user — checkpoint saved, run with --resume to continue")
            save_checkpoint(args.output, results, total, config)
            sys.exit(0)
        except Exception as exc:
            logger.error(f"Error on {entry['id']}: {exc}", exc_info=True)
            result = {
                "id":         entry["id"],
                "question":   entry["question"],
                "db_id":      entry.get("db_id", ""),
                "gold_sql":   entry.get("gold_sql", ""),
                "final_sql":  "",
                "passed":     False,
                "iterations": 0,
                "error":      str(exc),
            }

        sample_elapsed = round(time.time() - sample_start, 2)
        result["elapsed_seconds"] = sample_elapsed

        results.append(result)
        print_progress(len(results), total, results, start_time, sample_elapsed)

        # Checkpoint after every sample so nothing is lost
        save_checkpoint(args.output, results, total, config)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - start_time
    finalize(args.output, results, total_elapsed, config)


if __name__ == "__main__":
    main()
