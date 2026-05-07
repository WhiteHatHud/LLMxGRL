#!/usr/bin/env python3
"""
graph_hitl_pipeline.py
======================
Full Graph-Guided HITL ablation pipeline.

Runs four conditions on Spider text-to-SQL samples:

  A: Baseline HITL          — cached results from hitl_results.json
  B: +Graph Feedback        — FK path injected into proxy feedback
  C: +GRL Router            — routes complex queries to more iterations
  D: Full Graph-Guided HITL — router + error classifier + graph feedback

Conditions B/C/D are run only on the samples that FAILED in Condition A
(the 52 failing samples). Condition A accuracy is read from cache.

Usage
-----
python graph_hitl_framework/graph_hitl_pipeline.py \\
    --answer_key   answer_key.json \\
    --baseline     hitl_results.json \\
    --embeddings   graph_hitl_framework/cache/schema_embeddings.pkl \\
    --router_model graph_hitl_framework/cache/grl_router_model.pkl \\
    --tables       StructGPT-Ollama/data/spider/spider_data/tables.json \\
    --db_dir       StructGPT-Ollama/data/spider/spider_data/database \\
    --output       graph_hitl_framework/ablation_results.json

Resume a partial run
---------------------
python graph_hitl_framework/graph_hitl_pipeline.py ... --resume
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "StructGPT-Ollama"))
sys.path.insert(0, str(ROOT / "graph_hitl_framework"))

from schema_graph import SchemaGraph, clean_sql_prediction

# Reusable HITL helpers from the main testbed
from hitl_sql_testbed import (
    AIAgent,
    HumanProxy,
    _call_lmstudio,
    _parse_agent_json,
    _extract_sql,
    execute_sql,
    results_match,
    MAX_ITERATIONS,
    AI_AGENT_MODEL,
    HUMAN_PROXY_MODEL,
    AI_AGENT_SYSTEM,
    HUMAN_PROXY_SYSTEM_TEMPLATE,
)

from grl_error_classifier import GRLErrorClassifier
from graph_oracle_feedback import GraphOracleFeedback

# Optional router — loaded only if model file exists
_router_available = False
try:
    from grl_router import GRLRouter
    _router_available = True
except ImportError:
    pass


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────────────────

# Condition A accuracy comes from cache — 198/250 passed → 79.2%
# (or whatever is stored in the cached hitl_results.json summary)

CONDITIONS = ["B", "C", "D"]

CONDITION_LABELS = {
    "A": "Baseline HITL (cached)",
    "B": "+Graph Feedback only",
    "C": "+GRL Router only",
    "D": "Full Graph-Guided HITL",
}

# When the router says "complex", give the agent extra iterations
COMPLEX_MAX_ITERATIONS = 5   # 2 extra on top of the standard 3


# ─── Schema serialisation (compact StructGPT format) ─────────────────────────

def _compact_schema(db_id: str, db_dir: str, sg: SchemaGraph) -> str:
    """
    Return `# Table(col1,col2,...);` lines + FK strings for db_id.
    Falls back to raw SQLite schema if SchemaGraph can't help.
    """
    entry = sg._db_index.get(db_id)
    if entry is None:
        return ""

    lines = []
    col_names  = entry.get("column_names_original", [])
    table_names = entry.get("table_names_original", [])

    # Group columns by table index
    from collections import defaultdict
    table_cols: dict[int, list[str]] = defaultdict(list)
    for (tbl_idx, col_name) in col_names:
        if tbl_idx >= 0:
            table_cols[tbl_idx].append(col_name)

    for tbl_idx, tbl_name in enumerate(table_names):
        cols = ", ".join(table_cols.get(tbl_idx, []))
        lines.append(f"# {tbl_name}({cols});")

    # FK strings
    fk_entries = entry.get("foreign_keys", [])
    pk_cols     = [c for (_, c) in col_names]
    for (src_idx, dst_idx) in fk_entries:
        if src_idx < len(col_names) and dst_idx < len(col_names):
            src_tbl_idx, src_col = col_names[src_idx]
            dst_tbl_idx, dst_col = col_names[dst_idx]
            if 0 <= src_tbl_idx < len(table_names) and \
               0 <= dst_tbl_idx < len(table_names):
                src_tbl = table_names[src_tbl_idx]
                dst_tbl = table_names[dst_tbl_idx]
                lines.append(
                    f"-- FK: {src_tbl}.{src_col} -> {dst_tbl}.{dst_col}"
                )

    return "\n".join(lines)


# ─── Graph-Aware Proxy ───────────────────────────────────────────────────────

class GraphAwareProxy:
    """
    Wraps HumanProxy: calls the base LLM proxy then optionally appends
    the FK traversal path for structural errors.
    """

    def __init__(self, base_proxy: HumanProxy,
                 graph_feedback: GraphOracleFeedback,
                 use_graph: bool = True):
        self._proxy    = base_proxy
        self._gof      = graph_feedback
        self.use_graph = use_graph

    def respond(self, ai_message: str, answer_key_entry: dict,
                pred_sql: str = "", gold_sql: str = "",
                db_id: str = "") -> tuple[str, dict | None]:
        """
        Returns (feedback_string, classification_dict_or_None).
        """
        base_feedback = self._proxy.respond(ai_message, answer_key_entry)

        if self.use_graph and pred_sql and gold_sql and db_id:
            augmented, clf = self._gof.classify_and_generate(
                proxy_text=base_feedback,
                pred_sql=pred_sql,
                gold_sql=gold_sql,
                db_id=db_id,
                use_graph=True,
            )
            return augmented, clf

        return base_feedback, None


# ─── Modified HITL Loop (Graph-Aware) ────────────────────────────────────────

def run_graph_hitl_loop(
    entry:         dict,
    agent:         AIAgent,
    proxy:         HumanProxy,
    graph_proxy:   GraphAwareProxy | None,
    router:        Any,           # GRLRouter | None
    use_graph:     bool,
    use_router:    bool,
    max_iter_override: int | None = None,
) -> dict:
    """
    Extended Create-Verify-Refine loop with optional graph augmentation.

    Parameters
    ----------
    entry            : answer key entry dict
    agent            : AIAgent instance (reset on entry)
    proxy            : plain HumanProxy (used when use_graph=False)
    graph_proxy      : GraphAwareProxy (used when use_graph=True)
    router           : GRLRouter or None
    use_graph        : inject FK paths into proxy feedback (Condition B/D)
    use_router       : use GRL Router to set iteration budget (Condition C/D)
    max_iter_override: force a specific iteration cap (ignores router)

    Returns
    -------
    dict with keys: id, question, db_id, gold_sql, final_sql, passed,
                    iterations, route, turns, error_classifications
    """
    question     = entry["question"]
    db_id        = entry["db_id"]
    gold_sql     = entry["gold_sql"]
    gold_results = entry["gold_results"]
    db_path      = entry.get("db_path", "")
    schema_text  = entry.get("schema_compact", entry.get("schema", ""))

    agent.reset()

    # ── Route decision ────────────────────────────────────────────────────────
    route_info = {"route": "medium", "predicted_joins": 2, "confidence": 0.5}
    if use_router and router is not None:
        try:
            route_info = router.route(question=question, db_id=db_id)
        except Exception as e:
            logger.warning(f"  Router error: {e} — defaulting to 'medium'")

    route = route_info["route"]

    # Determine iteration budget
    if max_iter_override is not None:
        max_iter = max_iter_override
    elif use_router:
        if route == "simple":
            max_iter = 1
        elif route == "complex":
            max_iter = COMPLEX_MAX_ITERATIONS
        else:
            max_iter = MAX_ITERATIONS
    else:
        max_iter = MAX_ITERATIONS

    logger.info(
        f"  Q: {question[:80]}"
        f"  route={route}  max_iter={max_iter}  graph={use_graph}"
    )

    current_message = (
        f"Database: {db_id}\n\n"
        f"Schema:\n{schema_text}\n\n"
        f"User question: {question}"
    )

    final_sql             = ""
    passed                = False
    iterations            = 0
    turns: list[dict]     = []
    error_classifications = []

    for iteration in range(max_iter):
        iterations += 1

        # ── AI Agent ─────────────────────────────────────────────────────────
        for _attempt in range(3):
            try:
                agent_resp = agent.send(current_message)
                break
            except Exception as e:
                logger.warning(f"  LM Studio error (attempt {_attempt+1}/3): {e}")
                if _attempt == 2:
                    raise
                time.sleep(10)
        turn: dict = {"iteration": iteration + 1, "agent_response": agent_resp,
                      "route": route}

        # ── Clarification branch ─────────────────────────────────────────────
        if agent_resp.get("clarification_needed"):
            clarification_q = (agent_resp.get("question") or
                               agent_resp.get("thought", "Please clarify.")).strip()
            oracle_ctx = {
                "question":     question,
                "gold_sql":     gold_sql,
                "gold_results": gold_results[:5],
            }
            proxy_reply = proxy.respond(clarification_q, oracle_ctx)
            turn["clarification_question"] = clarification_q
            turn["proxy_reply"]            = proxy_reply
            turns.append(turn)
            current_message = proxy_reply
            continue

        # ── SQL branch ───────────────────────────────────────────────────────
        predicted_sql = _extract_sql(agent_resp.get("sql", "").strip())
        if not predicted_sql:
            turn["error"] = "empty_sql"
            turns.append(turn)
            break

        final_sql = predicted_sql
        turn["predicted_sql"] = predicted_sql

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
                ai_msg = (
                    f"Your SQL raised an execution error: {exec_err}\n"
                    f"Analyse and fix it."
                )
                if use_graph and graph_proxy is not None:
                    proxy_feedback, clf_result = graph_proxy.respond(
                        ai_msg, oracle_ctx,
                        pred_sql=predicted_sql, gold_sql=gold_sql, db_id=db_id,
                    )
                    if clf_result:
                        turn["error_classification"] = clf_result
                        error_classifications.append(clf_result)
                else:
                    proxy_feedback = proxy.respond(ai_msg, oracle_ctx)

            elif results_match(pred_results, gold_results):
                logger.info(f"  [iter {iteration+1}] PASS ✓")
                turn["result"]       = "PASS"
                turn["pred_results"] = pred_results[:5]
                turns.append(turn)
                passed = True
                break

            else:
                turn["result"]       = "FAIL"
                turn["pred_results"] = pred_results[:5]
                oracle_ctx = {
                    "question":          question,
                    "gold_sql":          gold_sql,
                    "gold_results":      gold_results[:5],
                    "predicted_results": pred_results[:5],
                }
                ai_msg = (
                    f"I ran your SQL and got:\n{pred_results[:5]}\n"
                    f"This does not match the expected output."
                )
                if use_graph and graph_proxy is not None:
                    proxy_feedback, clf_result = graph_proxy.respond(
                        ai_msg, oracle_ctx,
                        pred_sql=predicted_sql, gold_sql=gold_sql, db_id=db_id,
                    )
                    if clf_result:
                        turn["error_classification"] = clf_result
                        error_classifications.append(clf_result)
                else:
                    proxy_feedback = proxy.respond(ai_msg, oracle_ctx)

        else:
            # No DB available
            if predicted_sql.strip().lower() == gold_sql.strip().lower():
                passed = True
                turn["result"] = "PASS (string match)"
                turns.append(turn)
                break
            oracle_ctx = {"question": question, "gold_sql": gold_sql}
            if use_graph and graph_proxy is not None:
                proxy_feedback, clf_result = graph_proxy.respond(
                    f"Your SQL is:\n{predicted_sql}", oracle_ctx,
                    pred_sql=predicted_sql, gold_sql=gold_sql, db_id=db_id,
                )
                if clf_result:
                    turn["error_classification"] = clf_result
                    error_classifications.append(clf_result)
            else:
                proxy_feedback = proxy.respond(
                    f"Your SQL is:\n{predicted_sql}", oracle_ctx
                )
            turn["result"] = "FAIL (string match)"

        turn["proxy_feedback"] = proxy_feedback
        turns.append(turn)

        if iteration < max_iter - 1:
            current_message = proxy_feedback

    return {
        "id":                   entry["id"],
        "question":             question,
        "db_id":                db_id,
        "gold_sql":             gold_sql,
        "final_sql":            final_sql,
        "passed":               passed,
        "iterations":           iterations,
        "route":                route,
        "route_confidence":     route_info.get("confidence", 0.0),
        "predicted_joins":      route_info.get("predicted_joins", 0),
        "turns":                turns,
        "error_classifications": error_classifications,
    }


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def _load_checkpoint(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_checkpoint(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─── Summary helpers ──────────────────────────────────────────────────────────

def _summarise_condition(results: list[dict], elapsed_s: float,
                         condition: str, total_samples: int) -> dict:
    """
    Build a summary dict for one ablation condition.
    total_samples = number run under this condition (52 failing samples).
    """
    passed  = sum(1 for r in results if r.get("passed"))
    total   = len(results)
    avg_it  = sum(r.get("iterations", 0) for r in results) / max(total, 1)

    fixed_at: dict[int, int] = {}
    for r in results:
        if r.get("passed"):
            it = r.get("iterations", 1)
            fixed_at[it] = fixed_at.get(it, 0) + 1

    route_dist: dict[str, int] = {}
    for r in results:
        rt = r.get("route", "medium")
        route_dist[rt] = route_dist.get(rt, 0) + 1

    # Structural vs surface breakdown
    structural_fixed = 0
    surface_fixed    = 0
    for r in results:
        if not r.get("passed"):
            continue
        for clf in r.get("error_classifications", []):
            if clf.get("error_type") == "structural":
                structural_fixed += 1
            elif clf.get("error_type") == "surface":
                surface_fixed += 1

    return {
        "condition":          condition,
        "label":              CONDITION_LABELS.get(condition, condition),
        "samples_run":        total,
        "passed":             passed,
        "failed":             total - passed,
        "recovery_rate":      round(passed / max(total, 1), 4),
        "avg_iterations":     round(avg_it, 2),
        "fixed_at_iteration": fixed_at,
        "route_distribution": route_dist,
        "structural_fixed":   structural_fixed,
        "surface_fixed":      surface_fixed,
        "elapsed_seconds":    round(elapsed_s),
        "elapsed_human":      str(timedelta(seconds=int(elapsed_s))),
    }


def _print_ablation_table(baseline_summary: dict,
                           condition_summaries: list[dict],
                           total_250: int) -> None:
    """Print a formatted ablation comparison table."""
    baseline_passed = baseline_summary.get("passed", 0)
    baseline_total  = baseline_summary.get("total", total_250)
    baseline_acc    = baseline_passed / max(baseline_total, 1)

    print(f"\n{'='*72}")
    print(f"  ABLATION RESULTS — Graph-Guided HITL (Spider)")
    print(f"{'='*72}")
    print(f"  {'Condition':<35} {'Accuracy':>10}  {'Recovery':>10}  {'Δ vs A':>8}")
    print(f"  {'-'*35} {'-'*10}  {'-'*10}  {'-'*8}")

    # Condition A
    print(f"  {'A: ' + CONDITION_LABELS['A']:<35} "
          f"{100*baseline_acc:>9.1f}%  "
          f"{'—':>10}  {'—':>8}")

    for s in condition_summaries:
        cond = s["condition"]
        # Overall accuracy = (baseline_passed + newly recovered) / total_250
        recovered       = s["passed"]
        overall_passed  = baseline_passed + recovered
        overall_acc     = overall_passed / max(total_250, 1)
        delta_pp        = 100 * (overall_acc - baseline_acc)
        recovery_rate   = 100 * s["recovery_rate"]
        delta_sign      = "+" if delta_pp >= 0 else ""

        print(
            f"  {cond + ': ' + CONDITION_LABELS[cond]:<35} "
            f"{overall_acc*100:>9.1f}%  "
            f"{recovery_rate:>9.1f}%  "
            f"{delta_sign}{delta_pp:>6.1f}pp"
        )

    print(f"{'='*72}\n")


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_ablation(
    answer_key_path:  str,
    baseline_path:    str,
    embeddings_path:  str,
    router_model_path: str,
    tables_path:      str,
    db_dir:           str,
    output_path:      str,
    resume:           bool = False,
) -> None:

    # ── Load answer key ───────────────────────────────────────────────────────
    with open(answer_key_path) as f:
        answer_key: list[dict] = json.load(f)

    # Map id → entry for fast lookup
    key_by_id = {e["id"]: e for e in answer_key}

    # Add db_path and compact schema to every entry
    sg = SchemaGraph(tables_path)
    for entry in answer_key:
        db_id   = entry["db_id"]
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        entry["db_path"]        = db_path
        entry["schema_compact"] = _compact_schema(db_id, db_dir, sg)

    # ── Load cached baseline (Condition A) ───────────────────────────────────
    with open(baseline_path) as f:
        baseline_data = json.load(f)

    baseline_results  = baseline_data.get("results", [])
    baseline_summary  = baseline_data.get("summary", {})
    total_250         = baseline_summary.get("total", len(baseline_results))
    baseline_passed   = baseline_summary.get("passed",
                            sum(1 for r in baseline_results if r.get("passed")))

    logger.info(
        f"Baseline (Condition A): {baseline_passed}/{total_250} passed "
        f"({100*baseline_passed/max(total_250,1):.1f}%)"
    )

    # ── Identify the 52 failing samples ──────────────────────────────────────
    failing_ids = {r["id"] for r in baseline_results if not r.get("passed")}
    failing_entries = []
    for entry in answer_key:
        if entry["id"] in failing_ids:
            failing_entries.append(entry)
        elif entry["id"] not in {r["id"] for r in baseline_results}:
            # ID not in baseline at all — include it
            failing_entries.append(entry)

    logger.info(f"Failing samples to rerun: {len(failing_entries)}")

    # ── Load optional components ──────────────────────────────────────────────
    router = None
    if _router_available and router_model_path and \
            Path(router_model_path).exists():
        try:
            router = GRLRouter.load(
                router_model_path,
                embeddings_path=embeddings_path if
                    Path(embeddings_path).exists() else None,
                use_encoder=True,
            )
            logger.info("GRL Router loaded.")
        except Exception as e:
            logger.warning(f"Could not load router: {e}")

    clf     = GRLErrorClassifier(sg)
    gof     = GraphOracleFeedback(sg, clf)
    agent   = AIAgent()
    proxy   = HumanProxy()

    graph_proxy = GraphAwareProxy(proxy, gof, use_graph=True)

    # ── Load checkpoint if resuming ───────────────────────────────────────────
    checkpoint = _load_checkpoint(output_path) if resume else {}
    condition_results: dict[str, list[dict]] = {
        cond: checkpoint.get("condition_results", {}).get(cond, [])
        for cond in CONDITIONS
    }
    condition_elapsed: dict[str, float] = {cond: 0.0 for cond in CONDITIONS}
    completed_ids: dict[str, set] = {
        cond: {r["id"] for r in condition_results[cond]}
        for cond in CONDITIONS
    }

    # ── Configuration flags per condition ────────────────────────────────────
    #  Condition B: graph=True,  router=False
    #  Condition C: graph=False, router=True
    #  Condition D: graph=True,  router=True
    cond_config = {
        "B": {"use_graph": True,  "use_router": False},
        "C": {"use_graph": False, "use_router": True},
        "D": {"use_graph": True,  "use_router": True},
    }

    # ── Run conditions ────────────────────────────────────────────────────────
    for cond in CONDITIONS:
        cfg        = cond_config[cond]
        use_graph  = cfg["use_graph"]
        use_router = cfg["use_router"]

        todo = [e for e in failing_entries
                if e["id"] not in completed_ids[cond]]

        if not todo:
            logger.info(f"Condition {cond}: all {len(failing_entries)} already done.")
            continue

        logger.info(
            f"\n{'─'*60}\n"
            f"  Condition {cond}: {CONDITION_LABELS[cond]}\n"
            f"  graph={use_graph}  router={use_router}\n"
            f"  samples={len(todo)} (of {len(failing_entries)} failing)\n"
            f"{'─'*60}"
        )

        start_t = time.time()

        for i, entry in enumerate(todo):
            result = run_graph_hitl_loop(
                entry=entry,
                agent=agent,
                proxy=proxy,
                graph_proxy=graph_proxy if use_graph else None,
                router=router if use_router else None,
                use_graph=use_graph,
                use_router=use_router,
            )
            condition_results[cond].append(result)

            done   = len(condition_results[cond])
            passed = sum(1 for r in condition_results[cond] if r.get("passed"))
            print(
                f"  [{done:>3}/{len(failing_entries)}]  "
                f"cond={cond}  pass={passed}  "
                f"this={'✓' if result['passed'] else '✗'}  "
                f"id={entry['id'][:20]}",
                flush=True,
            )

            # Save after every sample
            _save_checkpoint(output_path, {
                "config": {
                    "answer_key":    answer_key_path,
                    "baseline":      baseline_path,
                    "tables":        tables_path,
                    "db_dir":        db_dir,
                    "conditions":    CONDITIONS,
                    "failing_count": len(failing_entries),
                },
                "baseline_summary":  baseline_summary,
                "condition_results": condition_results,
                "condition_summaries": {},  # filled at end
            })

        elapsed = time.time() - start_t
        condition_elapsed[cond] += elapsed
        s = _summarise_condition(
            condition_results[cond], condition_elapsed[cond], cond, len(failing_entries)
        )
        logger.info(
            f"  Condition {cond} done: "
            f"{s['passed']}/{s['samples_run']} recovered  "
            f"({100*s['recovery_rate']:.1f}%)  "
            f"time={s['elapsed_human']}"
        )

    # ── Final output ──────────────────────────────────────────────────────────
    total_elapsed = sum(condition_elapsed.values())

    summaries = {
        cond: _summarise_condition(
            condition_results[cond],
            condition_elapsed[cond],
            cond,
            len(failing_entries),
        )
        for cond in CONDITIONS
    }

    final = {
        "config": {
            "answer_key":    answer_key_path,
            "baseline":      baseline_path,
            "tables":        tables_path,
            "db_dir":        db_dir,
            "conditions":    CONDITIONS,
            "total_250":     total_250,
            "failing_count": len(failing_entries),
        },
        "baseline_summary":    baseline_summary,
        "condition_summaries": summaries,
        "condition_results":   condition_results,
    }

    _save_checkpoint(output_path, final)
    logger.info(f"Ablation results saved → {output_path}")

    _print_ablation_table(
        baseline_summary,
        list(summaries.values()),
        total_250,
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Graph-Guided HITL Ablation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--answer_key",    required=True,
                   help="answer_key.json built from hitl_sql_testbed.py")
    p.add_argument("--baseline",      required=True,
                   help="hitl_results.json — cached Condition A results")
    p.add_argument("--embeddings",    required=True,
                   help="graph_hitl_framework/cache/schema_embeddings.pkl")
    p.add_argument("--router_model",  required=True,
                   help="graph_hitl_framework/cache/grl_router_model.pkl")
    p.add_argument("--tables",        required=True,
                   help="StructGPT-Ollama/data/spider/spider_data/tables.json")
    p.add_argument("--db_dir",        required=True,
                   help="StructGPT-Ollama/data/spider/spider_data/database")
    p.add_argument("--output",
                   default="graph_hitl_framework/ablation_results.json")
    p.add_argument("--resume", action="store_true",
                   help="Resume from a partial checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation(
        answer_key_path=args.answer_key,
        baseline_path=args.baseline,
        embeddings_path=args.embeddings,
        router_model_path=args.router_model,
        tables_path=args.tables,
        db_dir=args.db_dir,
        output_path=args.output,
        resume=args.resume,
    )
