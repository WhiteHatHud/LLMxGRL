#!/usr/bin/env bash
# =============================================================================
# run_ablation_1034.sh — Full 1034-sample Graph-Guided HITL Ablation
# =============================================================================
#
# Step 1: Retrain GRL Router on 1034-sample answer key (800 train / 232 test)
# Step 2: Run conditions B / C / D on the 113 failing samples
#
# Usage
# -----
#   bash run_ablation_1034.sh              # full run from scratch
#   bash run_ablation_1034.sh --resume     # resume step 2 after interruption
#   bash run_ablation_1034.sh --skip_router  # skip step 1 if router already built
#
# Time estimate
# -------------
#   Step 1 (router training) : ~2–5 minutes
#   Step 2 (ablation B/C/D)  : ~113 samples × 3 conditions × ~40s ≈ 3–4 hours
#
# Output files
# ------------
#   graph_hitl_framework/cache/grl_router_model_1034.pkl  — retrained router
#   graph_hitl_framework/ablation_results_1034.json        — final results
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Argument parsing ──────────────────────────────────────────────────────────

SKIP_ROUTER=false
RESUME_FLAG=""

for arg in "$@"; do
    case $arg in
        --skip_router) SKIP_ROUTER=true ;;
        --resume)      RESUME_FLAG="--resume" ;;
        *)             echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────

ANSWER_KEY="answer_key_1034.json"
BASELINE="hitl_results_1034.json"
EMBEDDINGS="graph_hitl_framework/cache/schema_embeddings.pkl"
ROUTER_MODEL="graph_hitl_framework/cache/grl_router_model_1034.pkl"
TABLES="StructGPT-Ollama/data/spider/spider_data/tables.json"
DB_DIR="StructGPT-Ollama/data/spider/spider_data/database"
ABLATION_OUT="graph_hitl_framework/ablation_results_1034.json"
PYTHON="venv/bin/python"

# ── Sanity checks ─────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Graph-Guided HITL Ablation — 1034-sample Spider dev"
echo "============================================================"

for f in "$ANSWER_KEY" "$BASELINE" "$EMBEDDINGS" "$TABLES"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required file not found: $f"
        echo "       Run the following first if missing:"
        echo "         answer key  → python hitl_full_1034.py --build_key --key_only"
        echo "         baseline    → bash run_full_1034.sh"
        echo "         embeddings  → bash run_embeddings.sh"
        exit 1
    fi
done

if [[ ! -d "$DB_DIR" ]]; then
    echo "ERROR: Spider database directory not found: $DB_DIR"
    exit 1
fi

# Check LM Studio
if ! curl -s --max-time 3 http://localhost:1234/v1/models > /dev/null 2>&1; then
    echo "WARNING: LM Studio not reachable at http://localhost:1234"
    echo "         Start LM Studio and load qwen/qwen2.5-coder-14b before step 2."
    echo "         (Step 1 router training does not need LM Studio.)"
    echo ""
fi

# ── Step 1: Train GRL Router ──────────────────────────────────────────────────

if [[ "$SKIP_ROUTER" == true ]]; then
    echo ""
    echo "  [Step 1] Skipping router training (--skip_router)"
    if [[ ! -f "$ROUTER_MODEL" ]]; then
        echo "  WARNING: Router model not found at $ROUTER_MODEL"
        echo "           Remove --skip_router or run step 1 manually."
        exit 1
    fi
else
    echo ""
    echo "  [Step 1] Training GRL Router on 1034-sample answer key"
    echo "           train_size=800  |  held-out=~232"
    echo "           Output: $ROUTER_MODEL"
    echo ""

    STEP1_START=$(date +%s)

    $PYTHON graph_hitl_framework/grl_router.py \
        --answer_key "$ANSWER_KEY" \
        --embeddings "$EMBEDDINGS" \
        --tables     "$TABLES" \
        --output     "$ROUTER_MODEL" \
        --train_size 800

    STEP1_END=$(date +%s)
    STEP1_ELAPSED=$(( STEP1_END - STEP1_START ))
    echo ""
    echo "  [Step 1] Done in ${STEP1_ELAPSED}s → $ROUTER_MODEL"
fi

# ── Step 2: Run Ablation (B / C / D) ─────────────────────────────────────────

echo ""
echo "  [Step 2] Running ablation conditions B / C / D"
echo "           Baseline  : $BASELINE"
echo "           Router    : $ROUTER_MODEL"
echo "           Output    : $ABLATION_OUT"
if [[ -n "$RESUME_FLAG" ]]; then
    echo "           Mode      : RESUME"
else
    echo "           Mode      : fresh run"
fi
echo ""
echo "  Est. time: ~3–4 hours (113 failing samples × 3 conditions)"
echo "  Safe to Ctrl-C — rerun with --resume to continue"
echo ""

STEP2_START=$(date +%s)

$PYTHON graph_hitl_framework/graph_hitl_pipeline.py \
    --answer_key  "$ANSWER_KEY" \
    --baseline    "$BASELINE" \
    --embeddings  "$EMBEDDINGS" \
    --router_model "$ROUTER_MODEL" \
    --tables      "$TABLES" \
    --db_dir      "$DB_DIR" \
    --output      "$ABLATION_OUT" \
    $RESUME_FLAG

STEP2_END=$(date +%s)
STEP2_ELAPSED=$(( STEP2_END - STEP2_START ))
STEP2_HUMAN=$(date -ud "@$STEP2_ELAPSED" +'%H:%M:%S' 2>/dev/null || \
              python3 -c "import datetime; print(str(datetime.timedelta(seconds=$STEP2_ELAPSED)))")

echo ""
echo "============================================================"
echo "  Ablation complete"
echo "  Step 1 (router)  : ${STEP1_ELAPSED:-skipped}s"
echo "  Step 2 (ablation): $STEP2_HUMAN"
echo "  Results          : $ABLATION_OUT"
echo "============================================================"
echo ""
