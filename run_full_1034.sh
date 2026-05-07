#!/usr/bin/env bash
# =============================================================================
# run_full_1034.sh — Full 1034-sample HITL baseline evaluation
# =============================================================================
#
# Model  : Qwen 2.5 Coder 14B  (AI Agent + Human Proxy, via LM Studio)
# Dataset: Spider dev set — all 1034 samples
# Output : hitl_results_1034.json  (checkpointed after every sample)
# Log    : hitl_full_1034.log
#
# Time estimate
# -------------
#   ~35 s / sample × 1034 = ~10 hours
#   Run overnight. Safe to interrupt (Ctrl-C) and resume.
#
# Usage
# -----
#   bash run_full_1034.sh              # fresh start (builds key if missing)
#   bash run_full_1034.sh --resume     # continue after interruption
#   bash run_full_1034.sh --key_only   # build answer key only, no HITL yet
#
# Requirements
# ------------
#   LM Studio running at http://localhost:1234
#   Model loaded: qwen/qwen2.5-coder-14b
#   Spider dev data at: StructGPT-Ollama/data/spider/spider_data/
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Sanity checks ─────────────────────────────────────────────────────────────

SPIDER_DEV="StructGPT-Ollama/data/spider/spider_data/dev.json"
DB_ROOT="StructGPT-Ollama/data/spider/spider_data/database"

if [[ ! -f "$SPIDER_DEV" ]]; then
    echo "ERROR: Spider dev.json not found at $SPIDER_DEV"
    echo "       Download Spider dataset first."
    exit 1
fi

if [[ ! -d "$DB_ROOT" ]]; then
    echo "ERROR: Spider database directory not found at $DB_ROOT"
    exit 1
fi

# Check LM Studio is reachable
if ! curl -s --max-time 3 http://localhost:1234/v1/models > /dev/null 2>&1; then
    echo "WARNING: LM Studio not reachable at http://localhost:1234"
    echo "         Start LM Studio and load qwen/qwen2.5-coder-14b before continuing."
    echo "         Proceeding anyway in case it starts up shortly..."
    echo ""
fi

# ── Run ───────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Full 1034-sample HITL Evaluation — Qwen 2.5 Coder 14B"
echo "  Press Ctrl-C to pause. Resume with: bash run_full_1034.sh --resume"
echo "============================================================"
echo ""

python hitl_full_1034.py --build_key "$@"

echo ""
echo "Done. Results in: hitl_results_1034.json"
echo "Log in:           hitl_full_1034.log"
