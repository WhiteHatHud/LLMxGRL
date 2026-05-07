#!/bin/bash
# run_graph_augmented.sh
# Graph-Augmented Prompt Injection — Priority 1 Experiment
#
# Targets all failing samples with >= JOIN_THRESHOLD joins in their gold SQL,
# injects the FK path chain into the prompt, and measures recovery rate.
#
# Usage:
#   bash scripts/run_graph_augmented.sh [join_threshold] [model]
#
# Examples:
#   bash scripts/run_graph_augmented.sh           # threshold=2, default model
#   bash scripts/run_graph_augmented.sh 3         # only 3+ join failures
#   bash scripts/run_graph_augmented.sh 2 qwen2.5-coder:7b

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="/home/hud/Projects/LLMxGRL/venv/bin/python3"

JOIN_THRESHOLD="${1:-2}"
MODEL="${2:-qwen2.5-coder:14b}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

INPUT="$REPO_DIR/outputs/spider/output_sample_250_qwen.jsonl"
OUTPUT="$REPO_DIR/outputs/spider/output_graph_augmented_t${JOIN_THRESHOLD}_${TIMESTAMP}.jsonl"
DB_DIR="$REPO_DIR/data/spider/spider_data/database"
TABLES="$REPO_DIR/data/spider/spider_data/tables.json"

echo "========================================================"
echo "  Graph-Augmented Re-run"
echo "========================================================"
echo "  Input          : $INPUT"
echo "  Output         : $OUTPUT"
echo "  Join threshold : >= $JOIN_THRESHOLD joins"
echo "  Model          : $MODEL"
echo "  Timestamp      : $TIMESTAMP"
echo "========================================================"
echo ""

# Sanity checks
if [ ! -f "$INPUT" ]; then
    echo "ERROR: Input file not found: $INPUT"
    exit 1
fi

if [ ! -d "$DB_DIR" ]; then
    echo "ERROR: Database directory not found: $DB_DIR"
    exit 1
fi

if [ ! -f "$TABLES" ]; then
    echo "ERROR: tables.json not found: $TABLES"
    exit 1
fi

# Check Ollama is running
if ! curl -sf "http://localhost:11434/api/tags" > /dev/null 2>&1; then
    echo "ERROR: Ollama is not running at http://localhost:11434"
    echo "  Start it with: ollama serve"
    exit 1
fi

echo "Ollama is running. Starting experiment …"
echo ""

cd "$REPO_DIR"
"$VENV_PYTHON" graph_augmented_rerun.py \
    --input    "$INPUT" \
    --output   "$OUTPUT" \
    --db_dir   "$DB_DIR" \
    --tables   "$TABLES" \
    --join_threshold "$JOIN_THRESHOLD" \
    --model    "$MODEL" \
    --max_tokens 400

echo ""
echo "Done. Results saved to: $OUTPUT"
