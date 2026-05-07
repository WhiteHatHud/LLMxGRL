# Graph-Guided HITL Framework

Graph Representation Learning-enhanced Human-in-the-Loop pipeline for Spider text-to-SQL.
Integrates Node2Vec schema embeddings directly into the HITL feedback loop.

---

## What This Adds on Top of Baseline HITL

| Component | What it does |
|-----------|-------------|
| **Node2Vec embeddings** | Learns structural table representations from FK graphs |
| **GRL Router** | Predicts join complexity, routes queries to right strategy |
| **GRL Error Classifier** | Detects surface vs structural errors after each failure |
| **Graph-Augmented Oracle** | Injects FK traversal path into proxy feedback |
| **Full Pipeline** | Wires all components, runs 4-condition ablation |

---

## Dependencies

```bash
pip install node2vec sentence-transformers scikit-learn
```

| Package | Version | Purpose |
|---------|---------|---------|
| `node2vec` | ≥0.4 | Random walk graph embeddings (Node2Vec algorithm) |
| `sentence-transformers` | ≥2.2 | Query encoding — downloads `all-MiniLM-L6-v2` (~80MB on first run) |
| `scikit-learn` | ≥1.0 | Logistic regression for GRL Router classifier |
| `networkx` | ≥2.8 | Graph construction (already in project) |
| `numpy` | ≥1.21 | Embedding operations (already in project) |

> **Note:** `sentence-transformers` will download `all-MiniLM-L6-v2` from
> HuggingFace (~80MB) on first run. Requires internet connection once,
> then cached locally at `~/.cache/huggingface/`.

---

## Existing Tools Reused (No Rewriting)

### From `StructGPT-Ollama/schema_graph.py`
- `SchemaGraph` — builds NetworkX FK graphs, computes shortest paths
- `extract_tables_from_sql()` — parse table names from SQL
- `count_joins_in_sql()` — join depth detection
- `clean_sql_prediction()` — strip markdown fences from model output

### From `StructGPT-Ollama/graph_augmented_rerun.py`
- `exec_sql()` — execute SQL against SQLite
- `results_match()` — execution-based correctness check
- `serialise_schema()` — schema text + FK text for prompts

### From `hitl_sql_testbed.py`
- `AIAgent` — stateful multi-turn Qwen agent
- `HumanProxy` — oracle proxy with gold-label access
- `_call_lmstudio()` — LM Studio inference
- `_parse_agent_json()` — robust JSON response parsing
- `save_checkpoint()` / `finalize_output()` — checkpointing

---

## File Structure

```
graph_hitl_framework/
├── README.md                    ← This file
├── PLAN.md                      ← Architecture and design decisions
├── node2vec_embeddings.py       ← Step 1: Train and cache schema embeddings
├── grl_router.py                ← Step 2: Train query complexity classifier
├── grl_error_classifier.py      ← Step 3: Surface vs structural error detection
├── graph_oracle_feedback.py     ← Step 4: Graph-path-augmented proxy feedback
├── graph_hitl_pipeline.py       ← Step 5: Full ablation pipeline
└── cache/                       ← Auto-created: cached embeddings + model
    ├── schema_embeddings.pkl    ← Node2Vec + structural features per DB
    └── grl_router_model.pkl     ← Trained logistic regression classifier
```

---

## How to Run (In Order)

### Step 1 — Train Node2Vec Embeddings
```bash
cd /home/hud/Projects/LLMxGRL
python graph_hitl_framework/node2vec_embeddings.py \
    --tables StructGPT-Ollama/data/spider/spider_data/tables.json \
    --output graph_hitl_framework/cache/schema_embeddings.pkl
```
Expected time: ~5 minutes for all Spider databases.

### Step 2 — Train GRL Router
```bash
python graph_hitl_framework/grl_router.py \
    --answer_key answer_key.json \
    --embeddings graph_hitl_framework/cache/schema_embeddings.pkl \
    --output graph_hitl_framework/cache/grl_router_model.pkl \
    --tables StructGPT-Ollama/data/spider/spider_data/tables.json
```
Trains on 200 samples, evaluates on 50. Prints per-class accuracy.

### Step 3 — Run Full Ablation Pipeline
```bash
python graph_hitl_framework/graph_hitl_pipeline.py \
    --answer_key answer_key.json \
    --baseline_results hitl_results.json \
    --embeddings graph_hitl_framework/cache/schema_embeddings.pkl \
    --router_model graph_hitl_framework/cache/grl_router_model.pkl \
    --tables StructGPT-Ollama/data/spider/spider_data/tables.json \
    --db_dir StructGPT-Ollama/data/spider/spider_data/database \
    --output graph_hitl_framework/ablation_results.json
```
Runs Conditions B, C, D on the 52 failing samples from Condition A (cached baseline).
Expected time: ~60 minutes.

---

## Ablation Conditions

| Condition | Router | Error Classifier | Graph Feedback | Expected |
|-----------|--------|-----------------|----------------|----------|
| A: Baseline HITL | ✗ | ✗ | ✗ | 84.8% (cached) |
| B: +Graph Feedback | ✗ | ✗ | ✓ | >A |
| C: +GRL Router | ✓ | ✗ | ✗ | >A |
| D: Full Graph-Guided HITL | ✓ | ✓ | ✓ | >B,C |

---

## Data Paths

| Data | Path |
|------|------|
| Spider tables.json | `StructGPT-Ollama/data/spider/spider_data/tables.json` |
| Spider databases | `StructGPT-Ollama/data/spider/spider_data/database/` |
| Answer key | `answer_key.json` |
| Cached baseline | `hitl_results.json` |
