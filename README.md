# Human-in-the-Loop Structured Reasoning with Schema Graph Augmentation

**Iterative Oracle Feedback for Text-to-SQL and Table QA**

Hudzaifah Bin Muhammad Taufiq · College of Computing and Data Science  
Supervisor: Assoc. Prof. Ke Yiping, Kelly  
Nanyang Technological University · 2026

---

## Overview

This project tackles a critical **representation bottleneck** in LLM reasoning over structured data: flat-text schema representations are topologically inert, causing systematic failures on complex relational tasks. Building on the StructGPT framework, a three-phase approach is implemented using **Qwen 2.5 Coder 14B**, evaluated across Spider, TabFact, and WikiTableQuestions.

The core insight: *listing foreign keys as isolated two-column declarations is like giving a navigator a list of street names rather than a map.* When the model encounters 3+ join queries, natural language feedback alone fails — the bottleneck is representational, not capability-based.

### Key Results

| Metric | Value |
|--------|-------|
| Multi-hop recovery (0% → **~37%**) | via Graph-Path Injection |
| Accuracy lift on TabFact & WTQ | **+25.6 pp** |
| Spider HITL accuracy (250 samples) | **84.8%** |
| Spider full-scale accuracy (1,032 samples) | **89.1%** |
| Iteration cliff | Recovery collapses to **0.0%** at 3+ joins |

---

## Three-Phase Methodology

```
Phase 1 — Create          Phase 2 — Verify (HITL)       Phase 3 — Refine
─────────────────         ──────────────────────         ──────────────────
Zero-Shot Baseline    →   Simulated Oracle Feedback  →   Schema Graph Augmentation
Model Selection           Feedback-and-Refine Loop       GRL Router + Error Classifier
Execution-Based Eval      Up to 3 iterations             Graph-Path Injection
```

### Phase 1 · Model Selection

Qwen 2.5 Coder 14B was selected over DeepSeek R1 8B:

| | DeepSeek R1 8B | Qwen 2.5 Coder 14B |
|--|--|--|
| Speed | 19.32 sec/sample | 50.43 sec/sample |
| Accuracy | ~1% (extraction failure) | **~30% normalized** |
| Output format | Buried in chain-of-thought | Structured JSON |
| Pipeline compatibility | Poor | Full HITL support |

> Key insight: string-based metrics systematically disadvantage reasoning-heavy models. Execution-based evaluation is essential for fair comparison.

### Phase 2 · Simulated Oracle Feedback Loop

Qwen 2.5 Coder 14B fulfils two roles simultaneously — an AI Agent for SQL generation and a Human Proxy Oracle as automated critic. An optional configuration substitutes Claude 3.5 Sonnet as the critic.

**Feedback-and-Refine Loop:**
1. Agent generates structured JSON with SQL
2. Oracle verifies against answer key; provides NL feedback (missing tables/filters — no final answer revealed)
3. Feedback appended to conversation history; agent revises
4. Loop capped at 3 iterations; terminates early on correct answer

### Phase 3 · Schema Graph Augmentation

| Component | Description |
|-----------|-------------|
| **Graph Construction** | Each schema → directed graph. Tables as nodes, foreign keys as edges. 64-dim Node2Vec embeddings (walks length 10, 50 walks/node). All 138 Spider schemas embedded in ~15 min |
| **GRL Complexity Router** | Multinomial logistic regression with 15-feature vector (graph stats, embedding similarity, linguistic signals). Routes to **Simple** (1 iter), **Medium** (2 iter), or **Complex** (5 iter) |
| **Graph-Topology Error Classifier** | Rule-based, deterministic — no training data required. Classifies errors as structural or surface-level via 3 graph-theoretic rules |
| **Graph-Path Injection** | Replaces vague NL hints with explicit FK traversal paths, e.g., `CAR_MAKERS → MODEL_LIST → CAR_NAMES`. Sourced from the actual graph object — structurally guaranteed correct |

---

## Results

### Spider — The Iteration Cliff

| Join Depth | Samples | Baseline Passed | HITL Recovered | Final Rate |
|------------|---------|-----------------|----------------|------------|
| 0 Joins | 145 | 131 | 0 | 90.3% |
| 1 Join | 78 | 59 | 10 | 88.5% |
| 2 Joins | 12 | 8 | 2 | 83.3% |
| **3+ Joins** | **15** | **0** | **0** | **0.0%** |
| **Total** | **250** | **198** | **12** | **84.0%** |

A hard topological threshold exists at 3+ joins. Despite three iterations of targeted Oracle feedback, the model achieved **exactly 0% recovery** on all 15 multi-hop samples. The bottleneck is representational, not capability-based — models shortcut four-table chains into incorrect two-hop joins even when explicitly told which tables are missing.

### Extended Benchmarking

| Dataset | StructGPT + ChatGPT (zero-shot) | My Work Baseline | **My Work (Qwen 14B + HITL)** | Margin |
|---------|----------------------------------|-----------------|-------------------------------|--------|
| TabFact | 87.1% | 71.2% | **96.8%** | +9.7 pp |
| WikiTableQuestions | 48.4% | 47.6% | **73.2%** | +24.8 pp |
| Spider | 74.8% | 84.0% | **84.8%** | +10.0 pp |

### Phase 3 · Ablation Study (250 samples, recovery on failing cases)

| Config | Description | Recovery Rate |
|--------|-------------|---------------|
| A — Baseline | Standard HITL | 84.8% accuracy |
| B — Graph-Path Feedback | FK paths injected into oracle feedback | **21.1%** |
| C — GRL Router | Complexity routing only | 7.9% |
| D — Full Pipeline (B+C) | Router + classifier + graph feedback | 26.3% |

At 1,032 samples, Config B (graph-path feedback alone) achieved **37.2% recovery** on failing samples, outperforming the full pipeline (Config D: 23.0%). The GRL Router introduces interference effects at scale — a key finding motivating further investigation.

---

## Project Structure

```
LLMxGRL/
├── graph_hitl_framework/          # Core GRL-enhanced HITL system
│   ├── node2vec_embeddings.py     # Train Node2Vec on Spider schema FK graphs
│   ├── grl_router.py              # Predict query complexity, route strategy
│   ├── grl_error_classifier.py    # Classify surface vs structural errors
│   ├── graph_oracle_feedback.py   # FK path-augmented oracle feedback
│   └── graph_hitl_pipeline.py     # Full ablation pipeline (Configs A/B/C/D)
│
├── StructGPT-Ollama/              # Baseline SQL generation & evaluation
│   ├── schema_graph.py            # FK graph construction, path finding
│   ├── graph_augmented_rerun.py   # SQL execution, result matching
│   ├── evaluate_for_spider.py     # Spider evaluation
│   ├── evaluate_for_tabfact.py    # TabFact evaluation
│   └── evaluate_for_webqsp.py     # WikiTableQuestions evaluation
│
├── backend/                       # FastAPI REST backend
│   ├── main.py                    # FastAPI app
│   ├── core/config.py             # Provider/model configuration
│   ├── agents/tools/              # LLM client, GNN adapter, graph utils
│   ├── models/                    # SQLAlchemy ORM (Graph, Node, Edge, Run)
│   ├── routers/                   # API endpoints
│   └── utils/                     # Metrics, charts, timers
│
├── hitl_sql_testbed.py            # Spider text-to-SQL HITL testbed
├── hitl_tabfact_testbed.py        # TabFact fact-verification HITL testbed
├── hitl_wtq_testbed.py            # WikiTableQuestions HITL testbed
├── single_pass_baseline.py        # Single-pass baseline (no HITL)
├── visualize_architecture.py      # Generate pipeline diagrams
├── visualize_results.py           # Plot accuracy vs k-hops, token usage
│
├── answer_key.json                # 250 Spider gold SQL + execution results
├── answer_key_1034.json           # 1,032-sample extended dataset
├── hitl_results.json              # Phase 2 Spider HITL results
│
├── run_full_1034.sh               # Full HITL run on 1,032 Spider samples
├── run_ablation_1034.sh           # Full ablation (A/B/C/D) on 1,032 samples
├── run_embeddings.sh              # Train Node2Vec embeddings
└── run_router.sh                  # Train GRL router
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Requirements:** LM Studio running locally on `localhost:1234` with Qwen 2.5 Coder 14B loaded.

---

## Running Experiments

### Phase 2 · HITL Baseline

```bash
# Spider text-to-SQL (250 samples)
python hitl_sql_testbed.py \
  --from_baseline StructGPT-Ollama/outputs/spider/output_sample_250_qwen.jsonl \
  --limit 250

# TabFact fact verification
python hitl_tabfact_testbed.py --limit 500

# WikiTableQuestions
python hitl_wtq_testbed.py --limit 500
```

### Phase 3 · Graph-Augmented HITL

```bash
# Step 1: Train Node2Vec embeddings on all 138 Spider schemas
bash run_embeddings.sh

# Step 2: Train GRL complexity router (200 labeled samples)
bash run_router.sh

# Step 3: Run full ablation (Configs A/B/C/D) on 1,032 samples
bash run_ablation_1034.sh
```

### Backend API

```bash
cd backend
uvicorn main:app --reload
```

Key endpoints:

| Endpoint | Purpose |
|----------|---------|
| `POST /graphs` | Create a schema graph |
| `POST /graphs/<id>/upload` | Upload nodes/edges/text |
| `POST /pipeline/<run_id>/unigraph` | Build unified graph |
| `POST /pipeline/<run_id>/gnn-adapter` | Compute GNN features |
| `POST /query/<run_id>` | Query with k-hop retrieval |
| `GET /results/<run_id>/export.csv` | Export results |

---

## Configuration

Key settings in `hitl_sql_testbed.py`:

```python
LMSTUDIO_BASE      = "http://localhost:1234/v1"
AI_AGENT_MODEL     = "qwen/qwen2.5-coder-14b"
HUMAN_PROXY_MODEL  = "qwen/qwen2.5-coder-14b"   # or "claude-3-5-sonnet"
HUMAN_PROXY_PROVIDER = "lmstudio"                # or "anthropic"
MAX_ITERATIONS     = 3
```

LLM providers supported: LM Studio (local), OpenRouter, OpenAI, Anthropic.

---

## Discussion

**Why flat-text schemas fail:** LLMs treat graph prompts as unstructured text, failing to parse adjacency relations. Without human oversight, hallucinations and contextual errors propagate undetected.

**Complexity-Difficulty Misalignment:** Join count ≠ correction difficulty. The GRL Router routes on join complexity but correction difficulty depends on error type — a finding that explains why the full pipeline (Config D) underperforms graph-path feedback alone (Config B) at scale.

**Why graph-path injection works:** Replacing vague NL hints ("you are missing a table") with explicit FK traversal paths (`CAR_MAKERS → MODEL_LIST → CAR_NAMES`) gives the model a navigational map rather than a street list. This broke the 0% multi-hop barrier.

---

## Citation

```bibtex
@misc{taufiq2026hitlgraph,
  title   = {Human-in-the-Loop Structured Reasoning with Schema Graph Augmentation},
  author  = {Hudzaifah Bin Muhammad Taufiq},
  year    = {2026},
  school  = {Nanyang Technological University},
  note    = {Final Year Project, College of Computing and Data Science.
             Supervisor: Assoc. Prof. Ke Yiping, Kelly}
}
```
