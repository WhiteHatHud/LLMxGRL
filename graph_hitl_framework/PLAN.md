# Graph-Guided HITL Framework — Implementation Plan

## What This Framework Builds

A Graph Representation Learning-enhanced Human-in-the-Loop pipeline that
integrates learned schema graph embeddings directly into the HITL feedback loop.
This makes GRL and HITL genuinely integrated, not two separate experiments.

---

## Existing Helper Tools (Do Not Rewrite)

### Schema & Graph
| Tool | Location | What to reuse |
|------|----------|---------------|
| `SchemaGraph` | `StructGPT-Ollama/schema_graph.py` | `get_fk_path()`, `_build_graph()`, `_find_chain()`, `_format_path()` |
| `extract_tables_from_sql()` | `StructGPT-Ollama/schema_graph.py` | Parse tables from SQL |
| `count_joins_in_sql()` | `StructGPT-Ollama/schema_graph.py` | Join depth detection |
| `clean_sql_prediction()` | `StructGPT-Ollama/schema_graph.py` | Strip markdown fences |
| `serialise_schema()` | `StructGPT-Ollama/graph_augmented_rerun.py` | Schema text + FK text |
| `exec_sql()` / `results_match()` | `StructGPT-Ollama/graph_augmented_rerun.py` | SQL execution matching |

### LLM Inference
| Tool | Location | What to reuse |
|------|----------|---------------|
| `_call_lmstudio()` | `hitl_sql_testbed.py` | LM Studio inference |
| `_parse_agent_json()` | `hitl_sql_testbed.py` | JSON response parsing |
| `AIAgent` class | `hitl_sql_testbed.py` | Stateful multi-turn agent |
| `HumanProxy` class | `hitl_sql_testbed.py` | Oracle proxy |

### Answer Matching & Evaluation
| Tool | Location | What to reuse |
|------|----------|---------------|
| `answers_match()` | `hitl_wtq_testbed.py` | WTQ answer matching |
| `verdict_matches()` | `hitl_tabfact_testbed.py` | TabFact verdict matching |
| `normalize_verdict()` | `hitl_tabfact_testbed.py` | Yes/no normalisation |
| `serialize_table()` | `hitl_wtq_testbed.py` | Table to text |
| `finalize_output()` | `hitl_sql_testbed.py` | Results summary |
| `save_checkpoint()` | `hitl_sql_testbed.py` | Checkpointing |

### Structural Features (Stubs to Activate)
| Tool | Location | What to activate |
|------|----------|-----------------|
| `GNNAdapter.compute_features()` | `backend/agents/tools/gnn_adapter.py` | degree, pagerank, clustering |
| `GraphTranslator` templates | `backend/agents/tools/graph_translator.py` | Node/edge serialization |

---

## Architecture: Four New Components

```
QUERY IN
    │
    ▼
┌─────────────────────────────┐
│  1. GRL ROUTER              │  ← Node2Vec embeddings + classifier
│  Predicts join complexity   │    Routes: single-pass / standard HITL
│  from graph structure       │            / graph-augmented HITL
└─────────────┬───────────────┘
              │
    ┌─────────▼──────────┐
    │  AI AGENT (Qwen)   │  ← Existing, reused unchanged
    │  Generates SQL/    │
    │  answer/verdict    │
    └─────────┬──────────┘
              │ FAIL?
    ┌─────────▼──────────────────┐
    │  2. GRL ERROR CLASSIFIER   │  ← Graph features → surface vs structural
    │  Classifies error type     │
    └─────────┬──────────────────┘
              │
    ┌─────────▼────────────────────────┐
    │  3. GRAPH-AUGMENTED ORACLE       │  ← Existing proxy + FK path injection
    │  Generates structure-aware       │    Path computed from SchemaGraph
    │  feedback with graph path        │
    └─────────┬────────────────────────┘
              │
    ┌─────────▼──────────┐
    │  AI AGENT retries  │  ← With graph-path context in history
    └─────────┬──────────┘
              │
    ┌─────────▼──────────────────────┐
    │  4. EVALUATION & COMPARISON    │  ← Same metrics as baseline HITL
    │  Compare: Baseline vs          │    Stratified by join depth
    │  Graph-Guided HITL             │
    └────────────────────────────────┘
```

---

## Five Files to Build

```
graph_hitl_framework/
├── PLAN.md                          ← This file
├── node2vec_embeddings.py           ← Component 1: Train/load Node2Vec on Spider schemas
├── grl_router.py                    ← Component 2: Classify query complexity, route to strategy
├── grl_error_classifier.py          ← Component 3: Surface vs structural error detection
├── graph_oracle_feedback.py         ← Component 4: Graph-path-augmented proxy feedback
└── graph_hitl_pipeline.py           ← Component 5: Full wired system (router → HITL → classifier → feedback)
```

---

## Component 1: node2vec_embeddings.py

**Purpose:** Learn structural embeddings for each table in every Spider database schema.

**Inputs:** `StructGPT-Ollama/data/spider/spider_data/tables.json`

**Method:**
- Build NetworkX graph per database using existing `SchemaGraph._build_graph()`
- Run Node2Vec (random walks, 64-dim embeddings)
- Also compute structural features: degree, PageRank, clustering coefficient
  (reusing `GNNAdapter` logic, adapted for Spider schemas)
- Cache embeddings to disk as `graph_hitl_framework/cache/schema_embeddings.pkl`

**Output:** Dict mapping `{db_id: {table_name: embedding_vector}}`

**Dependencies:** `node2vec`, `networkx`, `numpy`, existing `schema_graph.py`

---

## Component 2: grl_router.py

**Purpose:** Before any inference, predict query complexity and route to appropriate strategy.

**Inputs:** Natural language question + db_id

**Method:**
- Encode question with `sentence-transformers` (all-MiniLM-L6-v2)
- Load cached Node2Vec table embeddings for db_id
- Compute: max cosine similarity between query and each table embedding
- Add graph features: schema diameter, avg degree, number of tables
- Train logistic regression classifier on existing 250-sample labels
  (join depth already known from `count_joins_in_sql()` on gold SQL)

**Routes:**
- `simple` (0–1 joins predicted) → single pass
- `medium` (2 joins predicted) → standard HITL (3 iterations, no graph)
- `complex` (3+ joins predicted) → graph-augmented HITL

**Output:** `{"route": "complex", "predicted_joins": 3, "confidence": 0.87}`

---

## Component 3: grl_error_classifier.py

**Purpose:** After a failed iteration, classify whether error is surface or structural.

**Inputs:** Failed SQL, gold SQL, db_id, schema graph

**Method:**
- Extract tables from both predicted and gold SQL
- Compute: missing tables, graph distance between attempted joins, join count delta
- Rule-based classifier (no training needed — deterministic from graph):
  - `structural` if: missing tables AND graph_distance > 1
  - `surface` otherwise

**Output:** `{"error_type": "structural", "missing_tables": ["MODEL_LIST"], "graph_distance": 2}`

**Why rule-based is fine here:** The graph distance is an exact computation,
not a prediction. Structural errors have a precise graph-theoretic definition.

---

## Component 4: graph_oracle_feedback.py

**Purpose:** Generate richer oracle feedback that includes the computed FK traversal path.

**For SQL (Spider):**
- Standard proxy feedback + appended graph path:
  `"The schema graph shows the only valid path is: CAR_MAKERS -[Id=Maker]-> MODEL_LIST -[Model=Model]-> CAR_NAMES -[MakeId=Id]-> CARS_DATA"`
- Uses existing `SchemaGraph.get_fk_path()`

**For TabFact / WTQ:**
- No schema graph available — falls back to standard proxy
- These tasks have no FK structure to inject

**Output:** Augmented feedback string for injection into conversation history

---

## Component 5: graph_hitl_pipeline.py

**Purpose:** Wire all four components into a unified pipeline.
Run on 250 samples each for Spider, TabFact, WTQ.

**Evaluation outputs:**
- Per-sample: route taken, error types at each iteration, pass/fail
- Aggregate: accuracy by route type, accuracy by join depth
- Comparison table: Baseline HITL vs Graph-Guided HITL

**Checkpointing:** Same pattern as existing testbeds (resume-safe)

---

## Ablation Study (Built-in)

Run four conditions on the same 250 Spider samples:

| Condition | Router | Error Classifier | Graph Feedback |
|-----------|--------|-----------------|----------------|
| A: Baseline HITL | ✗ | ✗ | ✗ |
| B: + Graph Feedback only | ✗ | ✗ | ✓ |
| C: + GRL Router only | ✓ | ✗ | ✗ |
| D: Full Graph-Guided HITL | ✓ | ✓ | ✓ |

This isolates the contribution of each component.

---

## Questions for User Before Starting

See bottom of this file — answers needed before writing any code.

---

## Dependencies to Install

```bash
pip install node2vec sentence-transformers scikit-learn
```

- `node2vec`: Random walk graph embeddings
- `sentence-transformers`: Query encoding (all-MiniLM-L6-v2, runs locally, ~80MB)
- `scikit-learn`: Logistic regression classifier for GRL Router

All run locally, no API required.

---

## Data Paths (Confirmed)

| Data | Path |
|------|------|
| Spider tables.json | `StructGPT-Ollama/data/spider/spider_data/tables.json` |
| Spider databases | `StructGPT-Ollama/data/spider/spider_data/database/` |
| Spider baseline JSONL | `StructGPT-Ollama/outputs/spider/output_sample_250_qwen.jsonl` |
| TabFact answer key | `tabfact_answer_key.json` |
| WTQ answer key | `wtq_answer_key.json` |
| Existing HITL results | `hitl_results.json`, `hitl_tabfact_results.json`, `hitl_wtq_results.json` |

---

## Open Questions (Need Answers Before Coding)

1. **Node2Vec scope**: Should embeddings be trained only on Spider schemas,
   or also attempt a graph representation for TabFact/WTQ tables?
   (TabFact/WTQ have no FK structure — flat tables — so Node2Vec may not apply.)

2. **GRL Router training split**: The 250 Spider samples are the full evaluation set.
   Should the router be trained on ALL 250 (transductive) or split 200 train / 50 test?
   Using all 250 risks data leakage since those same samples are evaluated.

3. **LM Studio availability**: Will LM Studio be running during the full 250-sample
   runs for Spider, TabFact, and WTQ? (~3-4 hours total inference time expected.)

4. **Sentence transformer**: Can `all-MiniLM-L6-v2` be downloaded (~80MB)?
   Or should query encoding use a simpler TF-IDF approach instead?

5. **Ablation scope**: Run all four conditions (A/B/C/D) on Spider only,
   or also on TabFact and WTQ? (TabFact/WTQ won't benefit from graph routing
   since they have no FK structure — but confirms the router correctly
   routes them as "simple".)

6. **Existing HITL results**: The current `hitl_results.json` already has
   250-sample Spider results. Should the new pipeline RE-RUN those 250 samples
   from scratch (ground truth comparison), or use the cached results as
   Condition A (baseline) and only run new conditions B/C/D?
