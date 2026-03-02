# HITL SQL Testbed — Session Context for Claude

## Project Goal

Compare a **Human-in-the-Loop (HITL) Create-Verify-Refine loop** against a
**baseline of 84% execution-match accuracy** on 250 Spider SQL queries using
local LLMs via LM Studio.

The hypothesis: giving the AI up to 3 iterations of oracle feedback will push
accuracy above 84%.

---

## What Has Been Built

### File: `hitl_sql_testbed.py` (project root)

A standalone Python testbed with:

| Component | Role |
|---|---|
| `AIAgent` | Stateful multi-turn LLM session. Translates NL → SQL. Responds in structured JSON `{thought, clarification_needed, question, sql}`. Uses **Chain-of-Thought** before writing SQL. |
| `HumanProxy` | Stateless Oracle simulator. Receives the answer key entry and the AI's message. Gives clarification answers or targeted feedback. |
| `run_hitl_loop()` | Core loop: clarification → SQL → execute → compare → feedback → refine (up to `MAX_ITERATIONS=3`). |
| `build_answer_key_from_baseline()` | Builds `answer_key.json` from the existing `output_sample_250_qwen.jsonl` baseline, reusing pre-computed `answer_text` gold results. |
| Checkpointing | Saves `hitl_results.json` after **every sample** so the run can be resumed with `--resume`. |
| Final report | Prints HITL accuracy, delta vs baseline, avg iterations, and breakdown of passes by iteration number. |

### Other Modified Files

| File | Change |
|---|---|
| `backend/.env` | Switched `LLM_PROVIDER=lmstudio`, `LLM_ENDPOINT_BASE=http://localhost:1234/v1`, `LLM_MODEL=qwen/qwen2.5-coder-14b` |
| `backend/agents/tools/llm_client.py` | Added `"lmstudio"` to supported providers list (both request-build and response-parse branches) |

---

## LM Studio Status (last checked)

| Model | Status |
|---|---|
| `qwen/qwen2.5-coder-14b` | **Working** — used as `AI_AGENT_MODEL` |
| `deepseek/deepseek-r1-0528-qwen3-8b` | **Fails to load** — returns `400 Failed to load model` when called. Listed in `/v1/models` but does not actually serve requests. Likely a GPU memory issue. |
| `text-embedding-nomic-embed-text-v1.5` | Embedding model — not for chat |

### Blocking Issue

`HUMAN_PROXY_MODEL` is currently set to `deepseek/deepseek-r1-0528-qwen3-8b`
which fails. **This must be resolved before running the experiment.**

### Resolution Options (pick one)

1. **Use Qwen as both agents** — set `HUMAN_PROXY_MODEL = "qwen/qwen2.5-coder-14b"` in the config block at the top of `hitl_sql_testbed.py`. Simple, works immediately, both agents on same model.
2. **Load DeepSeek in LM Studio** — open LM Studio, go to the model, manually load it. If it loads successfully, re-test with the curl command below.
3. **Use Claude 3.5 as Human_Proxy** — set `HUMAN_PROXY_PROVIDER = "anthropic"`, `pip install anthropic`, `export ANTHROPIC_API_KEY=sk-ant-...`. Best quality oracle but requires API key.

### Quick test to check if DeepSeek loads

```bash
curl -s -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek/deepseek-r1-0528-qwen3-8b", "messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 10, "stream": false}'
```

---

## Baseline Data

| File | Description |
|---|---|
| `StructGPT-Ollama/outputs/spider/output_sample_250_qwen.jsonl` | 250-sample baseline run. 84% execution match. Fields: `question`, `query` (gold SQL), `db_id`, `answer_text` (gold results), `Prediction` (AI output). |
| `StructGPT-Ollama/data/spider/spider_data/database/` | SQLite databases for execution matching |
| `StructGPT-Ollama/data/spider/spider_data/dev.json` | Raw Spider dev set |

---

## How to Run the Experiment

### Step 1 — Resolve the Human_Proxy model issue (see above)

### Step 2 — Run (first time)

```bash
cd /home/hud/Projects/LLMxGRL

python hitl_sql_testbed.py \
  --from_baseline StructGPT-Ollama/outputs/spider/output_sample_250_qwen.jsonl \
  --limit 250
```

### Step 3 — Resume if interrupted

```bash
python hitl_sql_testbed.py \
  --from_baseline StructGPT-Ollama/outputs/spider/output_sample_250_qwen.jsonl \
  --limit 250 \
  --resume
```

### Step 4 — Read results

```bash
python3 -c "
import json
with open('hitl_results.json') as f: d = json.load(f)
s = d['summary']
print(f'HITL:     {s[\"passed\"]}/{s[\"total\"]}  ({100*s[\"hitl_accuracy\"]:.1f}%)')
print(f'Baseline: ({100*s[\"baseline_accuracy\"]:.1f}%)')
print(f'Delta:    {100*s[\"delta\"]:+.1f} pp')
print(f'Passes by iteration: {s[\"fixed_at_iteration\"]}')
"
```

---

## Config Block (top of `hitl_sql_testbed.py`)

```python
LMSTUDIO_BASE        = "http://localhost:1234/v1"
AI_AGENT_MODEL       = "qwen/qwen2.5-coder-14b"
HUMAN_PROXY_MODEL    = "deepseek/deepseek-r1-0528-qwen3-8b"   # ← BROKEN, change this
HUMAN_PROXY_PROVIDER = "lmstudio"    # "lmstudio" | "anthropic"
ANTHROPIC_MODEL      = "claude-3-5-sonnet-20241022"
MAX_ITERATIONS       = 3
REQUEST_TIMEOUT      = 180
BASELINE_ACCURACY    = 0.84
```

---

## System Prompts Used

### AI_Agent (Qwen 2.5 Coder)

> You are a Graph-Aware SQL Assistant. You translate natural language into SQL queries based on a structured database schema.
> Strategy: 1. Analyze (Chain-of-Thought), 2. Clarify if ambiguous, 3. Refine on feedback.
> Respond ONLY in JSON: `{"thought": "...", "clarification_needed": bool, "question": "...", "sql": "..."}`

### Human_Proxy (Oracle Simulator)

> You are an Oracle User Simulator. Your goal is to help an AI assistant correctly interpret a user's intent based on a provided Answer Key.
> — Answer clarifications using only the Answer Key.
> — If SQL is wrong, give specific actionable feedback.
> — Do not give away the final answer; guide the AI.
> — Max 2-3 sentences.

---

## Next Steps for Claude

1. **Ask the user which Human_Proxy option they want** (Qwen same model / load DeepSeek / Claude 3.5)
2. **Fix `HUMAN_PROXY_MODEL`** in `hitl_sql_testbed.py` accordingly
3. **Run the end-to-end test** (one sample) to confirm both agents call successfully before starting the full 250-sample run
4. **Start the full run** and monitor progress
5. **Analyze results** — compare HITL accuracy vs 84% baseline, look at the `fixed_at_iteration` breakdown
