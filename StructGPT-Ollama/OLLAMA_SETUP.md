# StructGPT with Ollama (Gemma3)

This is a modified version of StructGPT that uses **Ollama with Gemma3** instead of OpenAI's API.

## Changes Made

### 1. Removed OpenAI Dependencies
- Removed `openai` library dependency
- Replaced with direct HTTP calls to Ollama API using `requests` library
- No API keys needed!

### 2. Modified Python Scripts
Created three Ollama-compatible scripts:
- `structgpt_for_text_to_sql_ollama.py` - For Spider dataset (text-to-SQL)
- `structgpt_for_tableqa_ollama.py` - For table QA (TabFact, WTQ, WikiSQL)
- `structgpt_for_webqsp_ollama.py` - For knowledge graph QA (WebQSP)

### 3. Created Ollama Bash Scripts
All located in `scripts/` directory:
- `run_spider_wo_icl_v1_ollama.sh`
- `run_tabfact_wo_icl_v1_ollama.sh`
- `run_webqsp_wo_icl_v1_ollama.sh`
- `run_wtq_wo_icl_v1_ollama.sh`
- `run_wikisql_wo_icl_v1_ollama.sh`

## Quick Start Guide

### Step 1: Setup Python Environment

```bash
# Navigate to project directory
cd /home/hud/github/StructGPT

# Activate virtual environment
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Download NLTK data (required for evaluation)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

### Step 2: Setup Ollama

Open a **separate terminal** and start Ollama server:

```bash
# Start Ollama server (keep this running)
ollama serve
```

Then in **another terminal**, pull the model:

```bash
# Pull gemma3 model (one-time setup)
ollama pull gemma3

# Verify the model is available
ollama list
```

### Step 3: Run StructGPT Scripts

Back in your main terminal (with venv activated):

```bash
# Make sure you're in the project directory with venv activated
cd /home/hud/github/StructGPT
source venv/bin/activate

# Run Spider dataset (text-to-SQL)
bash ./scripts/run_spider_wo_icl_v1_ollama.sh

# Or run TabFact dataset (table verification)
bash ./scripts/run_tabfact_wo_icl_v1_ollama.sh

# Or run WikiTableQuestions
bash ./scripts/run_wtq_wo_icl_v1_ollama.sh

# Or run WikiSQL
bash ./scripts/run_wikisql_wo_icl_v1_ollama.sh

# Or run WebQSP (knowledge graph QA)
bash ./scripts/run_webqsp_wo_icl_v1_ollama.sh
```

### Step 4: Evaluate Results

After running a script, evaluate the predictions:

#### Simple String-Based Evaluation (No Database Required)

```bash
# Evaluate Spider results
python simple_evaluate.py ./outputs/spider/output_wo_icl_v1_ollama.jsonl

# Evaluate TabFact results
python simple_evaluate.py ./outputs/tabfact/output_wo_icl_v1_ollama.jsonl

# Evaluate WTQ results
python simple_evaluate.py ./outputs/wtq/output_wo_icl_v1_ollama.jsonl

# Evaluate WikiSQL results
python simple_evaluate.py ./outputs/wikisql/output_wo_icl_v1_ollama.jsonl
```

#### Full Evaluation (Requires Database Files)

**Note:** Full execution-based evaluation requires downloading the actual database files, which were not included in your dataset download.

If you have the database files:

```bash
# Spider evaluation with execution
python evaluate_for_spider.py \
    --path ./outputs/spider/output_wo_icl_v1_ollama.jsonl \
    --db data/spider/database \
    --table data/spider/tables.json \
    --etype exec
```

## Complete Workflow Example

Here's a complete example from start to finish:

```bash
# 1. Setup (one-time)
cd /home/hud/github/StructGPT
source venv/bin/activate
pip install -r requirements.txt

# 2. Start Ollama (in separate terminal)
# Terminal 2: ollama serve

# 3. Pull model (one-time, in another terminal)
# Terminal 3: ollama pull gemma3

# 4. Run Spider dataset
bash ./scripts/run_spider_wo_icl_v1_ollama.sh
# This will take ~1 hour for 1034 examples

# 5. Evaluate results
python simple_evaluate.py ./outputs/spider/output_wo_icl_v1_ollama.jsonl

# 6. View results
cat ./outputs/spider/output_wo_icl_v1_ollama.jsonl | head -5
```

### Configuration

To change the Ollama model, edit the model name in the Python scripts:

```python
# At the top of each *_ollama.py file
OLLAMA_MODEL = "gemma3"  # Change to your preferred model
```

To change the Ollama API endpoint (if running on a different host/port):

```python
OLLAMA_API_BASE = "http://localhost:11434/api"  # Change as needed
```

## Output

Results are saved to `./outputs/[dataset]/` directory:
- `output_wo_icl_v1_ollama.jsonl` - Predictions
- `chat_wo_icl_v1_ollama.txt` - Conversation logs

## Dependencies

All dependencies are listed in `requirements.txt`:
- `requests>=2.31.0` - For HTTP calls to Ollama
- `pandas>=2.0.0` - Data processing
- `numpy>=1.24.0` - Numerical operations
- `tqdm>=4.65.0` - Progress bars

## Troubleshooting

### Connection Error
If you see "Connection Error - Is Ollama running?":
1. Make sure Ollama is running: `ollama serve`
2. Check that the model is available: `ollama list`
3. Verify the API endpoint is correct

### Model Not Found
If Ollama says the model doesn't exist:
```bash
ollama pull gemma3
```

### Out of Memory
If you get OOM errors, try a smaller model or increase your system memory allocation.

## Expected Performance

### Spider Dataset (Text-to-SQL)
With Gemma3 model on 1034 test examples:
- **Processing Time:** ~58 minutes (~3.4 seconds per example)
- **Exact Match Accuracy:** ~27.56%
- **GPU Utilization:** 85-95% (RTX 3060 Ti)
- **VRAM Usage:** ~5-6 GB

### Comparison with OpenAI GPT-3.5-turbo
- GPT-3.5-turbo: ~70-75% accuracy
- Gemma3 (7-8B): ~27.56% accuracy
- Trade-off: Lower accuracy but free and runs locally

## Performance Notes

- Single process mode (`--num_process 1`) is used for simplicity
- For faster processing on multiple samples, you could implement parallel processing
- Ollama models are generally slower than API-based models but run locally and are free
- Expected inference speed: 10-30 tokens/second on consumer GPUs

## Checking GPU Usage

To verify Ollama is using your GPU:

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or use gpustat
pip install gpustat
gpustat -i 1
```

You should see:
- GPU utilization: 85-95%
- Memory usage: 5-6 GB for Gemma3
- Ollama process listed in GPU processes

## Tips for Better Results

1. **Use smaller, more focused prompts** - Gemma3 works better with concise instructions
2. **Try different models** - `phi3:mini` or `llama3.2:8b` might perform differently
3. **Adjust temperature** - Currently set to 0 for deterministic outputs
4. **Add few-shot examples** - Provide example SQL queries in prompts (ICL mode)
5. **Fine-tune on your data** - For production use, consider fine-tuning
