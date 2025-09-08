# README.md

# GraphLLM Backend

Local-first, API-driven backend for LLM × Graph Representation Learning experiments.

## Features

- Text-attributed graph processing with multimodal inputs
- TF-IDF based retrieval with k-hop expansion
- Order-sensitive graph serialization (GraphSOS)
- Provider-agnostic LLM integration
- Comprehensive experiment logging and metrics
- CSV/JSONL export and visualization

## Quick Start

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your LLM API key

# Run server
uvicorn main:app --reload
````

---

## Test Run Workflow

### 1. Setup
- Ensure your `.env` is configured with your OpenAI API key, endpoint, and model.
- Start the backend:
  ```sh
  python -m uvicorn main:app --reload
  ```

### 2. API Usage Example

#### Create a graph
```sh
curl -X POST http://localhost:8000/graphs \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Graph", "description": "Sample graph for testing"}'
```

#### Upload graph data
```sh
curl -X POST http://localhost:8000/graphs/<graph_id>/upload \
  -F "edges_file=@sample_data/edges.csv" \
  -F "nodes_file=@sample_data/nodes.csv" \
  -F "node_text_file=@sample_data/node_text.csv"
```

#### Create a run
```sh
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"graph_id": "<graph_id>", "task_type": "QA"}'
```

#### Run pipeline stages
```sh
curl -X POST http://localhost:8000/pipeline/<run_id>/unigraph
curl -X POST http://localhost:8000/pipeline/<run_id>/gnn-adapter
curl -X POST http://localhost:8000/pipeline/<run_id>/translator
curl -X POST http://localhost:8000/pipeline/<run_id>/rag/prepare
```

#### Query the system
```sh
curl -X POST http://localhost:8000/query/<run_id> \
  -H "Content-Type: application/json" \
  -d '{"query_text": "What is this graph about?", "top_k": 5, "hops": 2}'
```

#### Export results
```sh
curl http://localhost:8000/results/<run_id>/export.csv -o results.csv
curl -X POST http://localhost:8000/results/<run_id>/charts \
  -H "Content-Type: application/json" \
  -d '{"charts": ["acc_vs_k", "tokens_vs_context"]}'
```

---

### 3. Example LLM Response

```
The graph represents a professional network of individuals with expertise in software engineering, machine learning, and data science. It illustrates relationships among several key figures:

- Alice: senior software engineer
- Bob: data scientist
- Charlie: ML team lead
- Diana: engineering director
- Eve: intern

The paths connecting these individuals suggest collaboration and mentorship relationships, highlighting a network of expertise and collaboration in technology and data science.
```

---

### 4. Tips & Troubleshooting
- Ensure your `.env` is loaded and all LLM settings are correct.
- Use valid model names and API keys.
- If you encounter errors, check backend logs for details.
- To reset the backend, delete the database file and restart the server.

---