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