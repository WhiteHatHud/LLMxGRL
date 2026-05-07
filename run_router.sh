#!/bin/bash
venv/bin/python graph_hitl_framework/grl_router.py --answer_key answer_key.json --embeddings graph_hitl_framework/cache/schema_embeddings.pkl --tables StructGPT-Ollama/data/spider/spider_data/tables.json --output graph_hitl_framework/cache/grl_router_model.pkl
