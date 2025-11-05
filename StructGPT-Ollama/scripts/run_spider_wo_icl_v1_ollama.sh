#!/usr/bin/env bash

# StructGPT for Spider dataset using Ollama with gemma3 model

./venv/bin/python3 structgpt_for_text_to_sql_ollama.py \
--num_process 1 \
--prompt_path ./prompts/prompt_for_spider.json --prompt_name chat_v1 \
--input_path ./data/spider/dev.jsonl \
--output_path ./outputs/spider/output_wo_icl_v1_ollama.jsonl \
--chat_log_path ./outputs/spider/chat_wo_icl_v1_ollama.txt \
--db_path ./data/spider/all_tables_content.json \
--schema_path ./data/spider/tables.json

# Evaluate the results
./venv/bin/python3 evaluate_for_spider.py --path ./outputs/spider/output_wo_icl_v1_ollama.jsonl --db=data/spider/database --table=data/spider/tables.json --etype=exec
