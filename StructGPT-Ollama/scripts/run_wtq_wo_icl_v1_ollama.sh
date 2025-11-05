#!/usr/bin/env bash

# StructGPT for WTQ dataset using Ollama with gemma3 model

./venv/bin/python3 structgpt_for_tableqa_ollama.py \
--num_process 1 \
--prompt_path ./prompts/prompt_for_wtq.json --prompt_name chat_v1 \
--input_path ./data/wtq/wikitq_test.json \
--output_path ./outputs/wtq/output_wo_icl_v1_ollama.jsonl \
--chat_log_path ./outputs/wtq/chat_wo_icl_v1_ollama.txt --max_tokens 350

# Evaluate the results
./venv/bin/python3 evaluate_for_wikisql.py --ori_path ./data/wtq/wikitq_test.json --inp_path ./outputs/wtq/output_wo_icl_v1_ollama.jsonl
