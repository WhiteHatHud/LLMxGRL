#!/usr/bin/env bash

# StructGPT for TabFact dataset using Ollama with gemma3 model

./venv/bin/python3 structgpt_for_tableqa_ollama.py \
--num_process 1 \
--prompt_path ./prompts/prompt_for_tabfact.json --prompt_name chat_v1 \
--input_path ./data/tabfact/tab_fact_test.json \
--output_path ./outputs/tabfact/output_wo_icl_v1_ollama.jsonl \
--chat_log_path ./outputs/tabfact/chat_wo_icl_v1_ollama.txt --max_tokens 350

# Evaluate the results
./venv/bin/python3 evaluate_for_tabfact.py --ori_path ./data/tabfact/tab_fact_test.json --inp_path ./outputs/tabfact/output_wo_icl_v1_ollama.jsonl
