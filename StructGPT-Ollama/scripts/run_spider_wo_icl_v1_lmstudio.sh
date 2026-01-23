#!/usr/bin/env bash

# StructGPT for Spider dataset using LM Studio with Qwen 2.5 Coder 14B model

# Activate virtual environment
source ../venv/bin/activate

python3 structgpt_for_text_to_sql_lmstudio.py \
--num_process 1 \
--prompt_path ./prompts/prompt_for_spider.json --prompt_name chat_v1 \
--input_path ./outputs/spider/dev.jsonl \
--output_path ./outputs/spider/output_wo_icl_v1_lmstudio.jsonl \
--chat_log_path ./outputs/spider/chat_wo_icl_v1_lmstudio.txt \
--db_path ./outputs/spider/all_tables_content.json \
--schema_path ./outputs/spider/tables.json

# Evaluate the results
# python3 evaluate_for_spider.py --path ./outputs/spider/output_wo_icl_v1_lmstudio.jsonl --db=outputs/spider/database --table=outputs/spider/tables.json --etype=exec
