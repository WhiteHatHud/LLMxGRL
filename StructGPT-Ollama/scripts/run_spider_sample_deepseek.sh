#!/usr/bin/env bash

# StructGPT for Spider dataset (100 sample) using LM Studio with DeepSeek R1 model

# Activate virtual environment
source ../venv/bin/activate

python3 structgpt_for_text_to_sql_deepseek.py \
--num_process 1 \
--prompt_path ./prompts/prompt_for_spider.json --prompt_name chat_v1 \
--input_path ./outputs/spider/dev_sample_100.jsonl \
--output_path ./outputs/spider/output_sample_deepseek.jsonl \
--chat_log_path ./outputs/spider/chat_sample_deepseek.txt \
--db_path ./outputs/spider/all_tables_content.json \
--schema_path ./outputs/spider/tables.json

echo ""
echo "==========================================="
echo "DeepSeek R1 Test Complete!"
echo "Results saved to: ./outputs/spider/output_sample_deepseek.jsonl"
echo "==========================================="

# Count results
echo "Total examples processed:"
wc -l ./outputs/spider/output_sample_deepseek.jsonl
