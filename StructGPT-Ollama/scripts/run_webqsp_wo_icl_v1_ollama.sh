#!/usr/bin/env bash

# StructGPT for WebQSP dataset using Ollama with gemma3 model

./venv/bin/python3 structgpt_for_webqsp_ollama.py \
--num_process 1 \
--prompt_path ./prompts/prompt_for_webqsp.json --max_tokens 300 --prompt_name chat_v1 \
--kg_source_path ./data/webqsp/subgraph_2hop_triples.npy \
--ent_type_path ./data/webqsp/ent_type_ary.npy \
--ent2id_path ./data/webqsp/ent2id.pickle \
--rel2id_path ./data/webqsp/rel2id.pickle \
--ent2name_path ./data/webqsp/entity_name.pickle \
--max_triples_per_relation 60 \
--input_path ./data/webqsp/webqsp_simple_test.jsonl \
--output_path ./outputs/webqsp/output_wo_icl_v1_ollama.jsonl \
--chat_log_path ./outputs/webqsp/chat_wo_icl_v1_ollama.txt

# Evaluate the results
./venv/bin/python3 evaluate_for_webqsp.py --output_path ./outputs/webqsp/output_wo_icl_v1_ollama.jsonl
