# Model Benchmark Results - Spider SQL Generation

This directory contains comprehensive benchmark results comparing **Qwen 2.5 Coder 14B** and **DeepSeek R1 Qwen3 8B** models on the Spider SQL generation task (100 samples).

## Files

### Main Reports
- **`COMPARISON_REPORT.txt`** - Complete performance comparison (speed, file sizes, recommendations)
- **`ACCURACY_RESULTS.txt`** - SQL accuracy evaluation results and methodology
- **`qwen_2.5_coder_14b_results.txt`** - Detailed Qwen metrics
- **`deepseek_r1_qwen3_8b_results.txt`** - Detailed DeepSeek metrics

### Raw Data
- **`output_sample_qwen.jsonl`** - Qwen generated SQL queries (148 KB)
- **`output_sample_deepseek.jsonl`** - DeepSeek generated SQL queries (254 KB)
- **`chat_sample_qwen.txt`** - Qwen interaction logs (275 KB)
- **`chat_sample_deepseek.txt`** - DeepSeek interaction logs (568 KB)

### Evaluation Script
- **`simple_sql_accuracy.py`** - String-based SQL accuracy evaluator

## Quick Summary

### Speed Performance
| Model | Total Time | Avg/Sample | Winner |
|-------|-----------|------------|---------|
| Qwen 2.5 Coder 14B | 84m 3s | 50.43s | |
| DeepSeek R1 8B | 32m 12s | 19.32s | ✓ **2.6x faster** |

### Accuracy (String Matching)
| Model | Normalized Matches | Accuracy |
|-------|-------------------|----------|
| Qwen 2.5 Coder 14B | 30/100 | 30.00% |
| DeepSeek R1 8B | 1/100 | 1.00% |

**Note:** String-match accuracy favors concise outputs. DeepSeek's verbose reasoning format makes extraction challenging but doesn't necessarily indicate lower SQL quality.

### Output Characteristics
- **Qwen**: Concise SQL, minimal explanation, 148 KB output
- **DeepSeek**: Verbose reasoning with chain-of-thought, 254 KB output (72% larger)

## Usage Recommendations

### Choose Qwen 2.5 Coder 14B when:
- You need clean, concise SQL generation
- Direct output format is important
- Minimal file sizes matter
- Working with 14B model is acceptable

### Choose DeepSeek R1 8B when:
- Speed is critical (2.6x faster!)
- You want detailed reasoning and explanations
- Chain-of-thought analysis is valuable
- Smaller model (8B vs 14B) is preferred
- Processing large batches quickly

## Full Spider Evaluation

For production use, we recommend downloading the Spider database files and running the full execution-based evaluation:

```bash
# Download Spider dataset
wget https://yale-lily.github.io/spider/spider.zip
unzip spider.zip
cp -r spider/database outputs/spider/

# Run full evaluation
python evaluate_for_spider.py \
  --path outputs/spider/output_sample_qwen.jsonl \
  --db outputs/spider/database \
  --table outputs/spider/tables.json \
  --etype exec
```

This provides:
- Execution accuracy (queries run on databases)
- Component matching (SELECT, WHERE, GROUP BY)
- Difficulty-level breakdown (easy/medium/hard/extra)

## Test Environment
- **Date**: 2026-01-13
- **Platform**: LM Studio (localhost:1234)
- **Dataset**: Spider SQL Generation (100 sample subset)
- **Task**: Text-to-SQL generation
