# StructGPT-Ollama Evaluation Report
## Spider Dataset Text-to-SQL Evaluation

**Date:** November 5, 2025
**Model:** Gemma3 via Ollama
**Output File:** `outputs/spider/output_wo_icl_v1_ollama.jsonl`

---

## 1. Dataset Overview

- **Total Questions:** 1,034
- **Evaluation Method:** Execution-based (queries executed on actual SQLite databases)
- **Database Files:** 166 SQLite databases

---

## 2. Execution Accuracy Results

### Overall Performance

| Metric | Value |
|--------|-------|
| **Overall Execution Accuracy** | **63.2%** |
| **Total Correct Executions** | 654 / 1,034 |

### Breakdown by Difficulty Level

| Difficulty | Count | Execution Accuracy | Correct / Total |
|------------|-------|-------------------|-----------------|
| **Easy** | 248 | **81.5%** | 202 / 248 |
| **Medium** | 446 | **70.6%** | 315 / 446 |
| **Hard** | 174 | **50.6%** | 88 / 174 |
| **Extra** | 166 | **29.5%** | 49 / 166 |

---

## 3. String-Based Exact Match Evaluation

For comparison, here are the string-based exact match results:

- **Total Questions:** 1,034
- **Exact String Matches:** 285
- **String-Based Accuracy:** 27.56%

**Note:** The execution-based accuracy (63.2%) is significantly higher than string-based matching (27.56%) because:
- Queries can be semantically equivalent without being syntactically identical
- Different query structures can produce the same results
- Column/table ordering differences don't affect execution results

---

## 4. Generation Performance

### Time Metrics
- **Total Processing Time:** ~58 minutes
- **Average Time per Query:** ~3.4 seconds
- **Total Queries Processed:** 1,034

### Resource Usage
- **GPU Utilization:** 85-95%
- **VRAM Usage:** ~5-6 GB
- **GPU Model:** RTX 3060 Ti (as documented)

---

## 5. Common Error Patterns

Based on the evaluation output, the most common issues include:

1. **Missing DISTINCT Keywords** - Model sometimes omits DISTINCT when needed
2. **Incorrect INTERSECT/UNION Usage** - Model uses OR/AND instead of set operations
3. **Schema Understanding** - Some queries reference non-existent columns or tables
4. **Aggregation Errors** - Incorrect use of COUNT(*) vs COUNT(DISTINCT column)
5. **Subquery Complexity** - Struggles with complex nested subqueries
6. **Join Conditions** - Sometimes incorrect join predicates

---

## 6. Performance by SQL Complexity

### Easy Queries (81.5% accuracy)
- Simple SELECT statements
- Basic WHERE clauses
- Single table operations

### Medium Queries (70.6% accuracy)
- JOIN operations
- GROUP BY and aggregations
- Basic subqueries

### Hard Queries (50.6% accuracy)
- Complex JOINs (3+ tables)
- INTERSECT/EXCEPT operations
- Nested subqueries

### Extra Hard Queries (29.5% accuracy)
- Multiple set operations
- Complex nested queries
- Advanced aggregations with multiple conditions

---

## 7. Sample Predictions

### Successful Predictions

**Question:** "How many singers do we have?"
- **Gold SQL:** `SELECT count(*) FROM singer`
- **Predicted SQL:** `SELECT count(*) FROM singer`
- **Result:** ✅ Correct

### Failed Predictions

**Question:** "What is the maximum capacity and average of all stadiums?"
- **Gold SQL:** `SELECT max(capacity), average FROM stadium`
- **Predicted SQL:** `SELECT MAX(Capacity), AVG(Capacity) FROM stadium`
- **Result:** ❌ Used AVG(Capacity) instead of the 'average' column

---

## 8. Comparison with Baselines

| Model | Execution Accuracy | Notes |
|-------|-------------------|-------|
| **Gemma3 (Local)** | **63.2%** | This evaluation |
| GPT-3.5-turbo (Reported) | ~70-75% | From original StructGPT paper |
| String Match (Gemma3) | 27.56% | Exact string matching only |

---

## 9. Key Findings

### Strengths
1. ✅ **Strong performance on simple queries** (81.5% on easy questions)
2. ✅ **Good at basic JOIN operations**
3. ✅ **Handles aggregations reasonably well**
4. ✅ **Completely free and runs locally**

### Weaknesses
1. ❌ **Struggles with complex set operations** (INTERSECT, EXCEPT, UNION)
2. ❌ **Poor performance on extra-hard queries** (29.5%)
3. ❌ **Sometimes references non-existent schema elements**
4. ❌ **Inconsistent with DISTINCT keyword usage**

---

## 10. Recommendations for Improvement

1. **Fine-tuning:** Train on SQL-specific datasets to improve schema understanding
2. **Prompt Engineering:** Add more examples of complex queries (few-shot learning)
3. **Schema Validation:** Add post-processing to validate against actual schema
4. **Larger Model:** Try llama3.2:70b or other larger models
5. **Ensemble Approach:** Generate multiple candidates and select best via execution

---

## 11. Files Generated

- `outputs/spider/output_wo_icl_v1_ollama.jsonl` - Generated predictions
- `outputs/spider/chat_wo_icl_v1_ollama.txt` - Conversation logs
- `evaluation_report.txt` - Detailed evaluation output
- `EVALUATION_REPORT.md` - This summary report

---

## 12. Conclusion

The Gemma3 model via Ollama achieved **63.2% execution accuracy** on the Spider dataset, which is:

- ✅ **Impressive for a local, free model**
- ✅ **Better than expected** given the complexity of text-to-SQL
- ❌ **Below GPT-3.5-turbo** performance (~70-75%)
- ✅ **Usable for real applications** with proper error handling

The model excels at simple to medium complexity queries but struggles with very complex SQL operations, particularly set operations and deeply nested queries.

---

**Generated by:** StructGPT-Ollama Evaluation Pipeline
**Evaluation Script:** `evaluate_for_spider.py` (with markdown cleaning fixes)
