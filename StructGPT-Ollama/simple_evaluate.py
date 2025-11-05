#!/usr/bin/env python3
"""
Simple evaluation for Spider predictions without requiring database files.
Compares predicted SQL queries with gold SQL queries using exact string matching.
"""

import json
import re
from collections import defaultdict

def normalize_sql(sql):
    """Normalize SQL query for comparison"""
    # Remove markdown code blocks
    sql = re.sub(r'```\w*\n?', '', sql)
    sql = re.sub(r'```', '', sql)
    # Convert to lowercase
    sql = sql.lower()
    # Remove all whitespace around commas
    sql = re.sub(r'\s*,\s*', ',', sql)
    # Normalize whitespace around operators and parentheses
    sql = re.sub(r'\s*=\s*', '=', sql)
    sql = re.sub(r'\s*\(\s*', '(', sql)
    sql = re.sub(r'\s*\)\s*', ')', sql)
    # Remove extra whitespace
    sql = ' '.join(sql.split())
    # Remove trailing semicolons
    sql = sql.rstrip(';')
    return sql.strip()

def evaluate_predictions(predictions_file):
    """Evaluate predictions against gold SQL"""

    print(f"\n{'='*80}")
    print(f"Evaluating: {predictions_file}")
    print(f"{'='*80}\n")

    with open(predictions_file, 'r') as f:
        predictions = [json.loads(line) for line in f]

    total = len(predictions)
    exact_match = 0
    difficulty_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    errors = []

    for idx, pred in enumerate(predictions):
        gold_sql = normalize_sql(pred['query'])
        pred_sql = normalize_sql(pred.get('Prediction', ''))

        # Get difficulty level if available
        difficulty = pred.get('difficulty', 'unknown')
        difficulty_stats[difficulty]['total'] += 1

        if gold_sql == pred_sql:
            exact_match += 1
            difficulty_stats[difficulty]['correct'] += 1
        else:
            # Store first 10 errors for review
            if len(errors) < 10:
                errors.append({
                    'id': pred.get('id', idx),
                    'question': pred.get('question', ''),
                    'gold': gold_sql,
                    'pred': pred_sql
                })

    # Print overall results
    accuracy = (exact_match / total) * 100
    print(f"Overall Results:")
    print(f"  Total Questions: {total}")
    print(f"  Exact Matches: {exact_match}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print()

    # Print by difficulty if available
    if len(difficulty_stats) > 1 and 'unknown' not in difficulty_stats:
        print(f"Breakdown by Difficulty:")
        for difficulty in sorted(difficulty_stats.keys()):
            stats = difficulty_stats[difficulty]
            acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"  {difficulty.capitalize()}: {stats['correct']}/{stats['total']} ({acc:.2f}%)")
        print()

    # Show sample errors
    if errors:
        print(f"Sample Errors (showing first {len(errors)}):")
        print("-" * 80)
        for i, err in enumerate(errors[:5], 1):
            print(f"\n{i}. Question: {err['question']}")
            print(f"   Gold SQL: {err['gold']}")
            print(f"   Pred SQL: {err['pred']}")
        print()

    print(f"{'='*80}\n")

    return {
        'total': total,
        'exact_match': exact_match,
        'accuracy': accuracy,
        'difficulty_stats': dict(difficulty_stats)
    }

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python simple_evaluate.py <predictions_file>")
        print("Example: python simple_evaluate.py ./outputs/spider/output_wo_icl_v1_ollama.jsonl")
        sys.exit(1)

    predictions_file = sys.argv[1]
    results = evaluate_predictions(predictions_file)
