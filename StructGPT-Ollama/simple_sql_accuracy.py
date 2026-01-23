import json
import sys
import re

def extract_sql_from_response(text):
    """Extract SQL query from response that might contain reasoning/explanation"""
    if not text:
        return ""
    
    # Remove markdown code blocks
    text = re.sub(r'```sql\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Remove <think> tags and their content (DeepSeek R1 reasoning)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Extract SQL query - look for SELECT statements
    # Try to find the main SQL query
    sql_match = re.search(r'(SELECT\s+.+?(?:;|\n\n|$))', text, re.IGNORECASE | re.DOTALL)
    if sql_match:
        text = sql_match.group(1)
    
    return text.strip()

def normalize_sql(sql):
    """Normalize SQL for comparison"""
    # Extract SQL from response
    sql = extract_sql_from_response(sql)
    
    # Convert to lowercase
    sql = sql.lower().strip()
    
    # Remove extra whitespace and standardize spacing
    sql = re.sub(r'\s+', ' ', sql)
    
    # Remove spaces around parentheses and commas for better matching
    sql = re.sub(r'\s*\(\s*', '(', sql)
    sql = re.sub(r'\s*\)\s*', ')', sql)
    sql = re.sub(r'\s*,\s*', ',', sql)
    
    # Remove trailing semicolon
    sql = sql.rstrip(';').strip()
    
    return sql

def evaluate_sql_accuracy(jsonl_path):
    """Calculate exact match and normalized match accuracy"""
    with open(jsonl_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    exact_matches = 0
    normalized_matches = 0
    total = len(data)
    
    mismatches = []
    
    for idx, item in enumerate(data):
        ground_truth = item.get('query', '')
        prediction = item.get('Prediction', '')
        
        # Exact match
        if ground_truth == prediction:
            exact_matches += 1
            normalized_matches += 1
        else:
            # Normalized match
            gt_norm = normalize_sql(ground_truth)
            pred_norm = normalize_sql(prediction)
            if gt_norm == pred_norm:
                normalized_matches += 1
            else:
                # Store first few mismatches for debugging
                if len(mismatches) < 5:
                    mismatches.append({
                        'idx': idx,
                        'ground_truth': gt_norm,
                        'prediction': pred_norm[:200]  # Truncate long predictions
                    })
    
    exact_acc = (exact_matches / total) * 100 if total > 0 else 0
    norm_acc = (normalized_matches / total) * 100 if total > 0 else 0
    
    return {
        'total': total,
        'exact_matches': exact_matches,
        'normalized_matches': normalized_matches,
        'exact_accuracy': exact_acc,
        'normalized_accuracy': norm_acc,
        'mismatches': mismatches
    }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python simple_sql_accuracy.py <jsonl_file>")
        sys.exit(1)
    
    results = evaluate_sql_accuracy(sys.argv[1])
    
    print(f"\n{'='*60}")
    print(f"SQL Accuracy Evaluation")
    print(f"{'='*60}")
    print(f"Total Samples:           {results['total']}")
    print(f"Exact Matches:           {results['exact_matches']}")
    print(f"Normalized Matches:      {results['normalized_matches']}")
    print(f"Exact Match Accuracy:    {results['exact_accuracy']:.2f}%")
    print(f"Normalized Accuracy:     {results['normalized_accuracy']:.2f}%")
    print(f"{'='*60}")
    
    if results['mismatches']:
        print(f"\nSample Mismatches (first {len(results['mismatches'])}):")
        for m in results['mismatches']:
            print(f"\nSample {m['idx']}:")
            print(f"  Ground Truth: {m['ground_truth']}")
            print(f"  Prediction:   {m['prediction']}")
    print()
