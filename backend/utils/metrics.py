# utils/metrics.py
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from typing import List, Optional

def calculate_accuracy(predictions: List, labels: List) -> float:
    if not predictions or not labels:
        return 0.0
    return accuracy_score(labels, predictions)

def calculate_exact_match(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    return pred.strip().lower() == gold.strip().lower()

def calculate_macro_f1(predictions: List, labels: List) -> float:
    if not predictions or not labels:
        return 0.0
    return f1_score(labels, predictions, average='macro')

def calculate_hits_at_k(retrieved_nodes: List[str], relevant_nodes: List[str], k: int) -> float:
    if not relevant_nodes:
        return 0.0
    top_k = retrieved_nodes[:k]
    hits = len(set(top_k) & set(relevant_nodes))
    return hits / len(relevant_nodes)

def calculate_rouge(pred: str, gold: str) -> float:
    """Stub for ROUGE score"""
    # Placeholder: simple word overlap
    if not pred or not gold:
        return 0.0
    pred_words = set(pred.lower().split())
    gold_words = set(gold.lower().split())
    if not gold_words:
        return 0.0
    overlap = len(pred_words & gold_words)
    return overlap / len(gold_words)

def calculate_bertscore(pred: str, gold: str) -> float:
    """Stub for BERTScore"""
    # Placeholder: return NA
    return None# Accuracy, F1, ROUGE/BERTScore stubs
