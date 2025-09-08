# utils/charts.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict
import os

def save_chart(data: Dict, chart_type: str, run_id: str) -> str:
    """Generate and save charts without custom colors"""
    output_dir = f"data/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if chart_type == "acc_vs_k":
        k_values = data.get("k_values", [1, 3, 5, 7, 10])
        accuracies = data.get("accuracies", [0.6, 0.7, 0.75, 0.78, 0.8])
        ax.plot(k_values, accuracies, marker='o')
        ax.set_xlabel("K (Top-K Retrieval)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs K")
        ax.grid(True, alpha=0.3)
    
    elif chart_type == "tokens_vs_context":
        context_sizes = data.get("context_sizes", [100, 200, 300, 400, 500])
        token_counts = data.get("token_counts", [150, 280, 420, 560, 700])
        ax.plot(context_sizes, token_counts, marker='s')
        ax.set_xlabel("Context Size (chars)")
        ax.set_ylabel("Token Count")
        ax.set_title("Tokens vs Context Size")
        ax.grid(True, alpha=0.3)
    
    elif chart_type == "f1_vs_shots":
        shots = data.get("shots", [0, 1, 2, 3, 5])
        f1_scores = data.get("f1_scores", [0.5, 0.65, 0.72, 0.76, 0.78])
        ax.plot(shots, f1_scores, marker='^')
        ax.set_xlabel("Number of Shots")
        ax.set_ylabel("F1 Score")
        ax.set_title("F1 Score vs Few-Shot Examples")
        ax.grid(True, alpha=0.3)
    
    filepath = f"{output_dir}/{run_id}_{chart_type}.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=100)
    plt.close()
    
    return filepath# Matplotlib plots
