#!/usr/bin/env python3
"""
confusion_matrix_router.py
==========================
Generates a Confusion Matrix for the GRL Complexity Router
across the 1034 HITL results.

Axes:
  - Ground Truth: join-count complexity derived from gold_sql
  - Predicted:    GRL Router prediction (logistic regression)

Classes: Simple (0-1 joins), Medium (2 joins), Complex (3+ joins)
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "graph_hitl_framework"))
sys.path.insert(0, str(ROOT / "StructGPT-Ollama"))

from schema_graph import count_joins_in_sql
from grl_router import GRLRouter, join_count_to_label, LABEL_TO_ROUTE

# ── Load data ──────────────────────────────────────────────────────────────────

results_path    = ROOT / "hitl_results_1034.json"
model_path      = ROOT / "graph_hitl_framework/cache/grl_router_model_1034.pkl"
embeddings_path = ROOT / "graph_hitl_framework/cache/schema_embeddings.pkl"

with open(results_path) as f:
    data = json.load(f)

results = data["results"]
print(f"Loaded {len(results)} results.")

# ── Load router ────────────────────────────────────────────────────────────────

print("Loading GRL Router …")
router = GRLRouter.load(
    model_path=str(model_path),
    embeddings_path=str(embeddings_path),
    use_encoder=False,   # text-only features; keeps runtime fast
)
print("Router ready.")

# ── Compute ground truth & predictions ────────────────────────────────────────

y_true, y_pred, mismatches = [], [], []

for r in results:
    question = r.get("question", "")
    db_id    = r.get("db_id", "")
    gold_sql = r.get("gold_sql", "")

    if not question or not db_id or not gold_sql:
        continue

    # Ground truth from join count in gold SQL
    n_joins  = count_joins_in_sql(gold_sql)
    true_lbl = join_count_to_label(n_joins)

    # Router prediction
    route_out = router.route(question=question, db_id=db_id)
    pred_lbl  = {"simple": 0, "medium": 1, "complex": 2}[route_out["route"]]

    y_true.append(true_lbl)
    y_pred.append(pred_lbl)

    if true_lbl != pred_lbl:
        mismatches.append({
            "id":        r.get("id"),
            "question":  question,
            "gold_sql":  gold_sql,
            "n_joins":   n_joins,
            "true":      LABEL_TO_ROUTE[true_lbl],
            "predicted": LABEL_TO_ROUTE[pred_lbl],
            "passed":    r.get("passed"),
            "iterations": r.get("iterations"),
        })

y_true = np.array(y_true)
y_pred = np.array(y_pred)

labels     = [0, 1, 2]
label_names = ["Simple\n(0-1 joins)", "Medium\n(2 joins)", "Complex\n(3+ joins)"]
short_names = ["Simple", "Medium", "Complex"]

print(f"\nTotal samples: {len(y_true)}")
print(f"Misclassified: {len(mismatches)}")

# ── Print classification report ───────────────────────────────────────────────

print("\n── GRL Router Classification Report (full 1034 set) ──")
print(classification_report(y_true, y_pred, target_names=short_names))

# ── Build confusion matrix ─────────────────────────────────────────────────────

cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# ── Plot ───────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("GRL Complexity Router — Confusion Matrix\n(Spider dev set, n=1,034)",
             fontsize=14, fontweight="bold", y=1.01)

def draw_cm(ax, matrix, title, fmt, cmap, vmin, vmax, annot_fn):
    im = ax.imshow(matrix, interpolation="nearest", cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(title=title,
           xticks=range(3), yticks=range(3),
           xticklabels=label_names, yticklabels=label_names,
           xlabel="Predicted Complexity (GRL Router)",
           ylabel="Ground Truth Complexity (Gold SQL join count)")
    ax.xaxis.set_label_position("bottom")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    thresh = (matrix.max() + matrix.min()) / 2.0
    for i in range(3):
        for j in range(3):
            color = "white" if matrix[i, j] > thresh else "black"
            ax.text(j, i, annot_fn(matrix[i, j]),
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color=color)

    # Highlight diagonal
    for k in range(3):
        rect = plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                              linewidth=2.5, edgecolor="#2ecc71",
                              facecolor="none")
        ax.add_patch(rect)


# Left: raw counts
draw_cm(axes[0], cm,
        title="(a) Raw Counts",
        fmt=None, cmap="Blues",
        vmin=0, vmax=cm.max(),
        annot_fn=lambda v: f"{int(v)}")

# Right: row-normalised (recall per class)
draw_cm(axes[1], cm_norm,
        title="(b) Row-Normalised (Recall per class)",
        fmt=None, cmap="YlOrRd",
        vmin=0, vmax=1,
        annot_fn=lambda v: f"{v:.2f}")

# ── Add per-class accuracy annotation ─────────────────────────────────────────

for ax in axes:
    diag_txt = "  ".join(
        f"{short_names[i]}: {cm_norm[i,i]*100:.1f}%"
        for i in range(3)
    )
    ax.set_xlabel(
        f"Predicted Complexity (GRL Router)\n\nDiagonal recall — {diag_txt}",
        fontsize=9
    )

plt.tight_layout()
out_path = ROOT / "confusion_matrix_router.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out_path}")
plt.show()

# ── Misalignment summary ───────────────────────────────────────────────────────

print("\n── Misalignment analysis ──")
from collections import Counter

direction = Counter(
    f"{m['true']} → {m['predicted']}" for m in mismatches
)
for k, v in direction.most_common():
    pct = v / len(y_true) * 100
    print(f"  {k:30s}  n={v:4d}  ({pct:.1f}%)")

# Pass rate per (true, predicted) cell
print("\n── Pass rate per confusion cell ──")
cell_pass  = np.zeros((3, 3), dtype=float)
cell_total = np.zeros((3, 3), dtype=int)

for r, t, p in zip(results, y_true, y_pred):
    cell_total[t, p] += 1
    if r.get("passed"):
        cell_pass[t, p] += 1

with np.errstate(invalid="ignore"):
    cell_rate = np.where(cell_total > 0, cell_pass / cell_total, np.nan)

print(f"{'':30s}  " + "  ".join(f"{s:>10}" for s in short_names))
for i, row_name in enumerate(short_names):
    vals = "  ".join(
        f"{cell_rate[i,j]*100:9.1f}%" if not np.isnan(cell_rate[i,j]) else f"{'N/A':>10}"
        for j in range(3)
    )
    print(f"  GT {row_name:25s}  {vals}")
