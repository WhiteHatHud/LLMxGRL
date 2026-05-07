"""
visualize_results.py
Generates a 6-panel results figure from the LLMxGRL experiment JSON files.
Output: results_visualization.png
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── colour palette ──────────────────────────────────────────────────────────
BLUE   = "#4C8EDA"
GREEN  = "#2ECC71"
RED    = "#E74C3C"
ORANGE = "#F39C12"
PURPLE = "#9B59B6"
GREY   = "#95A5A6"
DARK   = "#2C3E50"
BG     = "#F8F9FA"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": BG,
    "figure.facecolor": "white",
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ── load data ────────────────────────────────────────────────────────────────
with open("hitl_results_1034.json") as f:
    hitl = json.load(f)

with open("graph_hitl_framework/ablation_results_1034.json") as f:
    ablation = json.load(f)

with open("graph_hitl_results_graph_sos.json") as f:
    graph_sos = json.load(f)

with open("baseline_tabfact_results.json") as f:
    base_tf = json.load(f)

with open("hitl_tabfact_results.json") as f:
    hitl_tf = json.load(f)

with open("baseline_wtq_results.json") as f:
    base_wtq = json.load(f)

with open("hitl_wtq_results.json") as f:
    hitl_wtq = json.load(f)

# ── derived numbers ───────────────────────────────────────────────────────────
hitl_s   = hitl["summary"]
abl_cs   = ablation["condition_summaries"]
gsos_s   = graph_sos["summary"]
btf_s    = base_tf["summary"]
htf_s    = hitl_tf["summary"]
bwtq_s   = base_wtq["summary"]
hwtq_s   = hitl_wtq["summary"]

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.suptitle(
    "LLMxGRL Experiment Results  ·  Qwen 2.5 Coder 14B  ·  Spider / TabFact / WTQ",
    fontsize=15, fontweight="bold", color=DARK, y=0.98,
)

gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38,
                      left=0.07, right=0.97, top=0.91, bottom=0.07)

ax1 = fig.add_subplot(gs[0, 0])   # Spider overall accuracy
ax2 = fig.add_subplot(gs[0, 1])   # Iteration waterfall
ax3 = fig.add_subplot(gs[0, 2])   # Ablation recovery
ax4 = fig.add_subplot(gs[1, 0])   # Multi-hop: 0% → Graph SOS
ax5 = fig.add_subplot(gs[1, 1])   # TabFact baseline vs HITL
ax6 = fig.add_subplot(gs[1, 2])   # WTQ baseline vs HITL

# ── helper ────────────────────────────────────────────────────────────────────
def bar_label(ax, rects, fmt="{:.1f}%", pad=3):
    for r in rects:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h + pad,
                fmt.format(h), ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=DARK)


# ══════════════════════════════════════════════════════════════════════════════
# 1 ·  Spider Phase 1 — overall pass/fail with accuracy callout
# ══════════════════════════════════════════════════════════════════════════════
labels = ["Passed", "Failed"]
sizes  = [hitl_s["passed"], hitl_s["failed"]]
colors = [GREEN, RED]
wedges, texts, autotexts = ax1.pie(
    sizes, labels=labels, colors=colors,
    autopct="%1.1f%%", startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2),
    textprops=dict(fontsize=10),
)
for at in autotexts:
    at.set_fontweight("bold")
    at.set_fontsize(11)
ax1.set_title(f"Spider — Phase 1 HITL\n({hitl_s['total']:,} samples)")
ax1.text(0, -1.45, f"Accuracy: {hitl_s['accuracy']*100:.2f}%   Avg iters: {hitl_s['avg_iterations']}",
         ha="center", fontsize=9, color=DARK,
         bbox=dict(boxstyle="round,pad=0.3", fc="#DFF0D8", ec=GREEN, lw=1.2))


# ══════════════════════════════════════════════════════════════════════════════
# 2 ·  Iteration waterfall — how many fixed at each pass
# ══════════════════════════════════════════════════════════════════════════════
iter_keys = ["1", "2", "3"]
iter_vals = [hitl_s["fixed_at_iteration"].get(k, 0) for k in iter_keys]
icolors   = [GREEN, BLUE, ORANGE]
x = np.arange(len(iter_keys))
rects = ax2.bar(x, iter_vals, color=icolors, width=0.5, edgecolor="white", linewidth=1.5)
ax2.set_xticks(x)
ax2.set_xticklabels(["Iteration 1\n(first try)", "Iteration 2\n(1 correction)", "Iteration 3\n(2 corrections)"])
ax2.set_ylabel("Queries Solved")
ax2.set_title("Spider — Queries Solved per Iteration")
ax2.set_ylim(0, max(iter_vals) * 1.18)
bar_label(ax2, rects, fmt="{:.0f}", pad=6)
# percentage of total
for rect, v in zip(rects, iter_vals):
    pct = v / hitl_s["total"] * 100
    ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height() / 2,
             f"{pct:.1f}%", ha="center", va="center",
             fontsize=9, color="white", fontweight="bold")


# ══════════════════════════════════════════════════════════════════════════════
# 3 ·  Ablation recovery rates on 113 failing samples
# ══════════════════════════════════════════════════════════════════════════════
conds  = ["B\n+Graph\nFeedback", "C\n+GRL\nRouter", "D\nFull\nGraph-HITL"]
rates  = [abl_cs["B"]["recovery_rate"]*100,
          abl_cs["C"]["recovery_rate"]*100,
          abl_cs["D"]["recovery_rate"]*100]
acolors = [BLUE, PURPLE, ORANGE]
x = np.arange(len(conds))
rects = ax3.bar(x, rates, color=acolors, width=0.5, edgecolor="white", linewidth=1.5)
ax3.set_xticks(x)
ax3.set_xticklabels(conds, fontsize=9)
ax3.set_ylabel("Recovery Rate (%)")
ax3.set_title("Ablation — Recovery on 113 Failing Samples")
ax3.set_ylim(0, max(rates) * 1.22)
bar_label(ax3, rects, fmt="{:.1f}%", pad=2)
# annotate recovered count
for rect, s_key in zip(rects, ["B", "C", "D"]):
    n = abl_cs[s_key]["passed"]
    ax3.text(rect.get_x() + rect.get_width() / 2, rect.get_height() / 2,
             f"{n}/113", ha="center", va="center",
             fontsize=9, color="white", fontweight="bold")
# best label
ax3.annotate("Best\ncondition", xy=(0, rates[0]), xytext=(0.6, rates[0] + 4),
             fontsize=8, color=BLUE,
             arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))


# ══════════════════════════════════════════════════════════════════════════════
# 4 ·  Multi-hop: baseline 0% → Graph SOS 7.89% on 38 hard samples
# ══════════════════════════════════════════════════════════════════════════════
labels4 = ["Baseline\n(flat schema)", "Graph SOS\n(Node2vec paths)"]
vals4   = [0.0, gsos_s["accuracy"] * 100]
cols4   = [RED, GREEN]
x = np.arange(2)
rects = ax4.bar(x, vals4, color=cols4, width=0.45, edgecolor="white", linewidth=1.5)
ax4.set_xticks(x)
ax4.set_xticklabels(labels4)
ax4.set_ylabel("Recovery Rate (%)")
ax4.set_title(f"Multi-hop Subset ({gsos_s['total_retested']} samples)\nBaseline = 0%  →  Graph-Augmented")
ax4.set_ylim(0, 18)
for rect, v in zip(rects, vals4):
    ax4.text(rect.get_x() + rect.get_width() / 2,
             max(v + 0.5, 0.8),
             f"{v:.2f}%", ha="center", va="bottom",
             fontsize=11, fontweight="bold", color=DARK)
# delta arrow
ax4.annotate("", xy=(1, vals4[1]), xytext=(0, vals4[0] + 0.3),
             arrowprops=dict(arrowstyle="->", color=DARK, lw=1.5,
                             connectionstyle="arc3,rad=-0.25"))
ax4.text(0.5, 6, f"+{vals4[1]:.2f} pp\n({gsos_s['passed']}/{gsos_s['total_retested']} recovered)",
         ha="center", fontsize=9, color=DARK,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, lw=1))


# ══════════════════════════════════════════════════════════════════════════════
# 5 ·  TabFact — baseline vs HITL
# ══════════════════════════════════════════════════════════════════════════════
def grouped_bench(ax, base_acc, hitl_acc, n_samples, title, delta_color=GREEN):
    vals   = [base_acc * 100, hitl_acc * 100]
    cols   = [GREY, GREEN]
    labels = ["Baseline\n(single pass)", "HITL\n(iterative)"]
    x = np.arange(2)
    rects = ax.bar(x, vals, color=cols, width=0.45, edgecolor="white", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{title}\n({n_samples} samples)")
    ax.set_ylim(0, 115)
    bar_label(ax, rects, fmt="{:.1f}%", pad=2)
    delta = (hitl_acc - base_acc) * 100
    # bracket + delta label
    y_bracket = max(vals) + 8
    ax.annotate("", xy=(1, y_bracket), xytext=(0, y_bracket),
                arrowprops=dict(arrowstyle="<->", color=delta_color, lw=1.8))
    ax.text(0.5, y_bracket + 2, f"+{delta:.1f} pp",
            ha="center", fontsize=11, fontweight="bold", color=delta_color)

grouped_bench(ax5,
              btf_s["accuracy"],
              htf_s["hitl_accuracy"],
              btf_s["total"],
              "TabFact — Binary Fact Verification")

# ══════════════════════════════════════════════════════════════════════════════
# 6 ·  WikiTableQuestions — baseline vs HITL
# ══════════════════════════════════════════════════════════════════════════════
grouped_bench(ax6,
              bwtq_s["accuracy"],
              hwtq_s["hitl_accuracy"],
              bwtq_s["total"],
              "WikiTableQuestions — Open-ended Reasoning",
              delta_color=BLUE)

# ── save ──────────────────────────────────────────────────────────────────────
out = "results_visualization.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
