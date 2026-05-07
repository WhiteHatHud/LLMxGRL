"""
visualize_architecture.py
=========================
Generates two FYP figures:

  1. fyp_pipeline_diagram.png  — HITL loop + Graph augmentation flow
  2. fyp_code_snippets.png     — 4 annotated code panels (key functions)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE    = "#2980B9"
LBLUE   = "#AED6F1"
GREEN   = "#27AE60"
LGREEN  = "#A9DFBF"
ORANGE  = "#E67E22"
LORANGE = "#FAD7A0"
RED     = "#C0392B"
LRED    = "#F5B7B1"
PURPLE  = "#8E44AD"
LPURPLE = "#D7BDE2"
GREY    = "#BDC3C7"
DARK    = "#2C3E50"
WHITE   = "#FFFFFF"
BG      = "#FDFEFE"

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
})


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Pipeline Diagram
# ══════════════════════════════════════════════════════════════════════════════

def draw_box(ax, x, y, w, h, label, sublabel="",
             fc=LBLUE, ec=BLUE, fontsize=10, radius=0.04):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=2, zorder=3)
    ax.add_patch(box)
    dy = 0.012 if sublabel else 0
    ax.text(x, y + dy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=DARK, zorder=4)
    if sublabel:
        ax.text(x, y - 0.045, sublabel, ha="center", va="center",
                fontsize=7.5, color="#555", zorder=4, style="italic")


def arrow(ax, x1, y1, x2, y2, label="", color=DARK, lw=1.8, style="->",
          lbl_offset=(0, 0.018)):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle="arc3,rad=0.0"),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2 + lbl_offset[0], (y1+y2)/2 + lbl_offset[1]
        ax.text(mx, my, label, ha="center", va="bottom",
                fontsize=7.5, color=color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc=WHITE, ec="none", alpha=0.8))


fig1, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig1.suptitle(
    "Figure 1 — Graph-Augmented HITL Pipeline Architecture",
    fontsize=14, fontweight="bold", color=DARK, y=0.97,
)

# ── Phase labels (background bands) ──────────────────────────────────────────
for (x0, x1, label, fc) in [
    (0.01, 0.30, "PHASE 1 — Create", "#EBF5FB"),
    (0.30, 0.60, "PHASE 2 — Verify", "#EAFAF1"),
    (0.60, 0.99, "PHASE 3 — Refine", "#FEF9E7"),
]:
    ax.add_patch(plt.Rectangle((x0, 0.04), x1-x0, 0.88,
                                fc=fc, ec="#CCC", lw=0.8, zorder=0, alpha=0.6))
    ax.text((x0+x1)/2, 0.90, label, ha="center", va="center",
            fontsize=9, color="#555", style="italic", fontweight="bold")

# ── Component boxes ───────────────────────────────────────────────────────────
W, H = 0.18, 0.09

# 1. User question (left)
draw_box(ax, 0.12, 0.70, W, H,
         "Natural Language\nQuestion", "e.g. 'List all singers\nborn after 1980'",
         fc="#D6EAF8", ec=BLUE)

# 2. Schema + DB
draw_box(ax, 0.12, 0.42, W, H,
         "Spider DB\n+ Schema", "SQLite + tables.json",
         fc="#D6EAF8", ec=BLUE)

# 3. AI Agent (Qwen)
draw_box(ax, 0.44, 0.70, W+0.02, H,
         "AI Agent", "Qwen 2.5 Coder 14B\nChain-of-Thought → SQL",
         fc=LGREEN, ec=GREEN)

# 4. SQL Executor
draw_box(ax, 0.44, 0.42, W, H,
         "SQL Executor", "execute_sql(pred, db)\nvs gold_results",
         fc=LORANGE, ec=ORANGE)

# 5. PASS
draw_box(ax, 0.44, 0.14, 0.14, 0.07,
         "✓  PASS", "Result matches gold",
         fc=LGREEN, ec=GREEN, fontsize=9)

# 6. GRL Error Classifier
draw_box(ax, 0.78, 0.70, W+0.02, H,
         "GRL Error Classifier", "Structural vs Surface\n(FK graph BFS)",
         fc=LPURPLE, ec=PURPLE)

# 7. Graph Oracle Feedback
draw_box(ax, 0.78, 0.42, W+0.02, H,
         "Graph Oracle Feedback", "Injects FK traversal\npath into prompt",
         fc=LPURPLE, ec=PURPLE)

# 8. Human Proxy (LLM Oracle)
draw_box(ax, 0.78, 0.14, W+0.02, H,
         "Human Proxy (Oracle)", "Simulated feedback LLM\n(same Qwen model)",
         fc=LORANGE, ec=ORANGE)

# 9. GRL Router
draw_box(ax, 0.12, 0.14, W, H,
         "GRL Router", "LogReg on 15 features\n→ simple/medium/complex",
         fc="#FDEDEC", ec=RED)

# ── Arrows ────────────────────────────────────────────────────────────────────
# Q → AI Agent
arrow(ax, 0.21, 0.70, 0.34, 0.70, "NL query + schema", color=BLUE)
# Schema → AI Agent (diagonal)
arrow(ax, 0.21, 0.46, 0.34, 0.66, "DB schema", color=BLUE, lbl_offset=(-0.05, 0.01))
# AI Agent → SQL Executor
arrow(ax, 0.44, 0.655, 0.44, 0.465, "predicted SQL", color=GREEN)
# PASS (downward from executor)
arrow(ax, 0.44, 0.375, 0.44, 0.178, "results match", color=GREEN)
# FAIL → Error Classifier
arrow(ax, 0.535, 0.70, 0.685, 0.70, "FAIL: pred+gold SQL", color=RED, lbl_offset=(0, 0.018))
# Error classifier → Graph Oracle
arrow(ax, 0.78, 0.655, 0.78, 0.465, "structural error\n+ FK path", color=PURPLE)
# Graph Oracle → Human Proxy
arrow(ax, 0.78, 0.375, 0.78, 0.185, "augmented\nfeedback", color=PURPLE)
# Human Proxy → AI Agent (feedback loop — curved)
ax.annotate("", xy=(0.44, 0.655), xytext=(0.685, 0.155),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8,
                            connectionstyle="arc3,rad=0.35"), zorder=2)
ax.text(0.53, 0.37, "proxy feedback\n(iterate ≤3×)", ha="center", fontsize=8,
        color=ORANGE,
        bbox=dict(boxstyle="round,pad=0.2", fc=WHITE, ec="none", alpha=0.85))
# Router → AI Agent (budget)
arrow(ax, 0.21, 0.14, 0.34, 0.66, "iteration budget", color=RED,
      lbl_offset=(-0.09, 0.0))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(fc=LGREEN,  ec=GREEN,  label="AI Agent (Qwen)"),
    mpatches.Patch(fc=LORANGE, ec=ORANGE, label="Oracle / Executor"),
    mpatches.Patch(fc=LPURPLE, ec=PURPLE, label="Graph Layer (GRL)"),
    mpatches.Patch(fc="#FDEDEC", ec=RED,  label="GRL Router"),
    mpatches.Patch(fc="#D6EAF8", ec=BLUE, label="Input / Data"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=8.5,
          framealpha=0.9, edgecolor=GREY)

# ── Iteration counter callout ─────────────────────────────────────────────────
ax.text(0.44, 0.58, "max 3 iterations\n(5 for 'complex' route)",
        ha="center", fontsize=8, color="#555",
        bbox=dict(boxstyle="round,pad=0.25", fc="#FDFEFE", ec=GREY, lw=1))

fig1.savefig("fyp_pipeline_diagram.png", dpi=150, bbox_inches="tight")
print("Saved → fyp_pipeline_diagram.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Annotated Code Snippets
# ══════════════════════════════════════════════════════════════════════════════

CODE_STYLE = dict(fontfamily="monospace", fontsize=7.8, va="top",
                  color=DARK, linespacing=1.55)
TITLE_STYLE = dict(fontsize=10, fontweight="bold", color=WHITE,
                   ha="left", va="center")


def code_panel(ax, title, code_lines, header_fc=BLUE, highlight_lines=None):
    """
    Draw a code panel: coloured header + monospace body + optional line highlights.
    highlight_lines: list of (line_index_0based, colour) tuples
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Header bar
    ax.add_patch(FancyBboxPatch((0, 0.88), 1, 0.12,
                                boxstyle="round,pad=0.01",
                                fc=header_fc, ec="none", zorder=2))
    ax.text(0.02, 0.935, title, **TITLE_STYLE, zorder=3)

    # Body background
    ax.add_patch(FancyBboxPatch((0, 0), 1, 0.88,
                                boxstyle="round,pad=0.01",
                                fc="#F8F8F8", ec=GREY, linewidth=1.2, zorder=1))

    # Highlight rows
    if highlight_lines:
        n = len(code_lines)
        row_h = 0.84 / n
        for (li, fc) in highlight_lines:
            y_top = 0.86 - li * row_h
            ax.add_patch(plt.Rectangle((0.005, y_top - row_h + 0.002),
                                        0.99, row_h - 0.002,
                                        fc=fc, ec="none", zorder=2, alpha=0.45))

    # Code text
    body = "\n".join(code_lines)
    ax.text(0.022, 0.855, body, transform=ax.transAxes,
            zorder=4, **CODE_STYLE)


# ── Four snippets ─────────────────────────────────────────────────────────────

snippet_A = [
    "# hitl_sql_testbed.py → run_hitl_loop()",
    "# ─────────────────────────────────────",
    "def run_hitl_loop(entry, agent, proxy):",
    "    agent.reset()                       # fresh conversation context",
    "    message = build_initial_prompt(entry)",
    "",
    "    for iteration in range(MAX_ITERATIONS):  # max 3 turns",
    "        resp    = agent.send(message)         # LLM call → JSON",
    "        sql     = extract_sql(resp['sql'])",
    "        results, err = execute_sql(sql, db_path)",
    "",
    "        if results_match(results, gold_results):",
    "            return record(passed=True, iters=iteration+1)",
    "",
    "        # Oracle sees wrong output → writes targeted feedback",
    "        feedback = proxy.respond(results, entry)",
    "        message  = feedback              # next iteration input",
    "",
    "    return record(passed=False, iters=MAX_ITERATIONS)",
]

snippet_B = [
    "# graph_oracle_feedback.py → generate()",
    "# ──────────────────────────────────────",
    "def generate(self, proxy_text, pred_sql,",
    "             gold_sql, db_id, use_graph):",
    "",
    "    classification = self.clf.classify(",
    "        pred_sql, gold_sql, db_id)      # BFS on FK graph",
    "    error_type = classification['error_type']",
    "",
    "    if error_type == 'structural':      # wrong JOIN path",
    "        fk_path = classification['fk_path']",
    "        if fk_path:",
    "            # Append the FK traversal road-map to LLM feedback",
    "            return proxy_text + GRAPH_PATH_SUFFIX.format(",
    "                       fk_path=fk_path)",
    "",
    "    return proxy_text   # surface error — plain feedback sufficient",
    "# Example injected suffix:",
    "# 'The schema graph shows the required path is:'",
    "# 'CAR_MAKERS -[Id=Maker]-> MODEL_LIST -[Model=Model]-> CARS_DATA'",
]

snippet_C = [
    "# grl_router.py → extract_features() (15-dim vector)",
    "# ─────────────────────────────────────────────────────",
    "def extract_features(question, db_id, cache, encoder):",
    "    # Group 1 — Text features (6 dims)",
    "    text  = [has_aggregation, has_multi_table,",
    "             word_count/30, char_count/200,",
    "             clause_density, wh_word_count/3]",
    "",
    "    # Group 2 — Embedding similarity (4 dims)",
    "    q_emb = encoder.encode(question)    # all-MiniLM-L6-v2",
    "    sims  = cosine_sim(q_emb, table_embs[db_id])",
    "    sim   = [max(sims), mean(sims), std(sims), min(sims)]",
    "",
    "    # Group 3 — Schema graph statistics (5 dims)",
    "    graph = [num_tables/20, num_edges/30, avg_degree/5,",
    "             max_degree/10, diameter≥3 ? 1.0 : 0.0]",
    "",
    "    return concat([text, sim, graph])   # → (15,) float32",
    "",
    "# Trained on 200 Spider samples → LogisticRegression",
    "# Predicts: 'simple' | 'medium' | 'complex' (join depth)",
]

snippet_D = [
    "# graph_hitl_pipeline.py → run_graph_hitl_loop()",
    "# ─────────────────────────────────────────────────",
    "def run_graph_hitl_loop(entry, agent, proxy,",
    "        graph_proxy, router, use_graph, use_router):",
    "",
    "    # Step 1 — Route: predict join complexity BEFORE inference",
    "    route = router.route(question, db_id)  # simple/medium/complex",
    "    max_iter = {simple:1, medium:3, complex:5}[route]",
    "",
    "    for iteration in range(max_iter):",
    "        resp = agent.send(current_message)",
    "        sql  = extract_sql(resp['sql'])",
    "        results, err = execute_sql(sql, db_path)",
    "",
    "        if results_match(results, gold):    # ← PASS",
    "            break",
    "",
    "        # Step 2 — Classify error + augment feedback",
    "        if use_graph:",
    "            feedback, clf = graph_proxy.respond(",
    "                msg, entry, pred_sql=sql, gold_sql=gold_sql)",
    "            # clf = {error_type, fk_path, missing_tables}",
    "        else:",
    "            feedback = proxy.respond(msg, entry)",
    "",
    "        current_message = feedback",
]

# highlights: line indices (0-based) + colour
hl_A = [(6, "#D5F5E3"), (11, "#D5F5E3"), (14, "#FCF3CF")]
hl_B = [(9,  "#E8DAEF"), (10, "#E8DAEF"), (11, "#E8DAEF"),
        (13, "#D5F5E3"), (15, "#FEF9E7")]
hl_C = [(3,  "#D6EAF8"), (4, "#D6EAF8"), (5, "#D6EAF8"),
        (8,  "#D5F5E3"), (9, "#D5F5E3"),
        (12, "#FDEDEC"), (13, "#FDEDEC")]
hl_D = [(5, "#EBF5FB"), (6, "#EBF5FB"),
        (16, "#E8DAEF"), (17, "#E8DAEF"), (18, "#E8DAEF")]

fig2, axes = plt.subplots(2, 2, figsize=(18, 14))
fig2.suptitle(
    "Figure 2 — Key Code Snippets  ·  LLMxGRL / Qwen 2.5 Coder 14B",
    fontsize=14, fontweight="bold", color=DARK, y=0.98,
)
fig2.patch.set_facecolor(BG)

panels = [
    (axes[0,0], "A  —  Core HITL Loop  (hitl_sql_testbed.py)",     snippet_A, BLUE,   hl_A),
    (axes[0,1], "B  —  Graph Oracle Feedback  (graph_oracle_feedback.py)", snippet_B, PURPLE, hl_B),
    (axes[1,0], "C  —  GRL Feature Extraction  (grl_router.py)",   snippet_C, GREEN,  hl_C),
    (axes[1,1], "D  —  Graph-Guided Loop  (graph_hitl_pipeline.py)", snippet_D, ORANGE, hl_D),
]

for ax, title, code, hdr, hl in panels:
    code_panel(ax, title, code, header_fc=hdr, highlight_lines=hl)

# shared legend for highlights
from matplotlib.lines import Line2D
legend_elements = [
    mpatches.Patch(fc="#D5F5E3", ec=GREEN,  label="PASS / key outcome"),
    mpatches.Patch(fc="#E8DAEF", ec=PURPLE, label="Graph augmentation step"),
    mpatches.Patch(fc="#FCF3CF", ec=ORANGE, label="Feedback / refine step"),
    mpatches.Patch(fc="#D6EAF8", ec=BLUE,   label="Text feature group"),
    mpatches.Patch(fc="#FDEDEC", ec=RED,    label="Graph feature group"),
]
fig2.legend(handles=legend_elements, loc="lower center", ncol=5,
            fontsize=8.5, framealpha=0.9, edgecolor=GREY,
            bbox_to_anchor=(0.5, 0.005))

plt.subplots_adjust(hspace=0.06, wspace=0.04,
                    left=0.01, right=0.99, top=0.94, bottom=0.06)
fig2.savefig("fyp_code_snippets.png", dpi=150, bbox_inches="tight")
print("Saved → fyp_code_snippets.png")
