#!/usr/bin/env python3
"""
Generate 4 PNG visualizations for LLM x GRL research presentation.
Outputs saved to ./assets/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
os.makedirs('assets', exist_ok=True)

# Global figure settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


def generate_slide08_misalignment():
    """
    Split-panel: LLM token space vs Graph topology space
    Shows misalignment between representations
    """
    fig = plt.figure(figsize=(19.2, 10.8), facecolor='white')

    # LEFT PANEL: LLM Embedding Space
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('LLM Representation Space\n(Token/Sequence)', fontsize=24, fontweight='bold', pad=20)

    # Generate 3 clusters of embeddings (simulated 2D projection)
    cluster1 = np.random.randn(30, 2) * 0.3 + np.array([1, 1])
    cluster2 = np.random.randn(30, 2) * 0.3 + np.array([-1, 0])
    cluster3 = np.random.randn(30, 2) * 0.3 + np.array([0, -1.5])

    ax1.scatter(cluster1[:, 0], cluster1[:, 1], s=100, alpha=0.6, c='#3498db', label='Query tokens')
    ax1.scatter(cluster2[:, 0], cluster2[:, 1], s=100, alpha=0.6, c='#e74c3c', label='Table tokens')
    ax1.scatter(cluster3[:, 0], cluster3[:, 1], s=100, alpha=0.6, c='#2ecc71', label='Column tokens')

    ax1.set_xlabel('Embedding Dim 1', fontsize=16)
    ax1.set_ylabel('Embedding Dim 2', fontsize=16)
    ax1.legend(fontsize=14, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)

    # Add annotation
    ax1.text(0, 2.2, 'Semantic clusters\nin continuous space',
             fontsize=14, ha='center', style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # RIGHT PANEL: Graph Topology
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Graph Representation Space\n(Nodes/Edges)', fontsize=24, fontweight='bold', pad=20)

    # Create a sample database schema graph
    G = nx.DiGraph()

    # Add nodes (tables)
    tables = ['singer', 'concert', 'stadium', 'song', 'album']
    G.add_nodes_from(tables)

    # Add edges (foreign keys)
    edges = [
        ('singer', 'concert'),
        ('singer', 'album'),
        ('concert', 'stadium'),
        ('song', 'album'),
        ('album', 'singer')
    ]
    G.add_edges_from(edges)

    # Layout
    pos = nx.spring_layout(G, seed=42, k=1.5)

    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='#95a5a6',
                          alpha=0.9, ax=ax2)
    nx.draw_networkx_edges(G, pos, edge_color='#34495e', width=3,
                          arrowsize=30, arrowstyle='->', ax=ax2,
                          connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', ax=ax2)

    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.axis('off')

    # Add annotation
    ax2.text(0, -1.3, 'Discrete topology\nwith structural constraints',
             fontsize=14, ha='center', style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # CENTER: Big "MISALIGNMENT" text
    fig.text(0.5, 0.5, '≠\nMISALIGNMENT', fontsize=60, ha='center', va='center',
             fontweight='bold', color='#e74c3c', alpha=0.8,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='#e74c3c', linewidth=4))

    # BOTTOM: Bullet points
    bullet_y = 0.08
    bullets = [
        "• LLMs optimize token likelihood, not topology constraints",
        "• GNNs encode neighborhood structure, not global semantics",
        "• Mismatch → hallucinated edges / invalid joins"
    ]

    for i, bullet in enumerate(bullets):
        fig.text(0.5, bullet_y - i*0.04, bullet, fontsize=16, ha='center',
                bbox=dict(boxstyle='round', facecolor='#fff9e6', alpha=0.8))

    plt.tight_layout(rect=[0, 0.15, 1, 0.96])
    output_path = 'assets/slide08_misalignment.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Generated: {output_path}")


def generate_slide09_cross_domain():
    """
    Bar chart showing accuracy drop across domains
    """
    fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor='white')

    # Data
    domains = ['Domain A\n(Seen)', 'Domain B\n(Shifted)', 'Domain C\n(Unseen)']
    accuracies = [0.35, 0.22, 0.15]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    # Create bars
    bars = ax.bar(domains, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.2f}',
                ha='center', va='bottom', fontsize=24, fontweight='bold')

    # Add decline arrows
    ax.annotate('', xy=(1, 0.23), xytext=(0, 0.34),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax.annotate('', xy=(2, 0.16), xytext=(1, 0.21),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))

    # Add decline percentage labels
    ax.text(0.5, 0.29, '-37%', fontsize=18, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))
    ax.text(1.5, 0.19, '-32%', fontsize=18, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))

    # Styling
    ax.set_ylabel('Accuracy', fontsize=22, fontweight='bold')
    ax.set_xlabel('Domain', fontsize=22, fontweight='bold')
    ax.set_title('Cross-Domain Generalization Drop\n(Illustrative Placeholder)',
                 fontsize=28, fontweight='bold', pad=30)
    ax.set_ylim(0, 0.45)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add subtitle
    fig.text(0.5, 0.08, 'Illustrative placeholder — Replace with real evaluation data',
             fontsize=16, ha='center', style='italic', color='#7f8c8d',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.7))

    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    output_path = 'assets/slide09_cross_domain_drop.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Generated: {output_path}")


def generate_slide10_hop_accuracy():
    """
    Line chart showing accuracy degradation with hop count
    """
    fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor='white')

    # Data
    hops = [0, 1, 2, 3, 4]
    accuracies = [0.40, 0.32, 0.25, 0.18, 0.12]

    # Plot line with markers
    ax.plot(hops, accuracies, marker='o', markersize=20, linewidth=4,
            color='#3498db', markerfacecolor='#e74c3c', markeredgewidth=3,
            markeredgecolor='#c0392b')

    # Add value labels on each point
    for x, y in zip(hops, accuracies):
        ax.text(x, y + 0.02, f'{y:.2f}', ha='center', va='bottom',
                fontsize=20, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

    # Add shaded region showing decline
    ax.fill_between(hops, accuracies, alpha=0.2, color='#e74c3c')

    # Styling
    ax.set_xlabel('Graph Traversal Depth (# Hops)', fontsize=22, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=22, fontweight='bold')
    ax.set_title('Performance Degrades with Multi-hop Reasoning\n(Illustrative Placeholder)',
                 fontsize=28, fontweight='bold', pad=30)
    ax.set_xticks(hops)
    ax.set_xticklabels(['0-hop\n(Single Table)', '1-hop\n(2 Tables)', '2-hop\n(3 Tables)',
                        '3-hop\n(4 Tables)', '4-hop\n(5+ Tables)'], fontsize=14)
    ax.set_ylim(0, 0.5)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add trend annotation
    ax.annotate('Accuracy drops\n~20% per hop', xy=(2, 0.25), xytext=(3.2, 0.35),
                fontsize=18, color='#e74c3c', fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=3, color='#e74c3c'),
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#e74c3c', linewidth=2))

    # Add subtitle
    fig.text(0.5, 0.08, 'Illustrative placeholder — Spider dataset stratification planned',
             fontsize=16, ha='center', style='italic', color='#7f8c8d',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.7))

    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    output_path = 'assets/slide10_hop_accuracy_curve.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Generated: {output_path}")


def generate_slide11_graphrag_pipeline():
    """
    Flow diagram showing GraphRAG pipeline
    """
    fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor='white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    fig.text(0.5, 0.95, 'GraphRAG Pipeline for Structured Reasoning',
             fontsize=32, ha='center', fontweight='bold')

    # Box dimensions
    box_width = 1.6
    box_height = 0.8

    # Box positions (x, y)
    boxes = [
        (0.5, 3.0, 'Graph Store /\nSchema Graph', '#3498db'),
        (2.5, 3.0, 'Retriever\n(k-hop subgraph)', '#2ecc71'),
        (4.5, 3.0, 'LLM Agent\n(draft → check)', '#9b59b6'),
        (6.5, 3.0, 'Verifier\n(validate edges)', '#e67e22'),
        (8.5, 3.0, 'Final Answer\n/ SQL', '#e74c3c')
    ]

    # Draw boxes
    for x, y, text, color in boxes:
        box = FancyBboxPatch((x, y), box_width, box_height,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=color,
                            linewidth=3, alpha=0.8)
        ax.add_patch(box)

        # Add text with better formatting
        lines = text.split('\n')
        if len(lines) == 2:
            ax.text(x + box_width/2, y + box_height/2 + 0.15, lines[0],
                   ha='center', va='center', fontsize=16, fontweight='bold', color='white')
            ax.text(x + box_width/2, y + box_height/2 - 0.15, lines[1],
                   ha='center', va='center', fontsize=13, color='white')
        else:
            ax.text(x + box_width/2, y + box_height/2, text,
                   ha='center', va='center', fontsize=16, fontweight='bold',
                   color='white', multialignment='center')

    # Draw arrows between boxes
    arrow_y = 3.4
    for i in range(len(boxes) - 1):
        x_start = boxes[i][0] + box_width + 0.05
        x_end = boxes[i+1][0] - 0.05
        arrow = FancyArrowPatch((x_start, arrow_y), (x_end, arrow_y),
                              arrowstyle='->', mutation_scale=40,
                              linewidth=4, color='#34495e')
        ax.add_patch(arrow)

    # Add detailed labels under each box
    labels = [
        'nodes, edges,\ntext attributes',
        'join paths,\nconstraints',
        'self-check,\nreprompt',
        'validate\njoins exist',
        'structured\noutput'
    ]

    for (x, y, _, _), label in zip(boxes, labels):
        ax.text(x + box_width/2, y - 0.3, label, ha='center', va='top',
               fontsize=12, style='italic', color='#555',
               multialignment='center')

    # Add side bubble for LLM Agent
    bubble_x = 4.5 + box_width/2
    bubble_y = 4.5

    # Draw bubble
    bubble = mpatches.FancyBboxPatch((bubble_x - 0.9, bubble_y - 0.3), 1.8, 0.6,
                                     boxstyle="round,pad=0.1",
                                     edgecolor='#9b59b6', facecolor='#f3e5f5',
                                     linewidth=2, linestyle='--')
    ax.add_patch(bubble)

    # Bubble text
    ax.text(bubble_x, bubble_y, 'Ask clarifying questions\nwhen context is missing',
           ha='center', va='center', fontsize=13, style='italic',
           color='#6a1b9a', multialignment='center')

    # Arrow from bubble to LLM box
    bubble_arrow = FancyArrowPatch((bubble_x, bubble_y - 0.3),
                                  (bubble_x, 3.0 + box_height + 0.05),
                                  arrowstyle='->', mutation_scale=20,
                                  linewidth=2, color='#9b59b6', linestyle='--')
    ax.add_patch(bubble_arrow)

    # Add feedback loop arrow (from Verifier back to Retriever)
    feedback_start_x = 6.5 + box_width/2
    feedback_end_x = 2.5 + box_width/2
    feedback_y_start = 3.0 - 0.1
    feedback_y_mid = 2.0

    # Curved feedback arrow
    from matplotlib.patches import ConnectionPatch
    feedback_arrow = mpatches.FancyBboxPatch((2.3, 1.8), 4.5, 0.4,
                                            boxstyle="round,pad=0.05",
                                            edgecolor='#e67e22', facecolor='none',
                                            linewidth=2, linestyle=':', alpha=0)

    # Draw feedback path with arrows
    ax.annotate('', xy=(feedback_end_x, feedback_y_start),
               xytext=(feedback_start_x, feedback_y_start),
               arrowprops=dict(arrowstyle='<-', lw=3, color='#e67e22',
                             linestyle=':', connectionstyle="arc3,rad=-.5"))

    ax.text(4.5, 2.2, 'Feedback: Re-retrieve if validation fails',
           ha='center', fontsize=12, color='#e67e22', style='italic',
           bbox=dict(boxstyle='round', facecolor='#fff3e0', alpha=0.8))

    plt.tight_layout()
    output_path = 'assets/slide11_graphrag_pipeline.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Generated: {output_path}")


def main():
    """Generate all 4 visualization assets"""
    print("\n" + "="*60)
    print("Generating Research Visualization Assets")
    print("="*60 + "\n")

    print("Creating assets directory...")
    os.makedirs('assets', exist_ok=True)

    print("\nGenerating visualizations...\n")

    generate_slide08_misalignment()
    generate_slide09_cross_domain()
    generate_slide10_hop_accuracy()
    generate_slide11_graphrag_pipeline()

    print("\n" + "="*60)
    print("✓ All assets generated successfully!")
    print("="*60)
    print("\nOutput files:")
    print("  - assets/slide08_misalignment.png")
    print("  - assets/slide09_cross_domain_drop.png")
    print("  - assets/slide10_hop_accuracy_curve.png")
    print("  - assets/slide11_graphrag_pipeline.png")
    print("\nReady to insert into slides 8-11!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
