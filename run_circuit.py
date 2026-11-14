#!/usr/bin/env python3
"""Evolve a resistor network to a target resistance, with random or RL-guided crossover.

Writes the best network's diagram to topology.pdf.

Usage:
    python run_circuit.py                    # Random crossover
    python run_circuit.py --learned          # RL-guided crossover
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

from circuits import TARGET_RESISTANCE, evolve, prune_unused, solve_dc


# ============================================================================
# Topology figure
# ============================================================================

def plot_topology(top, filename: str = "topology.pdf", target: float = TARGET_RESISTANCE):
    """Draw resistor network as a dark blueprint graph with glowing edges and a clean side panel."""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib import font_manager
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    top = prune_unused(top)  # Ensure only nodes on V+→GND paths are drawn

    # ── Palette ───────────────────────────────────────────────────────────
    BG_DARK  = "#FFFFFF"
    BG_PANEL = "#F8F9FA"
    GRID_COL = "#E5E9EF"
    ACCENT   = "#1A6EBD"
    GOLD     = "#C8850A"
    TEXT_LT  = "#1A1A2E"
    TEXT_DIM = "#6B7280"
    RESISTOR_COLORS = ["#1A6EBD", "#D94F3D", "#2E8B57", "#C8850A", "#7B3FA0", "#2976A8"]
    NODE_COLORS = {0: "#D94F3D", 1: "#374151"}  # V+, GND
    tnr_candidates = [
        "/usr/share/fonts/TTF/Times.TTF",
        "/usr/share/fonts/TTF/Timesi.TTF",
        "/usr/local/share/fonts/Times New Roman.ttf",
        os.path.expanduser("~/.local/share/fonts/Times New Roman.ttf"),
    ]
    tnr_path = next((p for p in tnr_candidates if os.path.exists(p)), None)
    if tnr_path:
        font_prop = font_manager.FontProperties(fname=tnr_path)
    else:
        font_prop = font_manager.FontProperties(family="serif")
    font_scale = 14.0 / 10.0
    fs = lambda size: size * font_scale

    # ── Figure & axes ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5), facecolor="white")
    gs  = fig.add_gridspec(1, 2, width_ratios=[2.8, 1], wspace=0.04,
                           left=0.02, right=0.98, top=0.93, bottom=0.0)
    ax       = fig.add_subplot(gs[0])
    ax_panel = fig.add_subplot(gs[1])

    ax.set_facecolor(BG_DARK)
    ax.axis('off')
    ax_panel.set_facecolor(BG_PANEL)
    ax_panel.axis('off')
    for spine in ax_panel.spines.values():
        spine.set_color(GRID_COL); spine.set_linewidth(1.5)

    XLIM, YLIM = (-1.7, 1.7), (-0.6, 1.5)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect('equal')

    # Grid intentionally removed for publication-style clean layout.

    # ── Node positions ────────────────────────────────────────────────────
    n   = top.num_nodes
    pos = {0: (-1.15, 0.0), 1: (1.15, 0.0)}
    for i in range(2, n):
        angle  = np.pi * (0.2 + 0.6 * (i - 2) / max(1, n - 3))
        pos[i] = (np.cos(angle) * 0.85, np.sin(angle) * 0.85 + 0.2)

    # ── Resistor edges ────────────────────────────────────────────────────
    edge_counts, edge_idx = {}, {}
    for r in top.resistors:
        key = (min(r.n1, r.n2), max(r.n1, r.n2))
        edge_counts[key] = edge_counts.get(key, 0) + 1

    label_positions = []
    for i, r in enumerate(top.resistors):
        key   = (min(r.n1, r.n2), max(r.n1, r.n2))
        idx   = edge_idx.get(key, 0); edge_idx[key] = idx + 1
        x1, y1 = pos[r.n1]; x2, y2 = pos[r.n2]
        color = RESISTOR_COLORS[i % len(RESISTOR_COLORS)]
        rad   = 0.6 * (idx - (edge_counts[key] - 1) / 2) if edge_counts[key] > 1 else 0.0

        # Clean, rounded resistor segment (single stroke)
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            color=color, lw=12.0, alpha=1.0, zorder=2, arrowstyle="-",
            capstyle="round", joinstyle="round"
        ))

        norm = max(0.01, np.hypot(x2 - x1, y2 - y1))
        mx   = (x1 + x2) / 2 + rad * 0.35 * (y2 - y1) / norm
        my   = (y1 + y2) / 2 - rad * 0.35 * (x2 - x1) / norm
        label_positions.append((mx, my, f"R{i}", color))

    for mx, my, lbl, col in label_positions:
        ax.text(mx, my, lbl, fontsize=fs(10.0), ha='center', va='center',
            color=col, fontweight='bold', zorder=15, fontproperties=font_prop,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor=col, alpha=0.92, lw=1.2))

    # ── Nodes ─────────────────────────────────────────────────────────────
    for node, (x, y) in pos.items():
        color = NODE_COLORS.get(node, "#000000")
        label = {0: "V+", 1: "GND"}.get(node, f"N{node}")

        # Solid endpoint marker
        ax.scatter(x, y, s=260, c=color, alpha=1.0, zorder=9,
                   linewidths=1.5, edgecolors="white")

        ox = -0.18 if node == 0 else (0.18 if node == 1 else 0)
        oy = 0 if node in (0, 1) else 0.20
        ax.text(x + ox, y + oy, label, fontsize=fs(11), fontweight='bold',
                ha='right' if node == 0 else ('left' if node == 1 else 'center'),
                va='center' if node in (0, 1) else 'bottom',
                color=color, fontproperties=font_prop,
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    row_h = min(0.10, 0.68 / max(1, len(top.resistors)))
    list_h = row_h * max(1, len(top.resistors))
    y_pos = min(0.92, 0.5 + list_h / 2)
    for i, r in enumerate(top.resistors):
        color = RESISTOR_COLORS[i % len(RESISTOR_COLORS)]
        ty    = y_pos - 0.01
        ax_panel.add_patch(FancyBboxPatch(
            (0.04, ty - 0.018), 0.10, 0.036, boxstyle="round,pad=0.005",
            facecolor=color, edgecolor='none',
            transform=ax_panel.transAxes, zorder=5))
        ax_panel.text(0.20, ty + 0.010, f"R{i}  {int(r.value):,}\u03a9",
                  fontsize=fs(10), va='center', ha='left', color=TEXT_LT,
                      fontproperties=font_prop, fontweight='bold',
                      transform=ax_panel.transAxes)
        ax_panel.text(0.20, ty - 0.018, f"node {r.n1} \u2192 {r.n2}",
                  fontsize=fs(7.5), va='center', ha='left', color=TEXT_DIM,
                      fontproperties=font_prop, transform=ax_panel.transAxes)
        y_pos -= row_h

    # ── Resistance summary ────────────────────────────────────────────────
    r_eq = solve_dc(top)
    summary_top = max(0.10, y_pos - 0.03)
    ax_panel.axhline(summary_top + 0.04, color=GRID_COL, lw=0.8, xmin=0.05, xmax=0.95)
    ax_panel.text(0.08, summary_top, "Target",  fontsize=fs(8),  va='center', color=TEXT_DIM,
                  fontproperties=font_prop, transform=ax_panel.transAxes)
    ax_panel.text(0.92, summary_top, f"{target:.1f} \u03a9", fontsize=fs(9), fontweight='bold',
                  va='center', ha='right', color=ACCENT,
                  fontproperties=font_prop, transform=ax_panel.transAxes)
    r_eq_str = f"{r_eq:.1f} \u03a9" if r_eq is not None else "N/A"
    r_eq_col  = GOLD if r_eq is not None else "#D94F3D"
    ax_panel.text(0.08, summary_top - 0.06, "Evolved", fontsize=fs(8),  va='center', color=TEXT_DIM,
                  fontproperties=font_prop, transform=ax_panel.transAxes)
    ax_panel.text(0.92, summary_top - 0.06, r_eq_str, fontsize=fs(9), fontweight='bold',
                  va='center', ha='right', color=r_eq_col,
                  fontproperties=font_prop, transform=ax_panel.transAxes)

    plt.savefig(filename, format='pdf', bbox_inches='tight',
                facecolor='white', dpi=180)
    plt.close()
    print(f"Saved: {filename}")


# ============================================================================
# Command line
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve resistor network")
    parser.add_argument("--learned", action="store_true", help="Use RL-guided crossover")
    parser.add_argument("--target", type=float, default=53.9, help="Target resistance (Ω)")
    parser.add_argument("--gens", type=int, default=20)
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mode = "learned" if args.learned else "random"
    print(f"Evolving resistor network ({mode} crossover, target={args.target}Ω)...")

    best, score, front = evolve(target=args.target, gens=args.gens, pop_size=args.pop,
                                learned=args.learned, seed=args.seed, verbose=True)

    r_eq = solve_dc(best)
    print(f"\nBest: R_eq = {r_eq:.2f}Ω (target: {args.target}Ω)")
    print(f"Topology: {best}")

    print(f"\nPareto front ({len(front)} solutions):")
    for i, t in enumerate(sorted(front, key=lambda x: len(x.resistors))):
        req = solve_dc(t)
        err = abs(req - args.target) / args.target if req else float('inf')
        print(f"  {i+1}. R={len(t.resistors)}, R_eq={req:.1f}Ω, error={err:.2%}")

    plot_topology(best, "topology.pdf", target=args.target)


if __name__ == "__main__":
    main()
