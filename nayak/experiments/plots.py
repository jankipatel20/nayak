"""Publication-grade visualization suite generating benchmark figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nayak.controllers.fixed_time import FixedTimeController
from nayak.controllers.fuzzy_controller import EvolvedFuzzyController
from nayak.evolve.genome import decode_genome, encode_default_hand_designed
from nayak.fuzzy.rulebase import create_default_input_variables, create_default_output_variable
from nayak.sim.arrivals import ArrivalGenerator
from nayak.sim.intersection import Intersection
from nayak.config import SCENARIOS, DEFAULT_SIM_CONFIG

# Global plotting style configuration
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def plot_hypervolume_convergence(
    hv_history_path: Path | str,
    output_path: Path | str,
) -> None:
    """Plot hypervolume indicator growth across NSGA-II generations."""
    with open(hv_history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    gens = [entry["gen"] for entry in history]
    hvs = [entry["hypervolume"] for entry in history]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(gens, hvs, marker="o", color="#008080", linewidth=2.2, markersize=5, label="NSGA-II Hypervolume")
    ax.set_title("Hypervolume Indicator Convergence (Reference: [120, 250, 35])")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_pareto_front(
    pareto_npz_path: Path | str,
    reps_dir: Path | str,
    output_path: Path | str,
) -> None:
    """Plot 3 pairwise 2D projections and a 3D view of the Pareto front with picks annotated."""
    data = np.load(pareto_npz_path)
    F = data["F"]  # (N, 3): [f1=mean_wait, f2=p95_wait, f3=ambulance_delay]

    rep_names = ["Nayak-Fast", "Nayak-Fair", "Nayak-Rescue", "Nayak-Balanced"]
    rep_points = {}
    for name in rep_names:
        p = Path(reps_dir) / f"{name}.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
                rep_points[name] = [
                    d["objectives"]["mean_wait"],
                    d["objectives"]["wait_p95"],
                    d["objectives"]["ambulance_delay"],
                ]

    rep_colors = {
        "Nayak-Fast": "#d95f02",
        "Nayak-Fair": "#7570b3",
        "Nayak-Rescue": "#e7298a",
        "Nayak-Balanced": "#1b9e77",
    }

    fig = plt.figure(figsize=(14, 11), dpi=300)

    # 1. f1 vs f2 (Mean Wait vs p95 Wait)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(F[:, 0], F[:, 1], color="#2b83ba", alpha=0.7, s=40, label="Pareto Optimal Solutions")
    for name, pt in rep_points.items():
        ax1.scatter(pt[0], pt[1], color=rep_colors[name], s=120, edgecolors="black", linewidth=1.5, zorder=5)
        ax1.annotate(name, (pt[0], pt[1]), textcoords="offset points", xytext=(6, 6), fontweight="bold")
    ax1.set_xlabel("f1: Mean Wait Time (s)")
    ax1.set_ylabel("f2: 95th Percentile Wait Time (s)")
    ax1.set_title("Trade-off: Average Delay vs Fairness")

    # 2. f1 vs f3 (Mean Wait vs Ambulance Delay)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(F[:, 0], F[:, 3 - 1], color="#2b83ba", alpha=0.7, s=40)
    for name, pt in rep_points.items():
        ax2.scatter(pt[0], pt[2], color=rep_colors[name], s=120, edgecolors="black", linewidth=1.5, zorder=5)
        ax2.annotate(name, (pt[0], pt[2]), textcoords="offset points", xytext=(6, 6), fontweight="bold")
    ax2.set_xlabel("f1: Mean Wait Time (s)")
    ax2.set_ylabel("f3: Mean Ambulance Delay (s)")
    ax2.set_title("Trade-off: Average Delay vs Emergency Clearance")

    # 3. f2 vs f3 (Fairness vs Ambulance Delay)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(F[:, 1], F[:, 2], color="#2b83ba", alpha=0.7, s=40)
    for name, pt in rep_points.items():
        ax3.scatter(pt[1], pt[2], color=rep_colors[name], s=120, edgecolors="black", linewidth=1.5, zorder=5)
        ax3.annotate(name, (pt[1], pt[2]), textcoords="offset points", xytext=(6, 6), fontweight="bold")
    ax3.set_xlabel("f2: 95th Percentile Wait Time (s)")
    ax3.set_ylabel("f3: Mean Ambulance Delay (s)")
    ax3.set_title("Trade-off: Fairness vs Emergency Clearance")

    # 4. 3D Pareto Front
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    ax4.scatter(F[:, 0], F[:, 1], F[:, 2], color="#2b83ba", alpha=0.6, s=35)
    for name, pt in rep_points.items():
        ax4.scatter(pt[0], pt[1], pt[2], color=rep_colors[name], s=120, edgecolors="black", linewidth=1.5, label=name)
    ax4.set_xlabel("f1 (Mean Wait)")
    ax4.set_ylabel("f2 (p95 Wait)")
    ax4.set_zlabel("f3 (Ambulance)")
    ax4.set_title("3D Non-Dominated Pareto Surface")
    ax4.legend(loc="upper right", framealpha=0.8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_grouped_bar_metrics(
    df_raw: pd.DataFrame,
    output_path: Path | str,
) -> None:
    """Generate grouped bar charts with error bars for Mean Wait, p95 Wait, and Ambulance Delay."""
    scenarios = ["normal", "rush_hour", "two_wheeler_heavy"]
    controllers = [
        "FixedTime(30s)",
        "LongestQueueFirst(margin=2.0)",
        "HandDesignedFuzzy",
        "Nayak-Fast",
        "Nayak-Fair",
        "Nayak-Rescue",
        "Nayak-Balanced",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    metrics = [
        ("mean_wait", "Mean Wait Time (s)", axes[0]),
        ("wait_p95", "95th Percentile Wait Time (s)", axes[1]),
        ("mean_ambulance_delay", "Mean Ambulance Delay (s)", axes[2]),
    ]

    x = np.arange(len(scenarios))
    width = 0.11

    # Distinct palette
    palette = ["#7f7f7f", "#bcbd22", "#17becf", "#d95f02", "#7570b3", "#e7298a", "#1b9e77"]

    for col_idx, (metric_col, title, ax) in enumerate(metrics):
        for i, (ctrl, color) in enumerate(zip(controllers, palette)):
            means = []
            stds = []
            for sc in scenarios:
                sub = df_raw[(df_raw["scenario"] == sc) & (df_raw["controller"] == ctrl)][metric_col]
                means.append(sub.mean() if len(sub) > 0 else 0.0)
                stds.append(sub.std() if len(sub) > 0 else 0.0)

            offsets = x + (i - len(controllers) / 2.0 + 0.5) * width
            label = ctrl if col_idx == 0 else ""
            ax.bar(offsets, means, width, yerr=stds, capsize=3, label=label, color=color, alpha=0.9, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios])
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Seconds")

    fig.legend(controllers, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4, frameon=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_wait_time_boxplots(
    df_raw: pd.DataFrame,
    output_path: Path | str,
) -> None:
    """Generate box plots of non-ambulance mean wait distributions across seeds."""
    scenarios = ["normal", "rush_hour", "two_wheeler_heavy"]
    controllers = [
        "FixedTime(30s)",
        "LongestQueueFirst(margin=2.0)",
        "HandDesignedFuzzy",
        "Nayak-Fast",
        "Nayak-Fair",
        "Nayak-Rescue",
        "Nayak-Balanced",
    ]

    short_names = ["FixedTime", "LQF", "HandFuzzy", "Nayak-Fast", "Nayak-Fair", "Nayak-Rescue", "Nayak-Balanced"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    for i, sc in enumerate(scenarios):
        ax = axes[i]
        sc_data = []
        for ctrl in controllers:
            vals = df_raw[(df_raw["scenario"] == sc) & (df_raw["controller"] == ctrl)]["mean_wait"].values
            sc_data.append(vals if len(vals) > 0 else [0])

        box = ax.boxplot(sc_data, patch_artist=True, tick_labels=short_names)
        colors = ["#c7c7c7", "#dbdb8d", "#9edae5", "#fdae6b", "#bcbddc", "#f781bf", "#a1d99b"]
        for patch, c in zip(box["boxes"], colors):
            patch.set_facecolor(c)

        ax.set_title(f"Scenario: {sc.replace('_', ' ').title()}", fontweight="bold")
        ax.set_ylabel("Mean Wait (s)")
        ax.tick_params(axis="x", rotation=35)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_membership_functions_comparison(
    balanced_genome_path: Path | str,
    output_path: Path | str,
) -> None:
    """Plot membership function shapes before (Hand-Designed) vs after evolution (Nayak-Balanced)."""
    with open(balanced_genome_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    g_evolved = data["genome"]

    # Decode both controllers
    inputs_hand, out_hand, _ = decode_genome(encode_default_hand_designed())
    inputs_evo, out_evo, _ = decode_genome(g_evolved)

    # Variables to plot: q_green, q_red, bus_share_red, two_wheeler_share_green, extension
    plot_vars = [
        ("q_green", "Queue Green (PCU)"),
        ("q_red", "Queue Red (PCU)"),
        ("bus_share_red", "Bus Share on Red"),
        ("two_wheeler_share_green", "Two-Wheeler Share on Green"),
        ("extension", "Extension Output (s)"),
    ]

    fig, axes = plt.subplots(len(plot_vars), 2, figsize=(12, 13), dpi=300)

    colors = {"LOW": "#386cb0", "MEDIUM": "#7fc97f", "HIGH": "#f0027f",
              "ZERO": "#386cb0", "SHORT": "#fdc086", "LONG": "#f0027f"}

    for r_idx, (vname, display_title) in enumerate(plot_vars):
        if vname == "extension":
            v_hand = out_hand
            v_evo = out_evo
        else:
            v_hand = inputs_hand[vname]
            v_evo = inputs_evo[vname]

        # Left column: Hand-Designed
        ax_left = axes[r_idx, 0]
        grid = v_hand.universe_grid
        for tname, mf in v_hand.terms.items():
            y = mf.evaluate(grid)
            ax_left.plot(grid, y, label=tname, color=colors.get(tname, "black"), linewidth=2.0)
        ax_left.set_title(f"Hand-Designed: {display_title}")
        ax_left.set_ylabel("Membership")
        ax_left.legend(loc="upper right", framealpha=0.7)

        # Right column: Evolved (Nayak-Balanced)
        ax_right = axes[r_idx, 1]
        grid_evo = v_evo.universe_grid
        for tname, mf in v_evo.terms.items():
            y = mf.evaluate(grid_evo)
            ax_right.plot(grid_evo, y, label=tname, color=colors.get(tname, "black"), linewidth=2.0)
        ax_right.set_title(f"Evolved (Nayak-Balanced): {display_title}")
        ax_right.set_ylabel("Membership")
        ax_right.legend(loc="upper right", framealpha=0.7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_queue_length_time_series(
    balanced_genome_path: Path | str,
    output_path: Path | str,
    scenario_name: str = "rush_hour",
    seed: int = 1000,
    duration_seconds: int = 3600,
) -> None:
    """Plot total intersection queue (PCU) over time for FixedTime vs Nayak-Balanced."""
    with open(balanced_genome_path, "r", encoding="utf-8") as f:
        g = json.load(f)["genome"]

    scenario = SCENARIOS[scenario_name]
    sim_cfg = DEFAULT_SIM_CONFIG
    total_seconds = sim_cfg.warmup_seconds + duration_seconds

    # Simulate FixedTime
    ctrl_ft = FixedTimeController(30)
    inter_ft = Intersection(timing_config=sim_cfg.signal)
    gen_ft = ArrivalGenerator(scenario, sim_cfg, total_seconds, seed)

    q_ft = []
    for t in range(total_seconds):
        inter_ft.add_arrivals(gen_ft.generate_step(t))
        if ctrl_ft.decide(inter_ft.get_state()) == 1:
            inter_ft.request_phase_switch()
        inter_ft.step(t)
        if t >= sim_cfg.warmup_seconds:
            q_ft.append(sum(inter_ft.get_state().queue_pcu.values()))

    # Simulate Nayak-Balanced
    ctrl_nayak = EvolvedFuzzyController(g, name="Nayak-Balanced")
    inter_ny = Intersection(timing_config=sim_cfg.signal)
    gen_ny = ArrivalGenerator(scenario, sim_cfg, total_seconds, seed)

    q_ny = []
    for t in range(total_seconds):
        inter_ny.add_arrivals(gen_ny.generate_step(t))
        if ctrl_nayak.decide(inter_ny.get_state()) == 1:
            inter_ny.request_phase_switch()
        inter_ny.step(t)
        if t >= sim_cfg.warmup_seconds:
            q_ny.append(sum(inter_ny.get_state().queue_pcu.values()))

    time_hrs = np.arange(len(q_ft)) / 3600.0

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(time_hrs, q_ft, color="#d95f02", label="FixedTime (30s Split)", linewidth=1.5, alpha=0.85)
    ax.plot(time_hrs, q_ny, color="#1b9e77", label="Nayak-Balanced (Evolved Fuzzy)", linewidth=1.8)

    ax.set_title(f"Dynamic Queue Length (PCU) Over Time — {scenario_name.replace('_', ' ').title()} (Seed {seed})", fontweight="bold")
    ax.set_xlabel("Evaluation Time (Hours)")
    ax.set_ylabel("Total Intersection Queue (PCU)")
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
