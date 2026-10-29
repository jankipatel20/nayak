"""Master benchmark execution script running paired experiments, statistics, ablations, and figures."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Ensure nayak package is importable
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nayak.config import DEFAULT_SIM_CONFIG, SCENARIOS
from nayak.controllers.base import BaseController
from nayak.controllers.fixed_time import FixedTimeController
from nayak.controllers.fuzzy_controller import EvolvedFuzzyController, HandDesignedFuzzyController
from nayak.controllers.longest_queue import LongestQueueFirstController
from nayak.evolve.genome import decode_genome
from nayak.experiments.ablations import run_ablation_experiments
from nayak.experiments.plots import (
    plot_grouped_bar_metrics,
    plot_hypervolume_convergence,
    plot_membership_functions_comparison,
    plot_pareto_front,
    plot_queue_length_time_series,
    plot_wait_time_boxplots,
)
from nayak.experiments.stats import analyze_experiment_statistics
from nayak.fuzzy.explain import export_rule_report
from nayak.sim.runner import run_episode


def build_controller(name: str, genome: Optional[List[float]] = None) -> BaseController:
    """Instantiate a controller from its name and optional genome."""
    if name == "FixedTime(30s)":
        return FixedTimeController(30)
    elif name == "LongestQueueFirst(margin=2.0)":
        return LongestQueueFirstController(2.0)
    elif name == "HandDesignedFuzzy":
        return HandDesignedFuzzyController()
    else:
        if genome is None:
            raise ValueError(f"Genome required for evolved controller {name}")
        return EvolvedFuzzyController(genome, name=name)


def evaluate_single_run(
    controller_name: str,
    genome: Optional[List[float]],
    scenario_name: str,
    seed: int,
    episode_seconds: int,
    warmup_seconds: int,
) -> Dict[str, object]:
    """Execute a single episode and return metrics dictionary."""
    ctrl = build_controller(controller_name, genome)
    scenario = SCENARIOS[scenario_name]
    m, _ = run_episode(
        controller=ctrl,
        scenario=scenario,
        episode_seconds=episode_seconds,
        warmup_seconds=warmup_seconds,
        seed=seed,
    )
    return {
        "scenario": scenario_name,
        "controller": controller_name,
        "seed": seed,
        "mean_wait": m.mean_wait,
        "wait_p95": m.wait_p95,
        "max_wait": m.max_wait,
        "mean_ambulance_delay": m.mean_ambulance_delay,
        "throughput_vehicles": m.throughput_vehicles,
        "throughput_pcu": m.throughput_pcu,
        "avg_queue_pcu": m.avg_queue_pcu,
        "max_queue_pcu": m.max_queue_pcu,
        "unserved_queued": m.unserved_queued,
    }


def run_all_experiments(
    num_seeds: int = 20,
    episode_seconds: int = 3600,
    warmup_seconds: int = 300,
    n_workers: int = 8,
    results_dir: Optional[Path] = None,
) -> None:
    """Run full benchmark, compute statistics and ablations, and generate publication plots."""
    res_dir = results_dir or (ROOT_DIR / "results")
    fig_dir = res_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load evolved genome picks
    variants = ["Nayak-Fast", "Nayak-Fair", "Nayak-Rescue", "Nayak-Balanced"]
    genomes: Dict[str, List[float]] = {}
    for v in variants:
        v_path = res_dir / f"{v}.json"
        if not v_path.exists():
            raise FileNotFoundError(f"Missing pretrained genome file: {v_path}. Run evolution first!")
        with open(v_path, "r", encoding="utf-8") as f:
            genomes[v] = json.load(f)["genome"]

    # Build controllers list with optional genome
    controllers_to_run = [
        ("FixedTime(30s)", None),
        ("LongestQueueFirst(margin=2.0)", None),
        ("HandDesignedFuzzy", None),
        ("Nayak-Fast", genomes["Nayak-Fast"]),
        ("Nayak-Fair", genomes["Nayak-Fair"]),
        ("Nayak-Rescue", genomes["Nayak-Rescue"]),
        ("Nayak-Balanced", genomes["Nayak-Balanced"]),
    ]

    # Test seeds: strictly disjoint from training seeds
    test_seeds = [1000 + i for i in range(num_seeds)]
    scenarios = ["normal", "rush_hour", "two_wheeler_heavy"]

    print("=" * 104)
    print("NAYAK: FULL PAIRED EXPERIMENTS & EVALUATION SUITE")
    print(f"Controllers: {len(controllers_to_run)} | Scenarios: {len(scenarios)} | Held-Out Seeds: {num_seeds} ({test_seeds[0]}..{test_seeds[-1]})")
    print(f"Horizon: {episode_seconds}s episode (excluding {warmup_seconds}s warmup) | Parallel Workers: {n_workers}")
    print("=" * 104)

    # 2. Parallel episode execution
    tasks = [
        (c_name, genome, sc, seed)
        for c_name, genome in controllers_to_run
        for sc in scenarios
        for seed in test_seeds
    ]

    print(f"\n[1/5] Executing {len(tasks)} paired simulation episodes in parallel...", flush=True)
    results = Parallel(n_jobs=n_workers)(
        delayed(evaluate_single_run)(c_name, genome, sc, seed, episode_seconds, warmup_seconds)
        for c_name, genome, sc, seed in tasks
    )

    df_raw = pd.DataFrame(results)

    # 3. Save detailed raw results and summary
    raw_csv = res_dir / "results_raw.csv"
    df_raw.to_csv(raw_csv, index=False)

    # Grouped summary: mean and std across seeds
    summary_cols = [
        "mean_wait",
        "wait_p95",
        "max_wait",
        "mean_ambulance_delay",
        "throughput_vehicles",
        "throughput_pcu",
        "avg_queue_pcu",
        "max_queue_pcu",
    ]
    df_summary = df_raw.groupby(["scenario", "controller"])[summary_cols].agg(["mean", "std"])
    summary_csv = res_dir / "results.csv"
    df_summary.to_csv(summary_csv)
    print(f"Benchmark results successfully saved to: {summary_csv.relative_to(ROOT_DIR)}", flush=True)

    # 4. Statistical significance testing
    print("\n[2/5] Running paired Wilcoxon signed-rank tests with Holm-Bonferroni correction...", flush=True)
    df_stats = analyze_experiment_statistics(df_raw, metrics=["mean_wait", "wait_p95", "mean_ambulance_delay"])
    stats_csv = res_dir / "stats.csv"
    df_stats.to_csv(stats_csv, index=False)
    print(f"Statistical significance tests saved to: {stats_csv.relative_to(ROOT_DIR)}", flush=True)

    # 5. Ablation studies
    print("\n[3/5] Running ablation studies (Evolution Components, Mixed PCU, Ambulance Priority)...", flush=True)
    df_ablations = run_ablation_experiments(
        balanced_genome=genomes["Nayak-Balanced"],
        test_seeds=test_seeds[:10],  # run 10 test seeds for ablations
        episode_seconds=episode_seconds,
        warmup_seconds=warmup_seconds,
    )
    ablations_csv = res_dir / "ablations.csv"
    df_ablations.to_csv(ablations_csv, index=False)
    print(f"Ablation study records saved to: {ablations_csv.relative_to(ROOT_DIR)}", flush=True)

    # 6. Generate publication plots
    print("\n[4/5] Generating publication-grade matplotlib figures...", flush=True)
    hv_json = res_dir / "hypervolume_history.json"
    if hv_json.exists():
        plot_hypervolume_convergence(hv_json, fig_dir / "hypervolume_convergence.png")

    pareto_npz = res_dir / "pareto.npz"
    if pareto_npz.exists():
        plot_pareto_front(pareto_npz, res_dir, fig_dir / "pareto_front.png")

    plot_grouped_bar_metrics(df_raw, fig_dir / "grouped_bar_metrics.png")
    plot_wait_time_boxplots(df_raw, fig_dir / "wait_time_boxplots.png")
    plot_membership_functions_comparison(res_dir / "Nayak-Balanced.json", fig_dir / "membership_functions_comparison.png")
    plot_queue_length_time_series(res_dir / "Nayak-Balanced.json", fig_dir / "queue_length_time_series.png", scenario_name="rush_hour", seed=test_seeds[0])
    print(f"All 6 figures generated in: {fig_dir.relative_to(ROOT_DIR)}/", flush=True)

    # 7. Update rule report with evolved rules
    print("\n[5/5] Updating explainable rule base report...", flush=True)
    inputs_evo, out_evo, rules_evo = decode_genome(genomes["Nayak-Balanced"])
    rule_report_path = res_dir / "rule_report.md"
    export_rule_report(
        filepath=rule_report_path,
        rules=rules_evo,
        input_variables=inputs_evo,
        output_variable=out_evo,
        min_active_weight=0.10,
        title="NAYAK Evolved Fuzzy Controller (Nayak-Balanced) Rule Base & Membership Report",
    )
    print(f"Evolved rule report updated at: {rule_report_path.relative_to(ROOT_DIR)}", flush=True)

    # Print summary highlights
    print("\n" + "=" * 104)
    print("EXPERIMENTAL SUMMARY TABLE (Mean Wait, Ambulance Delay, Throughput across 20 Seeds):")
    print("=" * 104)
    for sc in scenarios:
        print(f"\n--- Scenario: {sc.upper()} ---")
        print(f"{'Controller':<30} | {'Mean Wait (s)':<14} | {'p95 Wait (s)':<13} | {'Amb Delay (s)':<14} | {'Throughput (veh)':<15}")
        print("-" * 104)
        for c_name, _ in controllers_to_run:
            sub = df_raw[(df_raw["scenario"] == sc) & (df_raw["controller"] == c_name)]
            mw = f"{sub['mean_wait'].mean():.2f} ± {sub['mean_wait'].std():.2f}"
            p95 = f"{sub['wait_p95'].mean():.2f} ± {sub['wait_p95'].std():.2f}"
            amb = f"{sub['mean_ambulance_delay'].mean():.2f} ± {sub['mean_ambulance_delay'].std():.2f}"
            tp = f"{sub['throughput_vehicles'].mean():.1f}"
            print(f"{c_name:<30} | {mw:>14} | {p95:>13} | {amb:>14} | {tp:>15}")
        print("-" * 104)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NAYAK paired experimental benchmarks.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of held-out test seeds (default: 20)")
    parser.add_argument("--episode-seconds", type=int, default=3600, help="Episode duration in seconds (default: 3600)")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 8), help="Worker processes")
    args = parser.parse_args()

    run_all_experiments(
        num_seeds=args.seeds,
        episode_seconds=args.episode_seconds,
        n_workers=args.workers,
    )


if __name__ == "__main__":
    main()
