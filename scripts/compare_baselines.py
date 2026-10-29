"""Baseline controllers comparative evaluation script across 10 held-out test seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

# Ensure nayak package is importable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nayak.config import DEFAULT_SIM_CONFIG, SCENARIOS, SimulationConfig
from nayak.controllers.base import BaseController
from nayak.controllers.fixed_time import FixedTimeController
from nayak.controllers.fuzzy_controller import HandDesignedFuzzyController
from nayak.controllers.longest_queue import LongestQueueFirstController
from nayak.fuzzy.explain import export_rule_report
from nayak.fuzzy.rulebase import (
    create_default_input_variables,
    create_default_output_variable,
    create_fixed_rule_base,
)
from nayak.sim.metrics import SimulationMetrics
from nayak.sim.runner import run_episode


def evaluate_controller_on_scenario(
    controller: BaseController,
    scenario_name: str,
    seeds: List[int],
    episode_seconds: int = 3600,
    warmup_seconds: int = 300,
) -> Dict[str, float]:
    """Evaluate a controller across seeds and return mean metrics."""
    scenario = SCENARIOS[scenario_name]
    metrics_list: List[SimulationMetrics] = []

    for seed in seeds:
        m, _ = run_episode(
            controller=controller,
            scenario=scenario,
            episode_seconds=episode_seconds,
            warmup_seconds=warmup_seconds,
            seed=seed,
        )
        metrics_list.append(m)

    return {
        "mean_wait": float(np.mean([m.mean_wait for m in metrics_list])),
        "wait_p95": float(np.mean([m.wait_p95 for m in metrics_list])),
        "max_wait": float(np.mean([m.max_wait for m in metrics_list])),
        "amb_delay": float(np.mean([m.mean_ambulance_delay for m in metrics_list])),
        "avg_queue": float(np.mean([m.avg_queue_pcu for m in metrics_list])),
        "throughput": float(np.mean([m.throughput_vehicles for m in metrics_list])),
    }


def run_baseline_comparison(
    seeds: List[int],
    episode_seconds: int = 3600,
    warmup_seconds: int = 300,
) -> None:
    """Run comparative benchmark of FixedTime, LongestQueueFirst, and HandDesignedFuzzy."""
    controllers: List[BaseController] = [
        FixedTimeController(30),
        LongestQueueFirstController(queue_margin_pcu=2.0),
        HandDesignedFuzzyController(),
    ]

    scenario_names = ["normal", "rush_hour", "two_wheeler_heavy"]

    print("=" * 104)
    print("PHASE 2: BASELINE CONTROLLERS BENCHMARK OVER 10 TEST SEEDS")
    print(f"Evaluation Horizon: {episode_seconds}s (warmup {warmup_seconds}s excluded) across seeds {seeds[0]}..{seeds[-1]}")
    print("=" * 104)

    for sc_name in scenario_names:
        print(f"\n--- SCENARIO: {sc_name.upper()} ({SCENARIOS[sc_name].description}) ---")
        print("-" * 104)
        print(
            f"{'Controller':<24} | {'Mean Wait (s)':<14} | {'p95 Wait (s)':<13} | {'Max Wait (s)':<13} | "
            f"{'Amb Delay (s)':<14} | {'Avg Q (PCU)':<11} | {'Throughput (veh)':<15}"
        )
        print("-" * 104)

        for ctrl in controllers:
            res = evaluate_controller_on_scenario(
                controller=ctrl,
                scenario_name=sc_name,
                seeds=seeds,
                episode_seconds=episode_seconds,
                warmup_seconds=warmup_seconds,
            )
            print(
                f"{ctrl.name:<24} | {res['mean_wait']:>12.2f} s | {res['wait_p95']:>11.2f} s | {res['max_wait']:>11.2f} s | "
                f"{res['amb_delay']:>12.2f} s | {res['avg_queue']:>9.2f} | {res['throughput']:>14.1f}"
            )
        print("-" * 104)

    # Export initial rule base report to results/rule_report.md
    report_path = ROOT_DIR / "results" / "rule_report.md"
    export_rule_report(
        filepath=report_path,
        rules=create_fixed_rule_base(),
        input_variables=create_default_input_variables(),
        output_variable=create_default_output_variable(),
        title="NAYAK Hand-Designed Fuzzy Rule Base & Initial Membership Report",
    )
    print(f"\nInitial rule base report exported to: {report_path.relative_to(ROOT_DIR)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Phase 2 baseline controllers.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds (default: 10)")
    parser.add_argument("--episode-seconds", type=int, default=3600, help="Episode seconds (default: 3600)")
    args = parser.parse_args()

    seed_list = [1000 + i for i in range(args.seeds)]
    run_baseline_comparison(seed_list, episode_seconds=args.episode_seconds)


if __name__ == "__main__":
    main()
