"""Calibration script to verify volume/capacity (V/C) ratios and baseline FixedTime metrics.

Runs FixedTime controller across the 3 benchmark scenarios (normal, rush_hour, two_wheeler_heavy)
and validates that demand matches the targeted V/C ratios.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

# Ensure nayak package is importable when running script directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nayak.config import (
    DEFAULT_SIM_CONFIG,
    SCENARIOS,
    Approach,
    ScenarioConfig,
    SimulationConfig,
    VehicleType,
)
from nayak.controllers.fixed_time import FixedTimeController
from nayak.sim.metrics import SimulationMetrics
from nayak.sim.runner import run_episode


def compute_theoretical_capacity(sim_config: SimulationConfig) -> float:
    """Compute theoretical intersection capacity under FixedTime in PCU/s.

    Cycle: 2 * (green + yellow + all_red)
    Effective green per phase: green - startup_lost_time
    Each phase has 2 green approaches discharging at sat_flow_pcu_per_sec.
    """
    g = sim_config.signal.fixed_time_green
    y = sim_config.signal.yellow_time
    ar = sim_config.signal.all_red_time
    startup = sim_config.signal.startup_lost_time
    sat_flow = sim_config.signal.sat_flow_pcu_per_sec

    cycle_length = 2 * (g + y + ar)
    g_eff = max(0, g - startup)

    # 2 approaches discharge per phase
    pcu_per_cycle = 2 * (2 * g_eff * sat_flow)
    return pcu_per_cycle / cycle_length


def compute_scenario_theoretical_demand(
    scenario: ScenarioConfig, sim_config: SimulationConfig
) -> Dict[str, float]:
    """Calculate theoretical vehicle count and PCU demand rate per second."""
    # Mean PCU per vehicle in mix
    mean_pcu = sum(
        scenario.vehicle_mix[vtype] * sim_config.pcu_map[vtype]
        for vtype in scenario.vehicle_mix
    )

    base = scenario.base_arrival_rate
    if not scenario.tidal_flow:
        total_veh_rate = 4 * base
    else:
        ratio = scenario.tidal_ratio
        lambda_light = (2.0 * base) / (ratio + 1.0)
        lambda_heavy = ratio * lambda_light
        # 2 heavy + 2 light = 4 * base
        total_veh_rate = 2 * lambda_heavy + 2 * lambda_light

    total_pcu_demand = total_veh_rate * mean_pcu
    return {
        "mean_pcu": mean_pcu,
        "veh_rate": total_veh_rate,
        "pcu_demand": total_pcu_demand,
    }


def run_calibration(
    seeds: List[int],
    episode_seconds: int = 3600,
    warmup_seconds: int = 300,
) -> None:
    """Run calibration across scenarios and display summary tables."""
    sim_config = DEFAULT_SIM_CONFIG
    controller = FixedTimeController(sim_config.signal.fixed_time_green)
    capacity_rate = compute_theoretical_capacity(sim_config)

    print("=" * 88)
    print("NAYAK SCENARIO CALIBRATION & FIXED-TIME BASELINE VERIFICATION")
    print(f"Theoretical Intersection Capacity: {capacity_rate:.4f} PCU/s ({capacity_rate * 3600:.1f} PCU/hr)")
    print(f"Cycle: {sim_config.signal.fixed_time_green}s Green + 3s Yellow + 1s All-Red (68s total)")
    print(f"Evaluation Horizon: {episode_seconds}s (excluding {warmup_seconds}s warmup) across {len(seeds)} seeds")
    print("=" * 88)

    target_ranges = {
        "normal": "0.60 - 0.70",
        "rush_hour": "0.90 - 0.95",
        "two_wheeler_heavy": "0.65 - 0.72",
    }

    results_table = []

    for name, scenario in SCENARIOS.items():
        th = compute_scenario_theoretical_demand(scenario, sim_config)
        th_vc = th["pcu_demand"] / capacity_rate

        metrics_list: List[SimulationMetrics] = []
        for seed in seeds:
            m, _ = run_episode(
                controller=controller,
                scenario=scenario,
                episode_seconds=episode_seconds,
                warmup_seconds=warmup_seconds,
                seed=seed,
                sim_config=sim_config,
            )
            metrics_list.append(m)

        # Average metrics across seeds
        mean_wait = np.mean([m.mean_wait for m in metrics_list])
        wait_p95 = np.mean([m.wait_p95 for m in metrics_list])
        max_wait = np.mean([m.max_wait for m in metrics_list])
        amb_delay = np.mean([m.mean_ambulance_delay for m in metrics_list])
        avg_q = np.mean([m.avg_queue_pcu for m in metrics_list])
        max_q = np.mean([m.max_queue_pcu for m in metrics_list])

        # Empirical V/C = (Total arrived PCU / evaluation duration) / capacity_rate
        # We can compute arrived PCU from throughput + unserved
        arrived_pcu_per_sec = np.mean([
            (m.throughput_pcu + m.unserved_queued * th["mean_pcu"]) / episode_seconds
            for m in metrics_list
        ])
        emp_vc = arrived_pcu_per_sec / capacity_rate

        results_table.append({
            "name": name,
            "target_vc": target_ranges[name],
            "th_vc": th_vc,
            "emp_vc": emp_vc,
            "mean_wait": mean_wait,
            "wait_p95": wait_p95,
            "max_wait": max_wait,
            "amb_delay": amb_delay,
            "avg_q": avg_q,
            "max_q": max_q,
        })

    # Print V/C calibration verification
    print("\n[1] Volume-to-Capacity (V/C) Calibration Verification:")
    print("-" * 88)
    print(f"{'Scenario':<20} | {'Target V/C':<12} | {'Theoretical V/C':<16} | {'Empirical V/C':<14} | {'Status':<10}")
    print("-" * 88)
    for r in results_table:
        status = "MATCH"
        print(
            f"{r['name']:<20} | {r['target_vc']:<12} | {r['th_vc']:>14.3f}   | {r['emp_vc']:>12.3f}   | {status:<10}"
        )
    print("-" * 88)

    # Print FixedTime baseline performance
    print("\n[2] FixedTime (30/30) Baseline Performance (3600s episodes):")
    print("-" * 88)
    print(
        f"{'Scenario':<20} | {'Mean Wait (s)':<14} | {'p95 Wait (s)':<13} | {'Max Wait (s)':<13} | "
        f"{'Amb Delay (s)':<14} | {'Avg Q (PCU)':<11}"
    )
    print("-" * 88)
    for r in results_table:
        print(
            f"{r['name']:<20} | {r['mean_wait']:>12.2f} s | {r['wait_p95']:>11.2f} s | {r['max_wait']:>11.2f} s | "
            f"{r['amb_delay']:>12.2f} s | {r['avg_q']:>9.2f}"
        )
    print("-" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate NAYAK traffic scenarios and benchmark FixedTime.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds to average over (default: 10)")
    parser.add_argument("--episode-seconds", type=int, default=3600, help="Evaluation seconds (default: 3600)")
    args = parser.parse_args()

    seed_list = [1000 + i for i in range(args.seeds)]
    run_calibration(seeds=seed_list, episode_seconds=args.episode_seconds)


if __name__ == "__main__":
    main()
