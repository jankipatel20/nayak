"""Ablation studies isolating the contributions of soft computing components in NAYAK."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
import pandas as pd

from nayak.config import SCENARIOS, Approach
from nayak.controllers.base import BaseController, ControllerAction
from nayak.controllers.fuzzy_controller import EvolvedFuzzyController
from nayak.evolve.genome import NUM_RULES, decode_genome, encode_default_hand_designed
from nayak.fuzzy.engine import MamdaniFIS
from nayak.fuzzy.rulebase import create_default_input_variables, create_default_output_variable, create_fixed_rule_base
from nayak.sim.intersection import IntersectionState
from nayak.sim.runner import run_episode


class RawCountFuzzyController(BaseController):
    """Fuzzy controller variant that replaces PCU-weighted queues with raw vehicle counts."""

    def __init__(self, base_evolved_controller: EvolvedFuzzyController) -> None:
        self.controller = base_evolved_controller

    @property
    def name(self) -> str:
        return "Nayak-RawCountInputs"

    def reset(self) -> None:
        self.controller.reset()

    def decide(self, state: IntersectionState) -> ControllerAction:
        """Substitute raw counts for q_green and q_red."""
        green_apps = (
            {Approach.N, Approach.S}
            if state.current_phase == 0
            else {Approach.E, Approach.W}
        )
        red_apps = set(Approach) - green_apps

        raw_q_green = float(sum(state.queue_count[app] for app in green_apps))
        raw_q_red = float(sum(state.queue_count[app] for app in red_apps))

        # Proxy state with raw counts replacing PCU queues
        proxy_state = IntersectionState(
            current_phase=state.current_phase,
            phase_state=state.phase_state,
            green_elapsed=state.green_elapsed,
            time_in_state=state.time_in_state,
            q_green=raw_q_green,
            q_red=raw_q_red,
            bus_share_red=state.bus_share_red,
            two_wheeler_share_green=state.two_wheeler_share_green,
            ambulance_red=state.ambulance_red,
            ambulance_green=state.ambulance_green,
            queue_pcu=state.queue_pcu,
            queue_count=state.queue_count,
        )
        return self.controller.decide(proxy_state)


def create_ablation_controllers(
    balanced_genome: Sequence[float],
) -> Dict[str, BaseController]:
    """Construct controller variants for Ablation A, B, and C."""
    g = list(balanced_genome)
    base_balanced = EvolvedFuzzyController(g, name="Both-Evolved (Nayak-Balanced)")

    # 1. Ablation A: MFs Only (Evolved MFs + all rule weights = 1.0)
    g_mfs_only = list(g)
    g_mfs_only[-NUM_RULES:] = [1.0] * NUM_RULES
    ctrl_mfs_only = EvolvedFuzzyController(g_mfs_only, name="MFs-Only-Evolved")

    # 2. Ablation A: Weights Only (Default HandDesigned MFs + Evolved rule weights)
    g_hand = list(encode_default_hand_designed())
    g_weights_only = list(g_hand)
    g_weights_only[-NUM_RULES:] = g[-NUM_RULES:]
    ctrl_weights_only = EvolvedFuzzyController(g_weights_only, name="Weights-Only-Evolved")

    # 3. Ablation B: Raw Vehicle-Count Inputs
    ctrl_raw_count = RawCountFuzzyController(base_balanced)

    # 4. Ablation C: Ambulance Rules Disabled
    g_no_rescue = list(g)
    # Rules 0, 1, 2 are ambulance rules in create_fixed_rule_base
    rule_offset = len(g) - NUM_RULES
    g_no_rescue[rule_offset + 0] = 0.0  # ambulance_red rule
    g_no_rescue[rule_offset + 1] = 0.0  # ambulance_green rule
    g_no_rescue[rule_offset + 2] = 0.0  # emergency conflict rule
    ctrl_no_rescue = EvolvedFuzzyController(g_no_rescue, name="Ambulance-Rules-OFF")

    return {
        "Both-Evolved (Balanced)": base_balanced,
        "MFs-Only-Evolved": ctrl_mfs_only,
        "Weights-Only-Evolved": ctrl_weights_only,
        "PCU-Weighted (Standard)": base_balanced,
        "Raw-Vehicle-Counts": ctrl_raw_count,
        "Ambulance-Priority-ON": base_balanced,
        "Ambulance-Priority-OFF": ctrl_no_rescue,
    }


def _run_single_ablation_episode(
    study: str,
    variant_name: str,
    ctrl: BaseController,
    sc_name: str,
    seed: int,
    episode_seconds: int,
    warmup_seconds: int,
) -> Dict[str, object]:
    """Execute a single episode for ablation analysis."""
    scenario = SCENARIOS[sc_name]
    m, _ = run_episode(
        controller=ctrl,
        scenario=scenario,
        episode_seconds=episode_seconds,
        warmup_seconds=warmup_seconds,
        seed=seed,
    )
    return {
        "study": study,
        "variant": variant_name,
        "scenario": sc_name,
        "seed": seed,
        "mean_wait": m.mean_wait,
        "wait_p95": m.wait_p95,
        "max_wait": m.max_wait,
        "mean_ambulance_delay": m.mean_ambulance_delay,
        "throughput_vehicles": m.throughput_vehicles,
        "avg_queue_pcu": m.avg_queue_pcu,
    }


def run_ablation_experiments(
    balanced_genome: Sequence[float],
    test_seeds: Sequence[int],
    episode_seconds: int = 3600,
    warmup_seconds: int = 300,
    n_workers: int = 8,
) -> pd.DataFrame:
    """Run all three ablations across test seeds in parallel and return a comparison DataFrame."""
    from joblib import Parallel, delayed

    controllers = create_ablation_controllers(balanced_genome)
    scenarios = ["normal", "rush_hour", "two_wheeler_heavy"]

    study_map = {
        "Both-Evolved (Balanced)": "Ablation A: Evolution Components",
        "MFs-Only-Evolved": "Ablation A: Evolution Components",
        "Weights-Only-Evolved": "Ablation A: Evolution Components",
        "PCU-Weighted (Standard)": "Ablation B: Mixed-Traffic Modeling",
        "Raw-Vehicle-Counts": "Ablation B: Mixed-Traffic Modeling",
        "Ambulance-Priority-ON": "Ablation C: Emergency Priority",
        "Ambulance-Priority-OFF": "Ablation C: Emergency Priority",
    }

    ablation_tasks = [
        (study_map[name], name, ctrl, sc_name, seed)
        for name, ctrl in controllers.items()
        for sc_name in scenarios
        for seed in test_seeds
    ]

    records = Parallel(n_jobs=n_workers)(
        delayed(_run_single_ablation_episode)(
            study, name, ctrl, sc_name, seed, episode_seconds, warmup_seconds
        )
        for study, name, ctrl, sc_name, seed in ablation_tasks
    )

    return pd.DataFrame(records)

