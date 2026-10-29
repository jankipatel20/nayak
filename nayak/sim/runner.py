"""Simulation execution engine for running episodes with a given controller."""

from __future__ import annotations

from typing import Tuple

from nayak.config import (
    DEFAULT_SIM_CONFIG,
    ScenarioConfig,
    SimulationConfig,
)
from nayak.controllers.base import BaseController, ControllerAction
from nayak.sim.arrivals import ArrivalGenerator
from nayak.sim.intersection import Intersection
from nayak.sim.metrics import SimulationMetrics, compute_metrics


def run_episode(
    controller: BaseController,
    scenario: ScenarioConfig,
    episode_seconds: int = 900,
    warmup_seconds: int = 300,
    seed: int = 42,
    sim_config: SimulationConfig | None = None,
) -> Tuple[SimulationMetrics, Intersection]:
    """Execute a single deterministic simulation episode.

    Args:
        controller: Signal controller under test.
        scenario: Traffic arrival scenario configuration.
        episode_seconds: Evaluation duration in seconds (after warm-up).
        warmup_seconds: Duration in seconds to run warm-up (excluded from metrics).
        seed: Random seed for arrival generation.
        sim_config: Overall simulation configuration.

    Returns:
        Tuple of (SimulationMetrics, final Intersection instance).
    """
    cfg = sim_config or DEFAULT_SIM_CONFIG
    total_seconds = warmup_seconds + episode_seconds

    intersection = Intersection(timing_config=cfg.signal)
    arrival_gen = ArrivalGenerator(
        scenario=scenario,
        sim_config=cfg,
        total_seconds=total_seconds,
        seed=seed,
    )
    controller.reset()

    queue_pcu_history = []

    for t in range(total_seconds):
        # 1. Arrival step
        arrivals = arrival_gen.generate_step(t)
        intersection.add_arrivals(arrivals)

        # 2. Signal control decision
        state = intersection.get_state()
        action = controller.decide(state)
        if action == ControllerAction.SWITCH:
            intersection.request_phase_switch()

        # 3. Intersection physics step (advances timer and discharges queues)
        intersection.step(t)

        # 4. Record queue metrics during post-warmup evaluation
        if t >= warmup_seconds:
            curr_state = intersection.get_state()
            total_q_pcu = sum(curr_state.queue_pcu.values())
            queue_pcu_history.append(total_q_pcu)

    # 5. Compute summary metrics over the evaluation horizon
    metrics = compute_metrics(
        all_vehicles=intersection.all_arrived_vehicles,
        queue_pcu_history=queue_pcu_history,
        total_seconds=total_seconds,
        warmup_seconds=warmup_seconds,
    )

    return metrics, intersection
