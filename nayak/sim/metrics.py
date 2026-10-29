"""Evaluation metrics calculation for traffic simulation runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import numpy as np

from nayak.sim.vehicles import Vehicle


@dataclass
class SimulationMetrics:
    """Summary metrics of a traffic simulation run."""
    mean_wait: float
    wait_p95: float
    max_wait: float
    mean_ambulance_delay: float
    throughput_vehicles: int
    throughput_pcu: float
    avg_queue_pcu: float
    max_queue_pcu: float
    total_arrived: int
    total_departed: int
    unserved_queued: int


def compute_metrics(
    all_vehicles: Sequence[Vehicle],
    queue_pcu_history: Sequence[float],
    total_seconds: int,
    warmup_seconds: int = 300,
) -> SimulationMetrics:
    """Compute performance metrics over the post-warmup evaluation horizon.

    Args:
        all_vehicles: All vehicles created in the simulation (departed + still queued).
        queue_pcu_history: Recorded total intersection queue in PCU at each evaluation second.
        total_seconds: Total simulation horizon in seconds.
        warmup_seconds: Initial seconds excluded from metric calculation.

    Returns:
        SimulationMetrics dataclass with all required objectives and metrics.
    """
    # Filter to vehicles that arrived during the post-warmup evaluation window
    eval_vehicles = [v for v in all_vehicles if v.arrival_time >= warmup_seconds]

    non_amb_waits: List[float] = []
    amb_waits: List[float] = []

    throughput_veh = 0
    throughput_pcu = 0.0
    unserved_count = 0

    for v in eval_vehicles:
        w = v.wait_time(float(total_seconds))
        if v.departed:
            throughput_veh += 1
            throughput_pcu += v.pcu
        else:
            unserved_count += 1

        if v.is_ambulance:
            amb_waits.append(w)
        else:
            non_amb_waits.append(w)

    # Calculate wait statistics for non-ambulance vehicles
    if non_amb_waits:
        mean_wait = float(np.mean(non_amb_waits))
        wait_p95 = float(np.percentile(non_amb_waits, 95))
        max_wait = float(np.max(non_amb_waits))
    else:
        mean_wait = 0.0
        wait_p95 = 0.0
        max_wait = 0.0

    # Calculate ambulance delays
    if amb_waits:
        mean_amb_delay = float(np.mean(amb_waits))
    else:
        mean_amb_delay = 0.0

    # Calculate queue statistics
    if queue_pcu_history:
        avg_q = float(np.mean(queue_pcu_history))
        max_q = float(np.max(queue_pcu_history))
    else:
        avg_q = 0.0
        max_q = 0.0

    return SimulationMetrics(
        mean_wait=mean_wait,
        wait_p95=wait_p95,
        max_wait=max_wait,
        mean_ambulance_delay=mean_amb_delay,
        throughput_vehicles=throughput_veh,
        throughput_pcu=throughput_pcu,
        avg_queue_pcu=avg_q,
        max_queue_pcu=max_q,
        total_arrived=len(eval_vehicles),
        total_departed=throughput_veh,
        unserved_queued=unserved_count,
    )
