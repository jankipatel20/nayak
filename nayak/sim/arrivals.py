"""Stochastic vehicle arrival processes and ambulance scheduling."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from nayak.config import Approach, ScenarioConfig, SimulationConfig, VehicleType
from nayak.sim.vehicles import Vehicle


class ArrivalGenerator:
    """Generates stochastic mixed-traffic arrivals and scheduled emergency vehicles.

    Uses an isolated numpy.random.Generator to maintain strict determinism across runs.
    """

    def __init__(
        self,
        scenario: ScenarioConfig,
        sim_config: SimulationConfig,
        total_seconds: int,
        seed: int,
    ) -> None:
        """Initialize the arrival generator.

        Args:
            scenario: Configuration for traffic mix and arrival rates.
            sim_config: Simulation parameters including PCU mapping.
            total_seconds: Total duration of the simulation episode.
            seed: Deterministic integer seed for reproducible generation.
        """
        self.scenario = scenario
        self.sim_config = sim_config
        self.total_seconds = total_seconds
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self._vehicle_id_counter = 0

        # Precompute vehicle mix cumulative distribution
        self._types: List[VehicleType] = list(self.scenario.vehicle_mix.keys())
        probabilities = [self.scenario.vehicle_mix[t] for t in self._types]
        total_prob = sum(probabilities)
        self._normalized_probs = [p / total_prob for p in probabilities]

        # Pre-schedule ambulances across the post-warmup evaluation horizon
        self._scheduled_ambulances: Dict[int, List[Approach]] = {}
        self._schedule_ambulances()

    def _schedule_ambulances(self) -> None:
        """Deterministic scheduling of 3 to 5 ambulances at random times and approaches."""
        min_arr = self.scenario.ambulance_min_arrivals
        max_arr = self.scenario.ambulance_max_arrivals
        count = int(self.rng.integers(min_arr, max_arr + 1))

        # Schedule between warmup + 30s and total - 30s so evaluation is active
        t_start = max(10, self.sim_config.warmup_seconds + 30)
        t_end = max(t_start + 1, self.total_seconds - 30)

        for _ in range(count):
            arrival_sec = int(self.rng.integers(t_start, t_end))
            approach_idx = int(self.rng.integers(0, 4))
            approach = Approach(approach_idx)

            if arrival_sec not in self._scheduled_ambulances:
                self._scheduled_ambulances[arrival_sec] = []
            self._scheduled_ambulances[arrival_sec].append(approach)

    def _get_arrival_rates(self, current_time: float) -> Dict[Approach, float]:
        """Compute arrival rates per approach, incorporating tidal flow if enabled."""
        base = self.scenario.base_arrival_rate
        if not self.scenario.tidal_flow:
            return {app: base for app in Approach}

        ratio = self.scenario.tidal_ratio
        # Total rate across 4 approaches is 4 * base
        lambda_light = (2.0 * base) / (ratio + 1.0)
        lambda_heavy = ratio * lambda_light

        # Tidal shift: first half NS is heavy, second half EW is heavy
        if current_time < (self.total_seconds / 2.0):
            return {
                Approach.N: lambda_heavy,
                Approach.S: lambda_heavy,
                Approach.E: lambda_light,
                Approach.W: lambda_light,
            }
        else:
            return {
                Approach.N: lambda_light,
                Approach.S: lambda_light,
                Approach.E: lambda_heavy,
                Approach.W: lambda_heavy,
            }

    def generate_step(self, current_time: int) -> Dict[Approach, List[Vehicle]]:
        """Generate arriving vehicles for the current simulation second.

        Args:
            current_time: Current simulation second.

        Returns:
            Dictionary mapping each Approach to any vehicles arriving this second.
        """
        rates = self._get_arrival_rates(current_time)
        arrivals: Dict[Approach, List[Vehicle]] = {app: [] for app in Approach}

        # 1. Scheduled emergency vehicles for this second
        if current_time in self._scheduled_ambulances:
            for approach in self._scheduled_ambulances[current_time]:
                self._vehicle_id_counter += 1
                amb = Vehicle(
                    id=self._vehicle_id_counter,
                    vehicle_type=VehicleType.AMBULANCE,
                    approach=approach,
                    arrival_time=float(current_time),
                    pcu=self.sim_config.pcu_map[VehicleType.AMBULANCE],
                )
                arrivals[approach].append(amb)

        # 2. Regular stochastic arrivals per approach (Bernoulli trial per second)
        for approach, rate in rates.items():
            if self.rng.random() < rate:
                type_idx = int(self.rng.choice(len(self._types), p=self._normalized_probs))
                vtype = self._types[type_idx]
                self._vehicle_id_counter += 1
                veh = Vehicle(
                    id=self._vehicle_id_counter,
                    vehicle_type=vtype,
                    approach=approach,
                    arrival_time=float(current_time),
                    pcu=self.sim_config.pcu_map[vtype],
                )
                arrivals[approach].append(veh)

        return arrivals
