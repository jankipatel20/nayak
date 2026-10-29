"""Microscopic 4-way intersection simulation core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from nayak.config import (
    Approach,
    PhaseState,
    SignalPhase,
    SignalTimingConfig,
    VehicleType,
)
from nayak.sim.vehicles import Vehicle


@dataclass
class IntersectionState:
    """Snapshot of intersection dynamics passed to signal controllers."""
    current_phase: SignalPhase
    phase_state: PhaseState
    green_elapsed: int
    time_in_state: int
    q_green: float
    q_red: float
    bus_share_red: float
    two_wheeler_share_green: float
    ambulance_red: int
    ambulance_green: int
    queue_pcu: Dict[Approach, float]
    queue_count: Dict[Approach, int]


class Intersection:
    """Discrete-time microscopic traffic intersection simulator.

    Simulates a 4-way single-lane junction with mixed Indian traffic,
    accounting for PCU road space, start-up lost time, clearance intervals,
    fractional discharge budgets, and ambulance filtering.
    """

    # Mapping of phases to their green approaches
    PHASE_APPROACHES: Dict[SignalPhase, Set[Approach]] = {
        SignalPhase.PHASE_NS: {Approach.N, Approach.S},
        SignalPhase.PHASE_EW: {Approach.E, Approach.W},
    }

    def __init__(self, timing_config: SignalTimingConfig | None = None) -> None:
        """Initialize intersection with queues and signal state.

        Args:
            timing_config: Signal timing constraints and discharge parameters.
        """
        self.timing = timing_config or SignalTimingConfig()

        self.current_phase: SignalPhase = SignalPhase.PHASE_NS
        self.phase_state: PhaseState = PhaseState.GREEN
        self.time_in_state: int = 0
        self.green_elapsed: int = 0

        # FIFO queues per approach
        self.queues: Dict[Approach, List[Vehicle]] = {app: [] for app in Approach}

        # Fractional discharge budget per approach in PCU
        self.capacity_budget: Dict[Approach, float] = {app: 0.0 for app in Approach}

        # History tracking
        self.all_arrived_vehicles: List[Vehicle] = []
        self.all_departed_vehicles: List[Vehicle] = []

    @property
    def active_green_approaches(self) -> Set[Approach]:
        """Return approaches that currently have a green signal."""
        if self.phase_state == PhaseState.GREEN:
            return self.PHASE_APPROACHES[self.current_phase]
        return set()

    def get_state(self) -> IntersectionState:
        """Extract current intersection state in natural units for the controller."""
        green_apps = self.PHASE_APPROACHES[self.current_phase]
        red_apps = set(Approach) - green_apps

        # Queues in PCU
        q_pcu = {app: sum(v.pcu for v in self.queues[app]) for app in Approach}
        q_cnt = {app: len(self.queues[app]) for app in Approach}

        q_green = sum(q_pcu[app] for app in green_apps)
        q_red = sum(q_pcu[app] for app in red_apps)

        # Bus share on red (fraction of PCU waiting on red approaches that are buses)
        red_bus_pcu = sum(
            v.pcu
            for app in red_apps
            for v in self.queues[app]
            if v.vehicle_type == VehicleType.BUS
        )
        bus_share_red = (red_bus_pcu / q_red) if q_red > 1e-6 else 0.0

        # Two-wheeler share on green (fraction of vehicle COUNT on green approaches that are two-wheelers)
        green_veh_count = sum(q_cnt[app] for app in green_apps)
        green_2w_count = sum(
            1
            for app in green_apps
            for v in self.queues[app]
            if v.vehicle_type == VehicleType.TWO_WHEELER
        )
        two_wheeler_share_green = (
            (green_2w_count / green_veh_count) if green_veh_count > 0 else 0.0
        )

        # Ambulance flags (1 if present, else 0)
        amb_red = 1 if any(
            v.is_ambulance for app in red_apps for v in self.queues[app]
        ) else 0
        amb_green = 1 if any(
            v.is_ambulance for app in green_apps for v in self.queues[app]
        ) else 0

        return IntersectionState(
            current_phase=self.current_phase,
            phase_state=self.phase_state,
            green_elapsed=self.green_elapsed,
            time_in_state=self.time_in_state,
            q_green=float(q_green),
            q_red=float(q_red),
            bus_share_red=float(bus_share_red),
            two_wheeler_share_green=float(two_wheeler_share_green),
            ambulance_red=amb_red,
            ambulance_green=amb_green,
            queue_pcu=q_pcu,
            queue_count=q_cnt,
        )

    def add_arrivals(self, arrivals: Dict[Approach, List[Vehicle]]) -> None:
        """Add newly arrived vehicles to their respective approach queues.

        Ambulances filter forward to the head of the queue (behind any earlier ambulances).
        """
        for app, vehicles in arrivals.items():
            for veh in vehicles:
                self.all_arrived_vehicles.append(veh)
                if veh.is_ambulance:
                    # Filter forward: find position after existing ambulances
                    insert_idx = 0
                    while (
                        insert_idx < len(self.queues[app])
                        and self.queues[app][insert_idx].is_ambulance
                    ):
                        insert_idx += 1
                    self.queues[app].insert(insert_idx, veh)
                else:
                    self.queues[app].append(veh)

    def request_phase_switch(self) -> bool:
        """Attempt to initiate a phase switch if min_green constraint is met.

        Returns:
            True if phase transition sequence was initiated, False otherwise.
        """
        if self.phase_state != PhaseState.GREEN:
            return False
        if self.green_elapsed < self.timing.min_green:
            return False

        # Transition to Yellow
        self.phase_state = PhaseState.YELLOW
        self.time_in_state = 0
        # Reset capacity budgets upon leaving green
        for app in self.PHASE_APPROACHES[self.current_phase]:
            self.capacity_budget[app] = 0.0
        return True

    def step(self, current_time: int) -> None:
        """Advance intersection physics by dt = 1 second.

        Handles signal state machine transitions, saturation discharge,
        and start-up lost time.

        Args:
            current_time: Current simulation timestamp in seconds.
        """
        if self.phase_state == PhaseState.GREEN:
            self.time_in_state += 1
            self.green_elapsed += 1

            # Only discharge after start-up lost time
            if self.time_in_state > self.timing.startup_lost_time:
                active_approaches = self.PHASE_APPROACHES[self.current_phase]
                for app in active_approaches:
                    self.capacity_budget[app] += self.timing.sat_flow_pcu_per_sec
                    queue = self.queues[app]
                    while queue:
                        head = queue[0]
                        if self.capacity_budget[app] >= head.pcu - 1e-9:
                            self.capacity_budget[app] -= head.pcu
                            head.departure_time = float(current_time)
                            self.all_departed_vehicles.append(head)
                            queue.pop(0)
                        else:
                            # Head vehicle cannot discharge yet; queue is blocked
                            break

            # Enforce max green ceiling
            if self.green_elapsed >= self.timing.max_green:
                self.request_phase_switch()

        elif self.phase_state == PhaseState.YELLOW:
            self.time_in_state += 1
            if self.time_in_state >= self.timing.yellow_time:
                self.phase_state = PhaseState.ALL_RED
                self.time_in_state = 0

        elif self.phase_state == PhaseState.ALL_RED:
            self.time_in_state += 1
            if self.time_in_state >= self.timing.all_red_time:
                # Switch to opposite green phase
                self.current_phase = (
                    SignalPhase.PHASE_EW
                    if self.current_phase == SignalPhase.PHASE_NS
                    else SignalPhase.PHASE_NS
                )
                self.phase_state = PhaseState.GREEN
                self.time_in_state = 0
                self.green_elapsed = 0
                for app in self.PHASE_APPROACHES[self.current_phase]:
                    self.capacity_budget[app] = 0.0
