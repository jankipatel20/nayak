"""Configuration schemas and default parameters for the NAYAK simulation and controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, Tuple


class VehicleType(str, Enum):
    """Supported vehicle categories in the mixed Indian traffic model."""
    TWO_WHEELER = "two_wheeler"
    AUTO = "auto"
    CAR = "car"
    BUS = "bus"
    AMBULANCE = "ambulance"


class Approach(IntEnum):
    """Intersection approaches."""
    N = 0
    E = 1
    S = 2
    W = 3


class SignalPhase(IntEnum):
    """Traffic signal phases."""
    PHASE_NS = 0  # North and South green together
    PHASE_EW = 1  # East and West green together


class PhaseState(str, Enum):
    """Operational state of the active signal phase."""
    GREEN = "green"
    YELLOW = "yellow"
    ALL_RED = "all_red"


# Passenger Car Unit (PCU) space equivalent weights
DEFAULT_PCU_MAP: Dict[VehicleType, float] = {
    VehicleType.TWO_WHEELER: 0.30,
    VehicleType.AUTO: 0.75,
    VehicleType.CAR: 1.00,
    VehicleType.BUS: 2.50,
    VehicleType.AMBULANCE: 1.00,
}


@dataclass(frozen=True)
class SignalTimingConfig:
    """Traffic signal timing and discharge constraints."""
    min_green: int = 8             # Minimum green hold seconds before consulting controller
    max_green: int = 60            # Hard ceiling for green hold seconds
    yellow_time: int = 3           # Lost clearance yellow time in seconds
    all_red_time: int = 1          # Lost clearance all-red time in seconds
    startup_lost_time: int = 2     # First N seconds of green with no vehicle discharge
    sat_flow_pcu_per_sec: float = 0.5  # Saturation discharge rate (PCU/s per green approach)
    fixed_time_green: int = 30     # Default green duration for FixedTime baseline
    extension_step: int = 4        # Interval (seconds) at which controller is consulted
    extension_threshold: float = 2.0  # Extension seconds below which phase switch is triggered


@dataclass(frozen=True)
class ScenarioConfig:
    """Traffic arrival scenario definition."""
    name: str
    description: str
    vehicle_mix: Dict[VehicleType, float]
    base_arrival_rate: float       # Base lambda (veh/sec/approach)
    tidal_flow: bool = False
    tidal_ratio: float = 1.5       # Ratio of heavy to light pair during rush hour
    ambulance_min_arrivals: int = 3
    ambulance_max_arrivals: int = 5


# Standard predefined scenarios
SCENARIOS: Dict[str, ScenarioConfig] = {
    "normal": ScenarioConfig(
        name="normal",
        description="Balanced approaches with standard mixed Indian traffic; V/C ~ 0.65",
        vehicle_mix={
            VehicleType.TWO_WHEELER: 0.45,
            VehicleType.AUTO: 0.17,
            VehicleType.CAR: 0.30,
            VehicleType.BUS: 0.08,
        },
        base_arrival_rate=0.175,
        tidal_flow=False,
    ),
    "rush_hour": ScenarioConfig(
        name="rush_hour",
        description="High demand near saturation (V/C ~ 0.92) with directional tidal flow (1.5:1)",
        vehicle_mix={
            VehicleType.TWO_WHEELER: 0.45,
            VehicleType.AUTO: 0.17,
            VehicleType.CAR: 0.30,
            VehicleType.BUS: 0.08,
        },
        base_arrival_rate=0.25,  # Combined 1.0 veh/s total, split 0.30 heavy / 0.20 light per lane
        tidal_flow=True,
        tidal_ratio=1.5,
    ),
    "two_wheeler_heavy": ScenarioConfig(
        name="two_wheeler_heavy",
        description="High vehicle count with predominant two-wheelers (60%); V/C ~ 0.68",
        vehicle_mix={
            VehicleType.TWO_WHEELER: 0.60,
            VehicleType.AUTO: 0.15,
            VehicleType.CAR: 0.20,
            VehicleType.BUS: 0.05,
        },
        base_arrival_rate=0.225,
        tidal_flow=False,
    ),
}


@dataclass(frozen=True)
class SimulationConfig:
    """Full simulation experiment parameters."""
    dt: float = 1.0
    warmup_seconds: int = 300
    evolution_episode_seconds: int = 900
    evaluation_episode_seconds: int = 3600
    pcu_map: Dict[VehicleType, float] = field(default_factory=lambda: dict(DEFAULT_PCU_MAP))
    signal: SignalTimingConfig = field(default_factory=SignalTimingConfig)
    train_seed_range: Tuple[int, int] = (0, 100)
    test_seed_range: Tuple[int, int] = (1000, 1030)


DEFAULT_SIM_CONFIG = SimulationConfig()
