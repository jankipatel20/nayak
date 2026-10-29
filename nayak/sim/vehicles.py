"""Vehicle representations and PCU modeling for mixed traffic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from nayak.config import Approach, VehicleType


@dataclass
class Vehicle:
    """Represents an individual vehicle in the microscopic simulation.

    Attributes:
        id: Unique identifier for tracking.
        vehicle_type: Category (two_wheeler, auto, car, bus, ambulance).
        approach: Origin approach (N, E, S, W).
        arrival_time: Simulation timestamp (seconds) when vehicle joined the approach.
        pcu: Passenger Car Unit road-space weight.
        departure_time: Simulation timestamp (seconds) when vehicle crossed the stop line.
    """
    id: int
    vehicle_type: VehicleType
    approach: Approach
    arrival_time: float
    pcu: float
    departure_time: Optional[float] = None

    @property
    def is_ambulance(self) -> bool:
        """Return True if this vehicle is an emergency priority vehicle."""
        return self.vehicle_type == VehicleType.AMBULANCE

    @property
    def departed(self) -> bool:
        """Return True if the vehicle has cleared the intersection."""
        return self.departure_time is not None

    def wait_time(self, current_time: float) -> float:
        """Calculate wait time so far or completed wait time upon departure.

        Args:
            current_time: Current simulation clock in seconds.

        Returns:
            Wait duration in seconds.
        """
        end_time = self.departure_time if self.departure_time is not None else current_time
        return max(0.0, end_time - self.arrival_time)
