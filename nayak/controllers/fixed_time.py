"""Fixed-time signal controller with equal green splits."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nayak.config import PhaseState
from nayak.controllers.base import BaseController, ControllerAction

if TYPE_CHECKING:
    from nayak.sim.intersection import IntersectionState


class FixedTimeController(BaseController):
    """Fixed-time signal controller operating on predetermined green durations.

    Default: 30 s green per phase with clearance lost time (yellow + all-red).
    """

    def __init__(self, green_seconds: int = 30) -> None:
        """Initialize fixed time controller.

        Args:
            green_seconds: Target green duration in seconds for each phase.
        """
        self.green_seconds = green_seconds

    @property
    def name(self) -> str:
        return f"FixedTime({self.green_seconds}s)"

    def decide(self, state: IntersectionState) -> ControllerAction:
        """Switch phase once green_elapsed reaches the configured green time."""
        if state.phase_state == PhaseState.GREEN:
            if state.green_elapsed >= self.green_seconds:
                return ControllerAction.SWITCH
        return ControllerAction.STAY
