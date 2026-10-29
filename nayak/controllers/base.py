"""Base controller interface and action definition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nayak.sim.intersection import IntersectionState


class ControllerAction(IntEnum):
    """Signal control action decisions."""
    STAY = 0    # Hold the current green phase
    SWITCH = 1  # Request phase transition to yellow/all-red sequence


class BaseController(ABC):
    """Abstract base class for all traffic signal controllers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier of the controller strategy."""
        pass

    @abstractmethod
    def decide(self, state: IntersectionState) -> ControllerAction:
        """Inspect intersection state and return STAY or SWITCH.

        Args:
            state: Current state of queues, elapsed green, and approaching vehicles.

        Returns:
            ControllerAction.STAY to maintain green, or ControllerAction.SWITCH to transition.
        """
        pass

    def reset(self) -> None:
        """Reset any internal controller memory between runs if applicable."""
        pass
