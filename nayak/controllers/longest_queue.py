"""Actuated Longest-Queue-First signal controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nayak.config import PhaseState
from nayak.controllers.base import BaseController, ControllerAction

if TYPE_CHECKING:
    from nayak.sim.intersection import IntersectionState


class LongestQueueFirstController(BaseController):
    """Actuated signal controller that serves the approach direction with greater queue pressure.

    Decision Rule:
    - Holds green until min_green (8s).
    - If green_elapsed >= max_green (60s), forces SWITCH.
    - Between min_green and max_green, switches if red queue PCU exceeds green queue PCU
      by a threshold margin, or if an emergency vehicle is waiting on a red approach.
    """

    def __init__(self, queue_margin_pcu: float = 2.0, min_green: int = 8, max_green: int = 60) -> None:
        """Initialize LongestQueueFirst controller.

        Args:
            queue_margin_pcu: PCU margin by which red queue must exceed green queue to switch.
            min_green: Minimum green hold duration in seconds.
            max_green: Maximum green ceiling in seconds.
        """
        self.queue_margin_pcu = float(queue_margin_pcu)
        self.min_green = min_green
        self.max_green = max_green

    @property
    def name(self) -> str:
        return f"LongestQueueFirst(margin={self.queue_margin_pcu:.1f})"

    def decide(self, state: IntersectionState) -> ControllerAction:
        """Evaluate queue pressure and ambulance flags to decide STAY or SWITCH."""
        if state.phase_state != PhaseState.GREEN:
            return ControllerAction.STAY

        # Respect min_green constraint
        if state.green_elapsed < self.min_green:
            return ControllerAction.STAY

        # Enforce max_green ceiling
        if state.green_elapsed >= self.max_green:
            return ControllerAction.SWITCH

        # Emergency vehicle priority override
        if state.ambulance_red == 1 and state.ambulance_green == 0:
            return ControllerAction.SWITCH

        # Actuated queue comparison
        if state.q_red > state.q_green + self.queue_margin_pcu:
            return ControllerAction.SWITCH

        return ControllerAction.STAY
