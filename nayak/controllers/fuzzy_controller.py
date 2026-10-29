"""Fuzzy logic traffic signal controllers."""

from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

from nayak.config import PhaseState
from nayak.controllers.base import BaseController, ControllerAction
from nayak.fuzzy.engine import MamdaniFIS
from nayak.fuzzy.rulebase import (
    create_default_input_variables,
    create_default_output_variable,
    create_fixed_rule_base,
)

if TYPE_CHECKING:
    from nayak.sim.intersection import IntersectionState


class FuzzySignalController(BaseController):
    """Signal controller operating on Mamdani Fuzzy Inference System decisions.

    Decision Protocol (Section 3):
    - Controller is consulted when min_green is reached (default 8s) and every 4s thereafter.
    - If extension_seconds < 2.0s, switch phase (via clearance yellow/all-red).
    - Otherwise, hold green and consult again at the next 4s check.
    - Hard switch forced at max_green (default 60s).
    """

    def __init__(
        self,
        fis: MamdaniFIS,
        name: str = "FuzzyController",
        min_green: int = 8,
        max_green: int = 60,
        consult_interval: int = 4,
        extension_threshold: float = 2.0,
    ) -> None:
        self.fis = fis
        self._name = name
        self.min_green = min_green
        self.max_green = max_green
        self.consult_interval = consult_interval
        self.extension_threshold = extension_threshold

        self._last_consulted_elapsed: Optional[int] = None
        self._last_extension: float = 0.0

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        """Reset internal consultation timing memory."""
        self._last_consulted_elapsed = None
        self._last_extension = 0.0

    def decide(self, state: IntersectionState) -> ControllerAction:
        """Determine whether to extend current green phase or switch."""
        if state.phase_state != PhaseState.GREEN:
            self._last_consulted_elapsed = None
            return ControllerAction.STAY

        # Force switch at max green
        if state.green_elapsed >= self.max_green:
            self._last_consulted_elapsed = None
            return ControllerAction.SWITCH

        # Hold green before min_green
        if state.green_elapsed < self.min_green:
            return ControllerAction.STAY

        # Check if consultation interval has arrived
        # Consult at green_elapsed == min_green, and every consult_interval (4s) thereafter
        should_consult = (
            self._last_consulted_elapsed is None
            or (state.green_elapsed - self._last_consulted_elapsed) >= self.consult_interval
        )

        if not should_consult:
            return ControllerAction.STAY

        # Prepare crisp inputs for fuzzy inference
        crisp_inputs: Dict[str, float] = {
            "q_green": state.q_green,
            "q_red": state.q_red,
            "bus_share_red": state.bus_share_red,
            "two_wheeler_share_green": state.two_wheeler_share_green,
            "green_elapsed": float(state.green_elapsed),
            "ambulance_red": float(state.ambulance_red),
            "ambulance_green": float(state.ambulance_green),
        }

        extension = self.fis.infer(crisp_inputs)
        self._last_consulted_elapsed = state.green_elapsed
        self._last_extension = extension

        if extension < self.extension_threshold:
            self._last_consulted_elapsed = None
            return ControllerAction.SWITCH

        return ControllerAction.STAY


class HandDesignedFuzzyController(FuzzySignalController):
    """Hand-designed fuzzy baseline using sensible default membership functions and unit rule weights."""

    def __init__(
        self,
        min_green: int = 8,
        max_green: int = 60,
        consult_interval: int = 4,
        extension_threshold: float = 2.0,
    ) -> None:
        inputs = create_default_input_variables()
        output = create_default_output_variable()
        rules = create_fixed_rule_base()
        fis = MamdaniFIS(input_variables=inputs, output_variable=output, rules=rules, default_output=0.0)

        super().__init__(
            fis=fis,
            name="HandDesignedFuzzy",
            min_green=min_green,
            max_green=max_green,
            consult_interval=consult_interval,
            extension_threshold=extension_threshold,
        )


class EvolvedFuzzyController(FuzzySignalController):
    """Fuzzy signal controller parameterized by an evolved genome."""

    def __init__(
        self,
        genome: Sequence[float],
        name: str = "EvolvedFuzzy",
        min_green: int = 8,
        max_green: int = 60,
        consult_interval: int = 4,
        extension_threshold: float = 2.0,
    ) -> None:
        from nayak.evolve.genome import decode_genome

        self.genome = list(genome)
        inputs, output, rules = decode_genome(self.genome)
        fis = MamdaniFIS(input_variables=inputs, output_variable=output, rules=rules, default_output=0.0)

        super().__init__(
            fis=fis,
            name=name,
            min_green=min_green,
            max_green=max_green,
            consult_interval=consult_interval,
            extension_threshold=extension_threshold,
        )

