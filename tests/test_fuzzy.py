"""Comprehensive unit tests for the NAYAK fuzzy logic engine."""

import numpy as np
import pytest

from nayak.controllers.fuzzy_controller import HandDesignedFuzzyController
from nayak.controllers.base import ControllerAction
from nayak.fuzzy.engine import MamdaniFIS
from nayak.fuzzy.membership import TrapezoidalMF, TriangularMF
from nayak.fuzzy.rulebase import (
    FuzzyRule,
    create_default_input_variables,
    create_default_output_variable,
    create_fixed_rule_base,
)
from nayak.fuzzy.variables import LinguisticVariable
from nayak.sim.intersection import IntersectionState
from nayak.config import PhaseState, SignalPhase, Approach


def test_membership_functions_known_points() -> None:
    """Verify triangular and trapezoidal membership evaluations at known boundary and peak points."""
    # Triangular: a=10, b=20, c=30
    tri = TriangularMF(10.0, 20.0, 30.0)
    assert tri.evaluate(5.0) == 0.0
    assert tri.evaluate(10.0) == 0.0
    assert tri.evaluate(15.0) == pytest.approx(0.5, abs=1e-6)
    assert tri.evaluate(20.0) == 1.0
    assert tri.evaluate(25.0) == pytest.approx(0.5, abs=1e-6)
    assert tri.evaluate(30.0) == 0.0
    assert tri.evaluate(35.0) == 0.0

    # Trapezoidal: a=10, b=20, c=30, d=40
    trap = TrapezoidalMF(10.0, 20.0, 30.0, 40.0)
    assert trap.evaluate(5.0) == 0.0
    assert trap.evaluate(10.0) == 0.0
    assert trap.evaluate(15.0) == pytest.approx(0.5, abs=1e-6)
    assert trap.evaluate(20.0) == 1.0
    assert trap.evaluate(25.0) == 1.0  # plateau
    assert trap.evaluate(30.0) == 1.0
    assert trap.evaluate(35.0) == pytest.approx(0.5, abs=1e-6)
    assert trap.evaluate(40.0) == 0.0
    assert trap.evaluate(45.0) == 0.0

    # Left shoulder trapezoid: a=0, b=0, c=5, d=10
    left_shoulder = TrapezoidalMF(0.0, 0.0, 5.0, 10.0)
    assert left_shoulder.evaluate(-2.0) == 1.0
    assert left_shoulder.evaluate(0.0) == 1.0
    assert left_shoulder.evaluate(3.0) == 1.0
    assert left_shoulder.evaluate(7.5) == pytest.approx(0.5, abs=1e-6)
    assert left_shoulder.evaluate(10.0) == 0.0


def test_centroid_known_symmetric_shape() -> None:
    """Verify that centroid defuzzification of a single symmetric triangular set gives its center."""
    out_var = LinguisticVariable(
        name="test_out",
        universe_range=(0.0, 10.0),
        terms={"CENTER": TriangularMF(2.0, 5.0, 8.0)},
        resolution=201,
    )
    in_var = LinguisticVariable(
        name="test_in",
        universe_range=(0.0, 10.0),
        terms={"ANY": TriangularMF(0.0, 5.0, 10.0)},
    )
    rule = FuzzyRule(antecedents={"test_in": "ANY"}, consequent="CENTER", weight=1.0)
    fis = MamdaniFIS(
        input_variables={"test_in": in_var},
        output_variable=out_var,
        rules=[rule],
    )

    result = fis.infer({"test_in": 5.0})
    # Centroid of symmetric triangle centered at 5.0 must be 5.0
    assert result == pytest.approx(5.0, abs=0.05)


def test_rule_weight_scaling() -> None:
    """Verify that rule weight scales firing strength proportionally."""
    rule1 = FuzzyRule(antecedents={"q_green": "HIGH"}, consequent="LONG", weight=1.0)
    rule_half = FuzzyRule(antecedents={"q_green": "HIGH"}, consequent="LONG", weight=0.5)

    fuzzified = {"q_green": {"HIGH": 0.8}}

    strength1 = rule1.compute_firing_strength(fuzzified)
    strength_half = rule_half.compute_firing_strength(fuzzified)

    assert strength1 == pytest.approx(0.8, abs=1e-6)
    assert strength_half == pytest.approx(0.4, abs=1e-6)


def test_no_rule_fires_fallback() -> None:
    """Verify that when no rule fires, the FIS gracefully returns the safe default output."""
    inputs = create_default_input_variables()
    output = create_default_output_variable()
    # A rule that will never fire on normal input values
    dummy_rule = FuzzyRule(antecedents={"q_green": "HIGH", "q_red": "HIGH"}, consequent="LONG", weight=0.0)
    fis = MamdaniFIS(
        input_variables=inputs,
        output_variable=output,
        rules=[dummy_rule],
        default_output=3.14,
    )

    result = fis.infer({"q_green": 0.0, "q_red": 0.0})
    assert result == pytest.approx(3.14, abs=1e-6)


def test_queue_dominance_sanity_check() -> None:
    """Verify domain sanity: high green queue + low red queue yields larger extension than the inverse."""
    inputs = create_default_input_variables()
    output = create_default_output_variable()
    rules = create_fixed_rule_base()
    fis = MamdaniFIS(inputs, output, rules)

    # State A: Green queue high (40 PCU), Red queue low (2 PCU)
    ext_high_green = fis.infer({
        "q_green": 40.0,
        "q_red": 2.0,
        "bus_share_red": 0.0,
        "two_wheeler_share_green": 0.0,
        "green_elapsed": 10.0,
        "ambulance_red": 0.0,
        "ambulance_green": 0.0,
    })

    # State B: Green queue low (2 PCU), Red queue high (40 PCU)
    ext_high_red = fis.infer({
        "q_green": 2.0,
        "q_red": 40.0,
        "bus_share_red": 0.0,
        "two_wheeler_share_green": 0.0,
        "green_elapsed": 10.0,
        "ambulance_red": 0.0,
        "ambulance_green": 0.0,
    })

    assert ext_high_green > ext_high_red + 5.0, (
        f"Expected high green extension ({ext_high_green:.2f}) >> high red extension ({ext_high_red:.2f})"
    )


def test_ambulance_priority_inference() -> None:
    """Verify emergency rules: ambulance on red yields minimum extension; ambulance on green yields long extension."""
    inputs = create_default_input_variables()
    output = create_default_output_variable()
    rules = create_fixed_rule_base()
    fis = MamdaniFIS(inputs, output, rules)

    # Ambulance waiting on RED
    ext_amb_red = fis.infer({
        "q_green": 20.0,
        "q_red": 10.0,
        "bus_share_red": 0.0,
        "two_wheeler_share_green": 0.0,
        "green_elapsed": 10.0,
        "ambulance_red": 1.0,
        "ambulance_green": 0.0,
    })
    # Should fire Rule 1 (ambulance_red is 1 -> extension is ZERO)
    assert ext_amb_red < 2.0, f"Expected near-zero extension for red ambulance, got {ext_amb_red}"

    # Ambulance waiting on GREEN
    ext_amb_green = fis.infer({
        "q_green": 10.0,
        "q_red": 20.0,
        "bus_share_red": 0.0,
        "two_wheeler_share_green": 0.0,
        "green_elapsed": 25.0,
        "ambulance_red": 0.0,
        "ambulance_green": 1.0,
    })
    # Should fire Rule 2 (ambulance_green is 1 -> extension is LONG)
    assert ext_amb_green > 10.0, f"Expected long extension for green ambulance, got {ext_amb_green}"


def test_hand_designed_fuzzy_controller_decision() -> None:
    """Verify that HandDesignedFuzzyController operates properly with IntersectionState."""
    controller = HandDesignedFuzzyController(min_green=8, max_green=60)

    # 1. Before min_green -> Must STAY
    state_early = IntersectionState(
        current_phase=SignalPhase.PHASE_NS,
        phase_state=PhaseState.GREEN,
        green_elapsed=4,
        time_in_state=4,
        q_green=0.0,
        q_red=30.0,
        bus_share_red=0.0,
        two_wheeler_share_green=0.0,
        ambulance_red=0,
        ambulance_green=0,
        queue_pcu={app: 0.0 for app in Approach},
        queue_count={app: 0 for app in Approach},
    )
    assert controller.decide(state_early) == ControllerAction.STAY

    # 2. At min_green (8s) with high red queue and empty green -> Must SWITCH
    state_switch = IntersectionState(
        current_phase=SignalPhase.PHASE_NS,
        phase_state=PhaseState.GREEN,
        green_elapsed=8,
        time_in_state=8,
        q_green=0.0,
        q_red=35.0,
        bus_share_red=0.5,
        two_wheeler_share_green=0.0,
        ambulance_red=0,
        ambulance_green=0,
        queue_pcu={app: 0.0 for app in Approach},
        queue_count={app: 0 for app in Approach},
    )
    assert controller.decide(state_switch) == ControllerAction.SWITCH
