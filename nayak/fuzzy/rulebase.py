"""Fixed rule base definition and default membership configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

from nayak.fuzzy.membership import TrapezoidalMF, TriangularMF
from nayak.fuzzy.variables import LinguisticVariable


@dataclass
class FuzzyRule:
    """Mamdani fuzzy IF-THEN rule with antecedent conjunction and rule weighting.

    Attributes:
        antecedents: Mapping of variable name to required linguistic term or crisp value.
        consequent: Output linguistic term (e.g. 'ZERO', 'SHORT', 'MEDIUM', 'LONG').
        weight: Importance weight in [0.0, 1.0].
    """
    antecedents: Dict[str, Union[str, int]]
    consequent: str
    weight: float = 1.0

    def compute_firing_strength(
        self,
        fuzzified_inputs: Dict[str, Dict[str, float]],
    ) -> float:
        """Compute Mamdani AND conjunction (min-operator) scaled by rule weight."""
        if not self.antecedents:
            return 0.0

        degrees: List[float] = []
        for var_name, term_req in self.antecedents.items():
            if var_name not in fuzzified_inputs:
                return 0.0
            term_key = str(term_req)
            degree = fuzzified_inputs[var_name].get(term_key, 0.0)
            degrees.append(degree)

        min_degree = min(degrees) if degrees else 0.0
        return float(self.weight * min_degree)

    def to_readable_string(self) -> str:
        """Return human-readable IF-THEN string representation."""
        ant_strs = []
        for var, term in self.antecedents.items():
            if isinstance(term, int) or term in ("0", "1"):
                ant_strs.append(f"{var} is {term}")
            else:
                ant_strs.append(f"{var} is {term}")
        ant_clause = " AND ".join(ant_strs)
        return f"IF {ant_clause} THEN extension is {self.consequent} (weight={self.weight:.2f})"


def create_default_input_variables() -> Dict[str, LinguisticVariable]:
    """Instantiate the 7 standard controller input linguistic variables with default MF shapes."""
    return {
        "q_green": LinguisticVariable(
            name="q_green",
            universe_range=(0.0, 50.0),
            terms={
                "LOW": TrapezoidalMF(0.0, 0.0, 5.0, 15.0),
                "MEDIUM": TriangularMF(10.0, 20.0, 30.0),
                "HIGH": TrapezoidalMF(25.0, 35.0, 50.0, 50.0),
            },
        ),
        "q_red": LinguisticVariable(
            name="q_red",
            universe_range=(0.0, 50.0),
            terms={
                "LOW": TrapezoidalMF(0.0, 0.0, 5.0, 15.0),
                "MEDIUM": TriangularMF(10.0, 20.0, 30.0),
                "HIGH": TrapezoidalMF(25.0, 35.0, 50.0, 50.0),
            },
        ),
        "bus_share_red": LinguisticVariable(
            name="bus_share_red",
            universe_range=(0.0, 1.0),
            terms={
                "LOW": TrapezoidalMF(0.0, 0.0, 0.10, 0.25),
                "MEDIUM": TriangularMF(0.15, 0.35, 0.55),
                "HIGH": TrapezoidalMF(0.45, 0.65, 1.0, 1.0),
            },
        ),
        "two_wheeler_share_green": LinguisticVariable(
            name="two_wheeler_share_green",
            universe_range=(0.0, 1.0),
            terms={
                "LOW": TrapezoidalMF(0.0, 0.0, 0.20, 0.35),
                "MEDIUM": TriangularMF(0.25, 0.50, 0.70),
                "HIGH": TrapezoidalMF(0.60, 0.75, 1.0, 1.0),
            },
        ),
        "green_elapsed": LinguisticVariable(
            name="green_elapsed",
            universe_range=(0.0, 60.0),
            terms={
                "LOW": TrapezoidalMF(0.0, 0.0, 12.0, 22.0),
                "MEDIUM": TriangularMF(18.0, 30.0, 42.0),
                "HIGH": TrapezoidalMF(38.0, 48.0, 60.0, 60.0),
            },
        ),
        "ambulance_red": LinguisticVariable(
            name="ambulance_red",
            universe_range=(0.0, 1.0),
            terms={},
            crisp_binary=True,
        ),
        "ambulance_green": LinguisticVariable(
            name="ambulance_green",
            universe_range=(0.0, 1.0),
            terms={},
            crisp_binary=True,
        ),
    }


def create_default_output_variable() -> LinguisticVariable:
    """Instantiate the signal extension output variable with default terms."""
    return LinguisticVariable(
        name="extension",
        universe_range=(0.0, 15.0),
        terms={
            "ZERO": TrapezoidalMF(0.0, 0.0, 0.5, 2.0),
            "SHORT": TriangularMF(1.5, 4.0, 7.0),
            "MEDIUM": TriangularMF(5.0, 8.5, 12.0),
            "LONG": TrapezoidalMF(10.0, 13.0, 15.0, 15.0),
        },
        resolution=151,
    )


def create_fixed_rule_base() -> List[FuzzyRule]:
    """Construct the fixed 18-rule base governing mixed-traffic extensions and emergency priority."""
    return [
        # Emergency priority overrides
        FuzzyRule({"ambulance_red": 1}, consequent="ZERO", weight=1.0),
        FuzzyRule({"ambulance_green": 1}, consequent="LONG", weight=1.0),
        FuzzyRule({"ambulance_green": 1, "ambulance_red": 1}, consequent="SHORT", weight=1.0),

        # Queue dominance rules
        FuzzyRule({"q_green": "LOW", "q_red": "HIGH"}, consequent="ZERO", weight=1.0),
        FuzzyRule({"q_green": "HIGH", "q_red": "LOW"}, consequent="LONG", weight=1.0),
        FuzzyRule({"q_green": "HIGH", "q_red": "HIGH"}, consequent="SHORT", weight=1.0),
        FuzzyRule({"q_green": "LOW", "q_red": "LOW"}, consequent="SHORT", weight=1.0),
        FuzzyRule({"q_green": "MEDIUM", "q_red": "MEDIUM"}, consequent="MEDIUM", weight=1.0),
        FuzzyRule({"q_green": "MEDIUM", "q_red": "HIGH"}, consequent="SHORT", weight=1.0),
        FuzzyRule({"q_green": "HIGH", "q_red": "MEDIUM"}, consequent="MEDIUM", weight=1.0),
        FuzzyRule({"q_green": "LOW", "q_red": "MEDIUM"}, consequent="ZERO", weight=1.0),

        # Mixed-traffic vehicle compositions
        FuzzyRule({"bus_share_red": "HIGH", "q_red": "HIGH"}, consequent="ZERO", weight=1.0),
        FuzzyRule({"bus_share_red": "HIGH", "q_red": "MEDIUM"}, consequent="SHORT", weight=1.0),
        FuzzyRule({"two_wheeler_share_green": "HIGH", "q_green": "MEDIUM"}, consequent="MEDIUM", weight=1.0),
        FuzzyRule({"two_wheeler_share_green": "HIGH", "q_red": "LOW"}, consequent="LONG", weight=1.0),

        # Green duration aging
        FuzzyRule({"green_elapsed": "HIGH", "q_red": "HIGH"}, consequent="ZERO", weight=1.0),
        FuzzyRule({"green_elapsed": "HIGH", "q_red": "MEDIUM"}, consequent="SHORT", weight=1.0),
        FuzzyRule({"green_elapsed": "LOW", "q_green": "HIGH"}, consequent="LONG", weight=1.0),
    ]
