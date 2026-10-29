"""Fuzzy linguistic variables and universe definitions."""

from __future__ import annotations

from typing import Dict, Optional, Sequence
import numpy as np

from nayak.fuzzy.membership import MembershipFunction, TrapezoidalMF, TriangularMF


class LinguisticVariable:
    """Represents a fuzzy input or output variable with named linguistic terms."""

    def __init__(
        self,
        name: str,
        universe_range: tuple[float, float],
        terms: Dict[str, MembershipFunction],
        resolution: int = 151,
        crisp_binary: bool = False,
    ) -> None:
        """Initialize a linguistic variable.

        Args:
            name: Identifier of the variable (e.g. 'q_green').
            universe_range: (min_value, max_value) bounds.
            terms: Dictionary mapping linguistic terms (e.g. 'LOW') to MembershipFunctions.
            resolution: Number of discretization steps for defuzzification integrals.
            crisp_binary: If True, indicates a crisp 0/1 indicator (e.g. ambulance present).
        """
        self.name = name
        self.min_val = float(universe_range[0])
        self.max_val = float(universe_range[1])
        self.terms = terms
        self.resolution = resolution
        self.crisp_binary = crisp_binary

        # Precompute discretized universe grid
        self.universe_grid = np.linspace(self.min_val, self.max_val, self.resolution)

    def fuzzify(self, crisp_value: float) -> Dict[str, float]:
        """Evaluate membership degrees for all terms at a given crisp input."""
        if self.crisp_binary:
            # Crisp indicator: value is 0 or 1
            val = 1 if crisp_value >= 0.5 else 0
            return {"0": 1.0 if val == 0 else 0.0, "1": 1.0 if val == 1 else 0.0}

        # Clamp input into universe range for safety
        clamped = float(np.clip(crisp_value, self.min_val, self.max_val))
        return {term_name: float(mf.evaluate(clamped)) for term_name, mf in self.terms.items()}

    def get_term_array(self, term_name: str) -> np.ndarray:
        """Evaluate a term's membership across the precomputed universe grid."""
        mf = self.terms[term_name]
        return np.asarray(mf.evaluate(self.universe_grid), dtype=float)

    def __repr__(self) -> str:
        return f"LinguisticVariable({self.name}, range=[{self.min_val}, {self.max_val}], terms={list(self.terms.keys())})"
