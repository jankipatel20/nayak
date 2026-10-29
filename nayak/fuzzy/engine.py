"""Mamdani Fuzzy Inference System (FIS) core engine."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np

from nayak.fuzzy.rulebase import FuzzyRule
from nayak.fuzzy.variables import LinguisticVariable


class MamdaniFIS:
    """Hand-crafted vector-accelerated Mamdani Fuzzy Inference System.

    Features:
    - Min-operator for antecedent conjunction (AND).
    - Rule-weight scaling on individual firing strengths.
    - Max-operator for rule aggregation.
    - Centroid defuzzification over discretized output universe.
    - Safe default fallback handling when no rules fire.
    """

    def __init__(
        self,
        input_variables: Dict[str, LinguisticVariable],
        output_variable: LinguisticVariable,
        rules: Sequence[FuzzyRule],
        default_output: float = 0.0,
    ) -> None:
        """Initialize the FIS engine.

        Args:
            input_variables: Dict mapping variable names to LinguisticVariable definitions.
            output_variable: LinguisticVariable defining output extension terms and grid.
            rules: Sequence of FuzzyRule instances.
            default_output: Fallback output value if total firing strength is zero.
        """
        self.inputs = input_variables
        self.output = output_variable
        self.rules = list(rules)
        self.default_output = float(default_output)

        # Precompute output term membership arrays across the output grid for fast inference
        self._output_grid = self.output.universe_grid
        self._consequent_arrays: Dict[str, np.ndarray] = {
            term: self.output.get_term_array(term) for term in self.output.terms
        }

    def infer(self, crisp_inputs: Dict[str, float]) -> float:
        """Execute full Mamdani fuzzy inference pipeline.

        Args:
            crisp_inputs: Dictionary mapping input variable names to float observations.

        Returns:
            Defuzzified centroid extension in seconds.
        """
        # 1. Fuzzification
        fuzzified: Dict[str, Dict[str, float]] = {}
        for var_name, var_def in self.inputs.items():
            val = crisp_inputs.get(var_name, 0.0)
            fuzzified[var_name] = var_def.fuzzify(val)

        # 2. Rule evaluation (firing strength with rule weight scaling)
        # 3. Implication and aggregation (min clipping, max aggregation)
        aggregated = np.zeros_like(self._output_grid)
        total_firing_strength = 0.0

        for rule in self.rules:
            if rule.weight <= 1e-6:
                continue
            alpha = rule.compute_firing_strength(fuzzified)
            if alpha > 1e-6:
                total_firing_strength += alpha
                consequent_shape = self._consequent_arrays[rule.consequent]
                # Clipped consequent fuzzy set
                clipped = np.minimum(alpha, consequent_shape)
                aggregated = np.maximum(aggregated, clipped)

        # 4. Fallback handling when no rules fire
        area = np.sum(aggregated)
        if area < 1e-9:
            return self.default_output

        # 5. Centroid Defuzzification: sum(u * mu(u)) / sum(mu(u))
        centroid = float(np.sum(self._output_grid * aggregated) / area)
        return float(np.clip(centroid, self.output.min_val, self.output.max_val))
