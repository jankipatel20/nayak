"""Genome encoding and decoding for fuzzy membership shapes and rule weights."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np

from nayak.fuzzy.membership import TrapezoidalMF, TriangularMF
from nayak.fuzzy.rulebase import (
    FuzzyRule,
    create_default_input_variables,
    create_default_output_variable,
    create_fixed_rule_base,
)
from nayak.fuzzy.variables import LinguisticVariable

# Number of cut-point genes per continuous input variable
NUM_INPUT_CUTPOINTS = 4
# Number of cut-point genes for 4-term output extension
NUM_OUTPUT_CUTPOINTS = 6
# Fixed number of rules
NUM_RULES = 18

CONTINUOUS_INPUT_NAMES = [
    "q_green",
    "q_red",
    "bus_share_red",
    "two_wheeler_share_green",
    "green_elapsed",
]

# Total genome length: 5 inputs * 4 + 1 output * 6 + 18 rules = 44 genes
GENOME_LENGTH = len(CONTINUOUS_INPUT_NAMES) * NUM_INPUT_CUTPOINTS + NUM_OUTPUT_CUTPOINTS + NUM_RULES


def _decode_variable_cutpoints(
    genes: np.ndarray,
    min_val: float,
    max_val: float,
    min_spacing: float,
) -> np.ndarray:
    """Sort normalized genes and map with guaranteed minimum spacing into natural units."""
    n = len(genes)
    span = max_val - min_val
    required_space = (n + 1) * min_spacing
    usable_span = max(1e-4, span - required_space)

    # Sort genes to guarantee strict monotonic ordering
    sorted_norm = np.sort(np.clip(genes, 0.0, 1.0))

    # Map with cumulative spacing offsets
    natural_points = np.zeros(n, dtype=float)
    for i in range(n):
        offset = (i + 1) * min_spacing
        natural_points[i] = min_val + offset + sorted_norm[i] * usable_span

    return natural_points


def decode_genome(
    genome: Sequence[float],
) -> Tuple[Dict[str, LinguisticVariable], LinguisticVariable, List[FuzzyRule]]:
    """Decode a real-valued genome in [0, 1]^44 into valid FIS components.

    Guarantees:
    - Strictly monotonic, non-degenerate membership function vertices.
    - Continuous coverage across each variable's universe.
    - Valid rule weights in [0, 1].

    Args:
        genome: Sequence of 44 real numbers in [0.0, 1.0].

    Returns:
        Tuple of (input_variables dict, output_variable, rules list).
    """
    g = np.asarray(genome, dtype=float)
    if len(g) != GENOME_LENGTH:
        raise ValueError(f"Expected genome length {GENOME_LENGTH}, got {len(g)}")

    # 1. Decode continuous input variables
    inputs: Dict[str, LinguisticVariable] = {}
    base_defaults = create_default_input_variables()

    gene_idx = 0
    for name in CONTINUOUS_INPUT_NAMES:
        var_def = base_defaults[name]
        min_v, max_v = var_def.min_val, var_def.max_val
        span = max_v - min_v
        min_spacing = span * 0.02  # at least 2% separation

        sub_genes = g[gene_idx : gene_idx + NUM_INPUT_CUTPOINTS]
        gene_idx += NUM_INPUT_CUTPOINTS

        knots = _decode_variable_cutpoints(sub_genes, min_v, max_v, min_spacing)
        c1, c2, c3, c4 = knots[0], knots[1], knots[2], knots[3]

        # LOW: Trapezoidal left-shoulder
        low_mf = TrapezoidalMF(min_v, min_v, c1, c2)
        # MEDIUM: Triangular centered between c2 and c3
        med_mf = TriangularMF(c1, (c2 + c3) / 2.0, c4)
        # HIGH: Trapezoidal right-shoulder
        high_mf = TrapezoidalMF(c3, c4, max_v, max_v)

        inputs[name] = LinguisticVariable(
            name=name,
            universe_range=(min_v, max_v),
            terms={"LOW": low_mf, "MEDIUM": med_mf, "HIGH": high_mf},
            resolution=var_def.resolution,
        )

    # Add crisp binary variables (unchanged)
    inputs["ambulance_red"] = base_defaults["ambulance_red"]
    inputs["ambulance_green"] = base_defaults["ambulance_green"]

    # 2. Decode output variable (extension)
    out_def = create_default_output_variable()
    o_min, o_max = out_def.min_val, out_def.max_val
    o_spacing = (o_max - o_min) * 0.02

    sub_genes_out = g[gene_idx : gene_idx + NUM_OUTPUT_CUTPOINTS]
    gene_idx += NUM_OUTPUT_CUTPOINTS
    k = _decode_variable_cutpoints(sub_genes_out, o_min, o_max, o_spacing)
    k1, k2, k3, k4, k5, k6 = k[0], k[1], k[2], k[3], k[4], k[5]

    zero_mf = TrapezoidalMF(o_min, o_min, k1, k2)
    short_mf = TriangularMF(k1, (k2 + k3) / 2.0, k4)
    med_mf = TriangularMF(k3, (k4 + k5) / 2.0, k6)
    long_mf = TrapezoidalMF(k5, k6, o_max, o_max)

    output = LinguisticVariable(
        name="extension",
        universe_range=(o_min, o_max),
        terms={
            "ZERO": zero_mf,
            "SHORT": short_mf,
            "MEDIUM": med_mf,
            "LONG": long_mf,
        },
        resolution=out_def.resolution,
    )

    # 3. Decode rule weights
    rule_weights = g[gene_idx : gene_idx + NUM_RULES]
    base_rules = create_fixed_rule_base()
    rules: List[FuzzyRule] = []
    for r, w in zip(base_rules, rule_weights):
        rules.append(FuzzyRule(antecedents=r.antecedents, consequent=r.consequent, weight=float(np.clip(w, 0.0, 1.0))))

    return inputs, output, rules


def encode_default_hand_designed() -> np.ndarray:
    """Construct an initial genome vector representing HandDesignedFuzzy parameters."""
    genome = np.zeros(GENOME_LENGTH, dtype=float)

    # Hand-picked normalized cut points corresponding roughly to default knots
    base_defaults = create_default_input_variables()
    gene_idx = 0

    for name in CONTINUOUS_INPUT_NAMES:
        var = base_defaults[name]
        min_v, max_v = var.min_val, var.max_val
        span = max_v - min_v
        # Default knots: low c, low d, high a, high b
        # e.g. for q_green [0, 50]: c1=5, c2=15, c3=25, c4=35 -> (c - min)/span
        if name in ("q_green", "q_red"):
            pts = [5.0, 15.0, 25.0, 35.0]
        elif name == "bus_share_red":
            pts = [0.10, 0.25, 0.45, 0.65]
        elif name == "two_wheeler_share_green":
            pts = [0.20, 0.35, 0.60, 0.75]
        elif name == "green_elapsed":
            pts = [12.0, 22.0, 38.0, 48.0]
        else:
            pts = [span * 0.2, span * 0.4, span * 0.6, span * 0.8]

        norm_pts = [(p - min_v) / span for p in pts]
        genome[gene_idx : gene_idx + NUM_INPUT_CUTPOINTS] = norm_pts
        gene_idx += NUM_INPUT_CUTPOINTS

    # Output extension [0, 15]: 0.5, 2.0, 4.0, 7.0, 10.0, 13.0
    out_pts = [0.5, 2.0, 4.5, 7.0, 10.0, 13.0]
    genome[gene_idx : gene_idx + NUM_OUTPUT_CUTPOINTS] = [p / 15.0 for p in out_pts]
    gene_idx += NUM_OUTPUT_CUTPOINTS

    # All rule weights = 1.0
    genome[gene_idx : gene_idx + NUM_RULES] = 1.0

    return np.clip(genome, 0.0, 1.0)
