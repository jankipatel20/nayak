"""Unit and integration tests for genome encoding, NSGA-II problem, and Pareto selection."""

import numpy as np
import pytest

from nayak.evolve.genome import (
    GENOME_LENGTH,
    decode_genome,
    encode_default_hand_designed,
)
from nayak.evolve.problem import TrafficSignalEvolutionProblem, evaluate_single_genome
from nayak.evolve.select import select_pareto_representatives
from nayak.fuzzy.engine import MamdaniFIS


def test_genome_length() -> None:
    """Verify standard genome length is exactly 44 genes."""
    assert GENOME_LENGTH == 44


def test_decode_random_genome_always_valid() -> None:
    """Verify that any randomly generated genome in [0, 1]^44 always decodes to valid, ordered MFs."""
    rng = np.random.default_rng(999)

    for _ in range(50):
        random_genome = rng.random(GENOME_LENGTH)
        inputs, output, rules = decode_genome(random_genome)

        # 1. Verify inputs monotonic ordering and knot constraints
        for name, var in inputs.items():
            if var.crisp_binary:
                continue
            terms = var.terms
            assert "LOW" in terms and "MEDIUM" in terms and "HIGH" in terms

            low_pts = terms["LOW"].breakpoints()
            assert low_pts[0] <= low_pts[1] <= low_pts[2] <= low_pts[3]

            med_pts = terms["MEDIUM"].breakpoints()
            assert med_pts[0] <= med_pts[1] <= med_pts[2]

            high_pts = terms["HIGH"].breakpoints()
            assert high_pts[0] <= high_pts[1] <= high_pts[2] <= high_pts[3]

        # 2. Verify output extension terms
        out_terms = output.terms
        assert "ZERO" in out_terms and "SHORT" in out_terms and "MEDIUM" in out_terms and "LONG" in out_terms
        for term_name, mf in out_terms.items():
            pts = mf.breakpoints()
            for i in range(len(pts) - 1):
                assert pts[i] <= pts[i + 1]

        # 3. Verify rule weights
        assert len(rules) == 18
        for r in rules:
            assert 0.0 <= r.weight <= 1.0

        # 4. Verify inference runs without numerical exceptions
        fis = MamdaniFIS(inputs, output, rules)
        val = fis.infer({
            "q_green": 15.0,
            "q_red": 25.0,
            "bus_share_red": 0.3,
            "two_wheeler_share_green": 0.5,
            "green_elapsed": 20.0,
            "ambulance_red": 0.0,
            "ambulance_green": 0.0,
        })
        assert 0.0 <= val <= 15.0


def test_hand_designed_encoding_decoding() -> None:
    """Verify default hand-designed genome decodes and executes cleanly."""
    genome = encode_default_hand_designed()
    assert len(genome) == GENOME_LENGTH
    assert np.all(genome >= 0.0) and np.all(genome <= 1.0)

    inputs, output, rules = decode_genome(genome)
    fis = MamdaniFIS(inputs, output, rules)
    ext = fis.infer({"q_green": 30.0, "q_red": 5.0, "green_elapsed": 10.0})
    assert ext > 5.0


def test_pareto_representative_selection() -> None:
    """Verify extraction of Fast, Fair, Rescue, and Balanced knee picks from an objective front."""
    # 4 synthetic Pareto solutions: [f1, f2, f3]
    # Point 0: best f1 (lowest mean wait)
    # Point 1: best f2 (lowest p95 wait)
    # Point 2: best f3 (lowest ambulance delay)
    # Point 3: balanced middle
    F = np.array([
        [15.0, 50.0, 20.0],
        [25.0, 30.0, 25.0],
        [30.0, 60.0, 5.0],
        [18.0, 38.0, 12.0],
    ])
    X = np.ones((4, GENOME_LENGTH))

    reps = select_pareto_representatives(F, X)
    assert reps["Nayak-Fast"]["pareto_index"] == 0
    assert reps["Nayak-Fair"]["pareto_index"] == 1
    assert reps["Nayak-Rescue"]["pareto_index"] == 2
    # Point 3 is the balanced compromise
    assert reps["Nayak-Balanced"]["pareto_index"] == 3


def test_end_to_end_smoke_evolution() -> None:
    """Smoke test: 2 generations of tiny population to verify entire NSGA-II pipeline executes."""
    problem = TrafficSignalEvolutionProblem(
        train_seeds=(1,),
        episode_seconds=60,
        warmup_seconds=20,
        scenarios=("normal",),
        n_workers=1,
    )

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    algorithm = NSGA2(pop_size=4)
    res = minimize(problem, algorithm, ("n_gen", 2), seed=42, verbose=False)

    assert res.F is not None
    assert len(res.F) > 0
    assert res.X is not None
    assert len(res.X) > 0
