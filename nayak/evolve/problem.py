"""Pymoo optimization problem definition for multi-objective signal evolution."""

from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np
from pymoo.core.problem import ElementwiseProblem, Problem

from nayak.config import SCENARIOS, DEFAULT_SIM_CONFIG
from nayak.controllers.fuzzy_controller import EvolvedFuzzyController
from nayak.evolve.genome import GENOME_LENGTH
from nayak.sim.runner import run_episode


def evaluate_single_genome(
    genome: np.ndarray,
    scenarios: Sequence[str],
    train_seeds: Sequence[int],
    episode_seconds: int = 900,
    warmup_seconds: int = 300,
) -> Tuple[float, float, float]:
    """Simulate a genome on all training scenarios and seeds, returning the 3 objectives.

    Objectives (all minimized):
        f1: Mean wait time (non-ambulance vehicles)
        f2: 95th percentile wait time (fairness)
        f3: Mean ambulance delay
    """
    controller = EvolvedFuzzyController(genome)

    f1_vals: List[float] = []
    f2_vals: List[float] = []
    f3_vals: List[float] = []

    for sc_name in scenarios:
        sc = SCENARIOS[sc_name]
        for seed in train_seeds:
            controller.reset()
            m, _ = run_episode(
                controller=controller,
                scenario=sc,
                episode_seconds=episode_seconds,
                warmup_seconds=warmup_seconds,
                seed=seed,
            )
            f1_vals.append(m.mean_wait)
            f2_vals.append(m.wait_p95)
            f3_vals.append(m.mean_ambulance_delay)

    return (
        float(np.mean(f1_vals)),
        float(np.mean(f2_vals)),
        float(np.mean(f3_vals)),
    )


class TrafficSignalEvolutionProblem(Problem):
    """Pymoo Problem for evolving fuzzy membership functions and rule weights."""

    def __init__(
        self,
        train_seeds: Sequence[int] = (42, 57),
        episode_seconds: int = 900,
        warmup_seconds: int = 300,
        scenarios: Sequence[str] = ("normal", "rush_hour", "two_wheeler_heavy"),
        n_workers: int = 1,
        **kwargs,
    ) -> None:
        """Initialize the multi-objective problem.

        Args:
            train_seeds: Sequence of seeds strictly within train_seed_range [0, 99].
            episode_seconds: Duration of each evaluation episode.
            warmup_seconds: Warm-up period excluded from objective calculation.
            scenarios: Scenario names to evaluate.
            n_workers: Parallel worker pool size if external runner is not supplied.
        """
        super().__init__(
            n_var=GENOME_LENGTH,
            n_obj=3,
            n_ieq_constr=0,
            xl=0.0,
            xu=1.0,
            **kwargs,
        )
        self.train_seeds = list(train_seeds)
        self.episode_seconds = episode_seconds
        self.warmup_seconds = warmup_seconds
        self.scenarios = list(scenarios)
        self.n_workers = n_workers

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Vectorized or batch evaluation of population matrix x of shape (N, 44)."""
        pop_size = len(x)
        f = np.zeros((pop_size, 3), dtype=float)

        if self.n_workers > 1:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=self.n_workers)(
                delayed(evaluate_single_genome)(
                    x[i],
                    self.scenarios,
                    self.train_seeds,
                    self.episode_seconds,
                    self.warmup_seconds,
                )
                for i in range(pop_size)
            )
            for i, res in enumerate(results):
                f[i] = res
        else:
            for i in range(pop_size):
                f[i] = evaluate_single_genome(
                    x[i],
                    self.scenarios,
                    self.train_seeds,
                    self.episode_seconds,
                    self.warmup_seconds,
                )

        out["F"] = f
