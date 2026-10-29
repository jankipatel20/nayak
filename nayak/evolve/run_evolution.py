"""Main entry point for NSGA-II multi-objective genetic evolution."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Sequence
import numpy as np

# Ensure nayak package is importable
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.indicators.hv import Hypervolume
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

from nayak.evolve.genome import encode_default_hand_designed
from nayak.evolve.problem import TrafficSignalEvolutionProblem
from nayak.evolve.select import save_pareto_artifacts


class InitialPopulationWithHandDesigned(FloatRandomSampling):
    """Initial population sampler that injects the HandDesignedFuzzy genome as individual 0."""

    def _do(self, problem, n_samples, **kwargs):
        val = super()._do(problem, n_samples, **kwargs)
        # Inject hand-designed baseline as the first individual
        val[0] = encode_default_hand_designed()
        return val


class EvolutionLoggerCallback(Callback):
    """Callback for generation logging, hypervolume tracking, and checkpoint persistence."""

    def __init__(
        self,
        ref_point: np.ndarray,
        checkpoint_dir: Path,
        checkpoint_freq: int = 5,
    ) -> None:
        super().__init__()
        self.ref_point = ref_point
        self.hv = Hypervolume(ref_point=ref_point)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_freq = checkpoint_freq
        self.gen_history: List[dict] = []

    def notify(self, algorithm) -> None:
        gen = algorithm.n_gen
        pop = algorithm.pop
        F = pop.get("F")

        best_f1 = float(np.min(F[:, 0]))
        best_f2 = float(np.min(F[:, 1]))
        best_f3 = float(np.min(F[:, 2]))

        # Calculate hypervolume for non-dominated front respecting reference point
        valid_mask = np.all(F <= self.ref_point, axis=1)
        if np.any(valid_mask):
            hv_score = float(self.hv(F[valid_mask]))
        else:
            hv_score = 0.0

        log_entry = {
            "gen": gen,
            "best_f1": best_f1,
            "best_f2": best_f2,
            "best_f3": best_f3,
            "hypervolume": hv_score,
        }
        self.gen_history.append(log_entry)

        print(
            f"Gen {gen:3d} | Best f1 (Wait): {best_f1:6.2f}s | "
            f"Best f2 (p95): {best_f2:6.2f}s | "
            f"Best f3 (Rescue): {best_f3:6.2f}s | "
            f"Hypervolume: {hv_score:10.1f}",
            flush=True,
        )

        # Checkpoint serialization
        if gen % self.checkpoint_freq == 0:
            ckpt_path = self.checkpoint_dir / f"checkpoint_gen_{gen}.pkl"
            with open(ckpt_path, "wb") as f:
                pickle.dump(algorithm, f)


def run_evolution(
    pop_size: int = 40,
    n_gen: int = 30,
    n_workers: int = 4,
    train_seeds: Sequence[int] = (42, 57),
    quick: bool = False,
    resume: bool = False,
    results_dir: Optional[Path] = None,
) -> None:
    """Execute NSGA-II multi-objective evolution.

    Args:
        pop_size: Population size.
        n_gen: Number of generations.
        n_workers: Parallel worker count.
        train_seeds: Disjoint training seeds in range [0, 99].
        quick: Fast mode for debugging (pop=20, gens=10, 1 seed).
        resume: Resume from latest checkpoint in results/checkpoints/.
        results_dir: Root results output directory.
    """
    res_dir = results_dir or (ROOT_DIR / "results")
    ckpt_dir = res_dir / "checkpoints"

    if quick:
        print("\n>>> RUNNING IN --quick DEBUG MODE <<<")
        pop_size = 20
        n_gen = 10
        train_seeds = (42,)

    print("=" * 88)
    print("NAYAK: MULTI-OBJECTIVE NSGA-II SIGNAL CONTROLLER EVOLUTION")
    print(f"Population Size: {pop_size} | Generations: {n_gen} | Workers: {n_workers}")
    print(f"Training Seeds: {list(train_seeds)} (Disjoint from test seeds 1000..1029)")
    print(f"Objectives: [f1: Mean Wait (min), f2: p95 Wait (min), f3: Ambulance Delay (min)]")
    print("=" * 88)

    problem = TrafficSignalEvolutionProblem(
        train_seeds=train_seeds,
        episode_seconds=900,
        warmup_seconds=300,
        n_workers=n_workers,
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=InitialPopulationWithHandDesigned(),
        crossover=SBX(prob=0.9, eta=15.0),
        mutation=PM(prob=1.0 / problem.n_var, eta=20.0),
        eliminate_duplicates=True,
    )

    # Reference point chosen conservatively above plausible objective bounds
    ref_point = np.array([120.0, 250.0, 35.0])
    callback = EvolutionLoggerCallback(
        ref_point=ref_point,
        checkpoint_dir=ckpt_dir,
        checkpoint_freq=5,
    )

    # Resume from checkpoint if requested
    if resume and ckpt_dir.exists():
        checkpoints = sorted(ckpt_dir.glob("checkpoint_gen_*.pkl"))
        if checkpoints:
            latest_ckpt = checkpoints[-1]
            print(f"Resuming evolution from checkpoint: {latest_ckpt.name}")
            with open(latest_ckpt, "rb") as f:
                algorithm = pickle.load(f)

    # Run optimization
    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        callback=callback,
        seed=1,
        verbose=False,
    )

    # Extract non-dominated Pareto front
    opt = res.opt
    F_pareto = opt.get("F")
    X_pareto = opt.get("X")

    print("\n" + "=" * 88)
    print(f"EVOLUTION COMPLETED! Non-dominated Pareto Front size: {len(F_pareto)}")
    print("=" * 88)

    # Save artifacts and representative picks
    reps = save_pareto_artifacts(F=F_pareto, X=X_pareto, results_dir=res_dir)

    # Also save hypervolume log history
    hv_log_path = res_dir / "hypervolume_history.json"
    import json
    with open(hv_log_path, "w", encoding="utf-8") as f:
        json.dump(callback.gen_history, f, indent=2)

    print("\nRepresentative Selected Variants:")
    print("-" * 88)
    for name, data in reps.items():
        objs = data["objectives"]
        print(
            f"{name:<16} | Mean Wait: {objs['mean_wait']:6.2f}s | "
            f"p95 Wait: {objs['wait_p95']:6.2f}s | "
            f"Ambulance Delay: {objs['ambulance_delay']:6.2f}s"
        )
    print("-" * 88)
    print(f"Artifacts successfully saved to: {res_dir.relative_to(ROOT_DIR)}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NAYAK NSGA-II fuzzy traffic signal evolution.")
    parser.add_argument("--pop", type=int, default=40, help="Population size (default: 40)")
    parser.add_argument("--gens", type=int, default=30, help="Generations (default: 30)")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 8), help="Worker processes")
    parser.add_argument("--quick", action="store_true", help="Quick mode (pop=20, gens=10) for debugging")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    run_evolution(
        pop_size=args.pop,
        n_gen=args.gens,
        n_workers=args.workers,
        quick=args.quick,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
