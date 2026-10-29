"""Representative genome selection from Pareto front and artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np


def select_pareto_representatives(
    F: np.ndarray,
    X: np.ndarray,
) -> Dict[str, Dict[str, object]]:
    """Extract four representative policies from the non-dominated Pareto front.

    Variants:
    - Nayak-Fast: Lowest f1 (minimum non-ambulance mean wait).
    - Nayak-Fair: Lowest f2 (minimum 95th percentile wait).
    - Nayak-Rescue: Lowest f3 (minimum emergency ambulance delay).
    - Nayak-Balanced: Knee point (closest to ideal normalized origin).

    Args:
        F: Array of shape (N, 3) containing evaluated objectives [f1, f2, f3].
        X: Array of shape (N, 44) containing corresponding genomes.

    Returns:
        Dict mapping variant name to dict with genome, objectives, and description.
    """
    n_points = len(F)
    if n_points == 0:
        raise ValueError("Cannot select from an empty Pareto front.")

    # 1. Nayak-Fast: lowest f1
    idx_fast = int(np.argmin(F[:, 0]))

    # 2. Nayak-Fair: lowest f2
    idx_fair = int(np.argmin(F[:, 1]))

    # 3. Nayak-Rescue: lowest f3
    idx_rescue = int(np.argmin(F[:, 2]))

    # 4. Nayak-Balanced: Knee point closest to ideal point in normalized objective space
    f_min = np.min(F, axis=0)
    f_max = np.max(F, axis=0)
    denom = np.where((f_max - f_min) > 1e-9, f_max - f_min, 1.0)
    f_norm = (F - f_min) / denom
    dist_to_ideal = np.linalg.norm(f_norm, axis=1)
    idx_balanced = int(np.argmin(dist_to_ideal))

    selections = {
        "Nayak-Fast": {
            "variant": "Nayak-Fast",
            "description": "Optimized strictly for minimum non-ambulance mean wait time (lowest f1)",
            "pareto_index": idx_fast,
            "objectives": {
                "mean_wait": float(F[idx_fast, 0]),
                "wait_p95": float(F[idx_fast, 1]),
                "ambulance_delay": float(F[idx_fast, 2]),
            },
            "genome": [float(v) for v in X[idx_fast]],
        },
        "Nayak-Fair": {
            "variant": "Nayak-Fair",
            "description": "Optimized for traffic equity and tail delay suppression (lowest f2 / 95th percentile wait)",
            "pareto_index": idx_fair,
            "objectives": {
                "mean_wait": float(F[idx_fair, 0]),
                "wait_p95": float(F[idx_fair, 1]),
                "ambulance_delay": float(F[idx_fair, 2]),
            },
            "genome": [float(v) for v in X[idx_fair]],
        },
        "Nayak-Rescue": {
            "variant": "Nayak-Rescue",
            "description": "Optimized strictly for emergency clearance (lowest f3 / minimum ambulance delay)",
            "pareto_index": idx_rescue,
            "objectives": {
                "mean_wait": float(F[idx_rescue, 0]),
                "wait_p95": float(F[idx_rescue, 1]),
                "ambulance_delay": float(F[idx_rescue, 2]),
            },
            "genome": [float(v) for v in X[idx_rescue]],
        },
        "Nayak-Balanced": {
            "variant": "Nayak-Balanced",
            "description": "Knee-point trade-off minimizing normalized Euclidean distance to ideal point across all 3 objectives",
            "pareto_index": idx_balanced,
            "objectives": {
                "mean_wait": float(F[idx_balanced, 0]),
                "wait_p95": float(F[idx_balanced, 1]),
                "ambulance_delay": float(F[idx_balanced, 2]),
            },
            "genome": [float(v) for v in X[idx_balanced]],
        },
    }

    return selections


def save_pareto_artifacts(
    F: np.ndarray,
    X: np.ndarray,
    results_dir: Path | str,
) -> Dict[str, Dict[str, object]]:
    """Save full Pareto front and export individual JSON files for the four picks."""
    res_dir = Path(results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save full front as compressed npz
    pareto_npz = res_dir / "pareto.npz"
    np.savez_compressed(pareto_npz, F=F, X=X)

    # 2. Extract representative picks and save individually
    reps = select_pareto_representatives(F, X)
    for name, data in reps.items():
        json_path = res_dir / f"{name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    return reps
