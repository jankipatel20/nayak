"""Statistical significance testing with Wilcoxon signed-rank test and Holm correction."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def holm_bonferroni_correction(p_values: Sequence[float]) -> List[float]:
    """Perform Holm-Bonferroni step-down correction on a list of p-values.

    Ensures family-wise error rate (FWER) control while providing more power than raw Bonferroni.

    Args:
        p_values: Unadjusted raw p-values.

    Returns:
        Adjusted p-values preserving the original input order.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values with original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    adjusted = np.zeros(m, dtype=float)
    running_max = 0.0

    for i in range(m):
        rank = i + 1
        adj_p = (m - rank + 1) * sorted_p[i]
        running_max = max(running_max, adj_p)
        adjusted[i] = min(1.0, running_max)

    # Restore original ordering
    out = np.zeros(m, dtype=float)
    out[sorted_indices] = adjusted
    return [float(p) for p in out]


def compute_paired_wilcoxon(
    a: Sequence[float],
    b: Sequence[float],
) -> Tuple[float, float, float]:
    """Execute paired two-sided Wilcoxon signed-rank test with effect size.

    Args:
        a: Metric observations for controller A (e.g. Nayak variant).
        b: Paired metric observations for controller B (baseline) on identical seeds.

    Returns:
        Tuple of (statistic, p_value, effect_size_r).
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    diff = arr_a - arr_b

    # Eliminate exact ties
    nonzero_diff = diff[diff != 0.0]
    n = len(nonzero_diff)
    if n == 0:
        return 0.0, 1.0, 0.0

    try:
        res = stats.wilcoxon(arr_a, arr_b, zero_method="wilcox", alternative="two-sided")
        stat = float(res.statistic)
        p_val = float(res.pvalue)

        # Matched-pairs rank-biserial effect size r = (W+ - W-) / (W+ + W-)
        ranks = stats.rankdata(np.abs(nonzero_diff))
        w_plus = np.sum(ranks[nonzero_diff > 0])
        w_minus = np.sum(ranks[nonzero_diff < 0])
        total_w = w_plus + w_minus
        effect_size = float((w_plus - w_minus) / total_w) if total_w > 0 else 0.0
    except Exception:
        stat = 0.0
        p_val = 1.0
        effect_size = 0.0

    return stat, p_val, effect_size


def analyze_experiment_statistics(
    df_raw: pd.DataFrame,
    metrics: Sequence[str] = (
        "mean_wait",
        "wait_p95",
        "mean_ambulance_delay",
        "throughput_vehicles",
    ),
) -> pd.DataFrame:
    """Run comprehensive Wilcoxon signed-rank comparisons of Nayak variants vs baselines.

    Args:
        df_raw: DataFrame containing per-episode records with columns:
                ['scenario', 'controller', 'seed', ...metrics]
        metrics: Metrics to analyze for significance.

    Returns:
        Summary DataFrame with test statistics, p-values, Holm-adjusted p-values, and effect sizes.
    """
    variants = ["Nayak-Fast", "Nayak-Fair", "Nayak-Rescue", "Nayak-Balanced"]
    baselines = ["FixedTime(30s)", "LongestQueueFirst(margin=2.0)", "HandDesignedFuzzy"]
    scenarios = df_raw["scenario"].unique()

    rows: List[Dict[str, object]] = []

    for sc in scenarios:
        df_sc = df_raw[df_raw["scenario"] == sc]

        for metric in metrics:
            metric_tests: List[Dict[str, object]] = []

            for var in variants:
                df_var = df_sc[df_sc["controller"] == var].sort_values("seed")
                vals_var = df_var[metric].values

                for base in baselines:
                    df_base = df_sc[df_sc["controller"] == base].sort_values("seed")
                    vals_base = df_base[metric].values

                    if len(vals_var) == len(vals_base) and len(vals_var) > 0:
                        stat, p_val, effect = compute_paired_wilcoxon(vals_var, vals_base)
                        mean_diff = float(np.mean(vals_var) - np.mean(vals_base))
                        pct_diff = (
                            float((np.mean(vals_var) - np.mean(vals_base)) / np.mean(vals_base) * 100.0)
                            if np.mean(vals_base) != 0
                            else 0.0
                        )

                        metric_tests.append({
                            "scenario": sc,
                            "metric": metric,
                            "variant": var,
                            "baseline": base,
                            "mean_variant": float(np.mean(vals_var)),
                            "mean_baseline": float(np.mean(vals_base)),
                            "mean_diff": mean_diff,
                            "pct_diff": pct_diff,
                            "wilcoxon_stat": stat,
                            "p_raw": p_val,
                            "effect_size_r": effect,
                        })

            # Apply Holm-Bonferroni within each (scenario, metric) family
            if metric_tests:
                raw_ps = [t["p_raw"] for t in metric_tests]
                adj_ps = holm_bonferroni_correction(raw_ps)
                for t, adj_p in zip(metric_tests, adj_ps):
                    t["p_holm"] = adj_p
                    t["significant_05"] = adj_p < 0.05
                    rows.append(t)

    return pd.DataFrame(rows)
