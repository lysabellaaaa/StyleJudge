"""
StyleBias Score (SBS) computation and effect size reporting.
SBS(judge, domain) = mean(score | formality=L4) - mean(score | formality=L2)
Negative SBS confirms H2: structured responses receive stricter (lower) scores.
"""
import math
import random
from collections import defaultdict


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _pooled_std(group_a: list[float], group_b: list[float]) -> float:
    if len(group_a) < 2 or len(group_b) < 2:
        return float("nan")
    n_a, n_b = len(group_a), len(group_b)
    var_a = sum((x - _mean(group_a)) ** 2 for x in group_a) / (n_a - 1)
    var_b = sum((x - _mean(group_b)) ** 2 for x in group_b) / (n_b - 1)
    return math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Cohen's d = (mean_a - mean_b) / pooled_std."""
    ps = _pooled_std(group_a, group_b)
    if math.isnan(ps) or ps == 0:
        return float("nan")
    return (_mean(group_a) - _mean(group_b)) / ps


def bootstrap_ci(
    group_a: list[float],
    group_b: list[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap 95% CI for (mean_a - mean_b)."""
    diffs = []
    for _ in range(n_resamples):
        sample_a = [random.choice(group_a) for _ in group_a]
        sample_b = [random.choice(group_b) for _ in group_b]
        diffs.append(_mean(sample_a) - _mean(sample_b))
    diffs.sort()
    lo = diffs[int(alpha / 2 * n_resamples)]
    hi = diffs[int((1 - alpha / 2) * n_resamples)]
    return lo, hi


def compute_sbs(
    scores: list[dict],
    judge_model: str,
    domain: str,
    n_resamples: int = 1000,
) -> dict:
    """
    Compute SBS, Cohen's d, and 95% CI for a specific judge × domain combination.
    scores: list of raw_scores records with fields: judge_model, domain, formality_level, score
    """
    filtered = [
        s for s in scores
        if s["judge_model"] == judge_model
        and s.get("domain") == domain
        and not s.get("is_adversarial", False)
    ]
    l4_scores = [s["score"] for s in filtered if s["formality_level"] == "L4"]
    l2_scores = [s["score"] for s in filtered if s["formality_level"] == "L2"]

    if not l4_scores or not l2_scores:
        return {
            "judge_model": judge_model,
            "domain": domain,
            "sbs": float("nan"),
            "cohens_d": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_l4": len(l4_scores),
            "n_l2": len(l2_scores),
        }

    sbs = _mean(l4_scores) - _mean(l2_scores)
    d = cohens_d(l4_scores, l2_scores)
    ci_lo, ci_hi = bootstrap_ci(l4_scores, l2_scores, n_resamples)

    return {
        "judge_model": judge_model,
        "domain": domain,
        "sbs": round(sbs, 4),
        "mean_l4": round(_mean(l4_scores), 4),
        "mean_l2": round(_mean(l2_scores), 4),
        "cohens_d": round(d, 4),
        "ci_lower": round(ci_lo, 4),
        "ci_upper": round(ci_hi, 4),
        "n_l4": len(l4_scores),
        "n_l2": len(l2_scores),
    }


def compute_sbs_matrix(
    scores: list[dict],
    judges: list[str],
    domains: list[str],
    n_resamples: int = 1000,
) -> list[dict]:
    """Compute SBS for all judge × domain combinations."""
    results = []
    for judge in judges:
        for domain in domains:
            results.append(compute_sbs(scores, judge, domain, n_resamples))
    return results
