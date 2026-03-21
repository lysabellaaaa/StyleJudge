"""
Error Detection Rate (EDR) and False Penalty Rate (FPR) computation.
Pilot note: n=6 adversarial instances. Results are directional only.
"""
from collections import defaultdict


def compute_edr(
    scores: list[dict],
    penalty_threshold: int = 2,
) -> dict[str, float]:
    """
    EDR per formality level: fraction of adversarial instances scored <= threshold.
    A low score on an adversarial response = judge caught the injected error.
    """
    by_level: dict[str, list[int]] = defaultdict(list)
    for s in scores:
        if s.get("is_adversarial", False):
            by_level[s["formality_level"]].append(s["score"])

    edr = {}
    for level, level_scores in by_level.items():
        caught = sum(1 for sc in level_scores if sc <= penalty_threshold)
        edr[level] = round(caught / len(level_scores) * 100, 1) if level_scores else float("nan")
    return edr


def compute_fpr(
    scores: list[dict],
    penalty_threshold: int = 2,
) -> dict[str, float]:
    """
    FPR per formality level: fraction of CORRECT responses (non-adversarial) scored <= threshold.
    A low score on a correct response = unfair penalty.
    """
    by_level: dict[str, list[int]] = defaultdict(list)
    for s in scores:
        if not s.get("is_adversarial", False):
            by_level[s["formality_level"]].append(s["score"])

    fpr = {}
    for level, level_scores in by_level.items():
        penalized = sum(1 for sc in level_scores if sc <= penalty_threshold)
        fpr[level] = round(penalized / len(level_scores) * 100, 1) if level_scores else float("nan")
    return fpr
