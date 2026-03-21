"""
Unit tests for StyleBias Score, EDR, FPR, and Effect Decomposition.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.style_bias_score import compute_sbs, cohens_d, bootstrap_ci
from src.metrics.error_detection import compute_edr, compute_fpr
from src.metrics.effect_decomposition import decompose_effects, _sbs_from_scores


def _make_scores(formality_level: str, score: int, n: int, judge: str = "gpt4o",
                 domain: str = "welfare_reasoning", is_adversarial: bool = False) -> list[dict]:
    return [
        {"instance_id": f"eval_{i}_{formality_level}", "judge_model": judge,
         "formality_level": formality_level, "domain": domain, "score": score,
         "is_adversarial": is_adversarial}
        for i in range(n)
    ]


# ─── SBS Tests ───────────────────────────────────────────────────────────────

def test_sbs_negative_when_l4_lower():
    """Structured responses score lower → negative SBS → confirms H2."""
    scores = _make_scores("L4", 3, 5) + _make_scores("L2", 4, 5)
    result = compute_sbs(scores, "gpt4o", "welfare_reasoning", n_resamples=100)
    assert result["sbs"] < 0, f"Expected negative SBS, got {result['sbs']}"


def test_sbs_positive_when_l4_higher():
    """Halo effect scenario: structured responses score higher → positive SBS."""
    scores = _make_scores("L4", 5, 5) + _make_scores("L2", 3, 5)
    result = compute_sbs(scores, "gpt4o", "welfare_reasoning", n_resamples=100)
    assert result["sbs"] > 0


def test_sbs_nan_when_empty():
    result = compute_sbs([], "gpt4o", "welfare_reasoning")
    assert result["sbs"] != result["sbs"]  # NaN check


def test_cohens_d_direction():
    d = cohens_d([2.0, 3.0, 4.0], [3.0, 4.0, 5.0])
    assert d < 0  # group_a mean < group_b mean → negative d


def test_bootstrap_ci_contains_true_value():
    import random
    random.seed(42)
    group_a = [3.0] * 10
    group_b = [4.0] * 10
    lo, hi = bootstrap_ci(group_a, group_b, n_resamples=500)
    true_diff = -1.0  # mean_a - mean_b = 3-4 = -1
    assert lo <= true_diff <= hi, f"CI [{lo}, {hi}] does not contain {true_diff}"


# ─── EDR / FPR Tests ─────────────────────────────────────────────────────────

def test_edr_all_caught():
    """All adversarial responses score <= 2 → 100% EDR."""
    scores = _make_scores("L4", 2, 4, is_adversarial=True) + _make_scores("L4", 4, 4, is_adversarial=False)
    edr = compute_edr(scores, penalty_threshold=2)
    assert edr.get("L4") == 100.0


def test_edr_none_caught():
    scores = _make_scores("L4", 4, 4, is_adversarial=True)
    edr = compute_edr(scores, penalty_threshold=2)
    assert edr.get("L4") == 0.0


def test_fpr_no_false_penalties():
    scores = _make_scores("L4", 4, 5, is_adversarial=False) + _make_scores("L2", 4, 5, is_adversarial=False)
    fpr = compute_fpr(scores, penalty_threshold=2)
    assert fpr.get("L4") == 0.0
    assert fpr.get("L2") == 0.0


def test_fpr_all_penalized():
    scores = _make_scores("L4", 1, 5, is_adversarial=False)
    fpr = compute_fpr(scores, penalty_threshold=2)
    assert fpr.get("L4") == 100.0


# ─── Effect Decomposition Tests ──────────────────────────────────────────────

def test_decompose_both_effects():
    """SBS_correct > 0 (halo), SBS_adversarial < 0 (scrutiny) → BOTH_EFFECTS."""
    correct = _make_scores("L4", 5, 4) + _make_scores("L2", 3, 4)  # L4 higher → halo
    adv = _make_scores("L4", 2, 3, is_adversarial=True) + _make_scores("L2", 4, 3, is_adversarial=False)
    scores = correct + adv
    result = decompose_effects(scores, judge_model="gpt4o")
    assert result["sbs_correct"] > 0
    assert "BOTH_EFFECTS" in result["interpretation"] or "HALO" in result["interpretation"]


def test_decompose_scrutiny_dominant():
    """SBS_correct < 0 and SBS_adversarial < 0 → SCRUTINY_DOMINANT."""
    correct = _make_scores("L4", 3, 4) + _make_scores("L2", 4, 4)
    adv = _make_scores("L4", 2, 3, is_adversarial=True) + _make_scores("L2", 4, 3, is_adversarial=False)
    scores = correct + adv
    result = decompose_effects(scores, judge_model="gpt4o")
    assert result["sbs_correct"] < 0
    assert "SCRUTINY" in result["interpretation"]
