"""
Halo Effect vs. Scrutiny Effect decomposition.

Key insight: The two effects operate on different stimulus conditions.
- Correct responses (no errors): Halo predicts L4 > L2; Scrutiny finds nothing to penalize.
  → SBS_correct > 0 means Halo dominates.
- Adversarial responses (injected error): Halo still pushes L4 up; Scrutiny catches error → lower.
  → SBS_adversarial < 0 means Scrutiny dominates when there's something to find.

Pattern interpretation:
  SBS_correct > 0, SBS_adversarial < 0  → Both effects exist; scrutiny wins on errors
  SBS_correct > 0, SBS_adversarial > 0  → Halo dominates even when errors present (danger zone)
  SBS_correct < 0, SBS_adversarial < 0  → Scrutiny dominates in all conditions (confirms H2)
"""
from src.metrics.style_bias_score import compute_sbs, _mean


def decompose_effects(
    scores: list[dict],
    judge_model: str,
    domain: str | None = None,
    n_resamples: int = 1000,
) -> dict:
    """
    Compute SBS_correct and SBS_adversarial separately.
    """
    def filter_scores(adversarial: bool) -> list[dict]:
        filtered = [s for s in scores if s["judge_model"] == judge_model
                    and s.get("is_adversarial", False) == adversarial]
        if domain:
            filtered = [s for s in filtered if s.get("domain") == domain]
        return filtered

    correct_scores = filter_scores(adversarial=False)
    adv_l4_scores = filter_scores(adversarial=True)

    sbs_correct = _sbs_from_scores(correct_scores)
    # SBS_adversarial: L4 flawed (adversarial=True) vs. L2 clean (adversarial=False).
    # There is no "L2 adversarial" — errors are only injected into L4 structured variants.
    # So the L2 baseline is the non-adversarial L2 from correct_scores.
    l1_correct = [s for s in correct_scores if s["formality_level"] == "L2"]
    l4_adv = [s for s in adv_l4_scores if s["formality_level"] == "L4"]
    if l4_adv and l1_correct:
        from src.metrics.style_bias_score import _mean
        sbs_adversarial = _mean([s["score"] for s in l4_adv]) - _mean([s["score"] for s in l1_correct])
    else:
        sbs_adversarial = None

    # Interpret the pattern
    if sbs_correct is not None and sbs_adversarial is not None:
        if sbs_correct > 0 and sbs_adversarial < 0:
            interpretation = "BOTH_EFFECTS: halo on correct, scrutiny on adversarial"
        elif sbs_correct > 0 and sbs_adversarial >= 0:
            interpretation = "HALO_DOMINANT: halo overrides scrutiny even on adversarial"
        elif sbs_correct <= 0 and sbs_adversarial < 0:
            interpretation = "SCRUTINY_DOMINANT: scrutiny dominates all conditions"
        else:
            interpretation = "UNCLEAR: no strong pattern"
    else:
        interpretation = "INSUFFICIENT_DATA"

    return {
        "judge_model": judge_model,
        "domain": domain or "all",
        "sbs_correct": round(sbs_correct, 4) if sbs_correct is not None else None,
        "sbs_adversarial": round(sbs_adversarial, 4) if sbs_adversarial is not None else None,
        "n_correct": len(correct_scores),
        "n_adversarial": len(adv_l4_scores),
        "interpretation": interpretation,
    }


def _sbs_from_scores(scores: list[dict]) -> float | None:
    l4 = [s["score"] for s in scores if s["formality_level"] == "L4"]
    l1 = [s["score"] for s in scores if s["formality_level"] == "L2"]
    if not l4 or not l1:
        return None
    return _mean(l4) - _mean(l1)


def run_full_decomposition(
    scores: list[dict],
    judges: list[str],
    domains: list[str],
) -> list[dict]:
    results = []
    for judge in judges:
        # Overall
        results.append(decompose_effects(scores, judge))
        # Per domain
        for domain in domains:
            results.append(decompose_effects(scores, judge, domain))
    return results
