"""
IRRAgent: Inter-Rater Reliability check on formality level assignments.
Uses GPT-4o to independently classify each variant as L1/L2/L3/L4.
Computes Cohen's Kappa. Threshold: >= 0.75 before evaluation phase.
"""
import json
from pathlib import Path

from src.api import openai_client
from src.utils import logger as log
from src.utils.state import ExperimentState


def _load_json(path: str) -> list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_json(path: str, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_prompt(path: str, **kwargs) -> str:
    text = Path(path).read_text(encoding="utf-8")
    for k, v in kwargs.items():
        text = text.replace("{" + k + "}", str(v))
    return text


def _cohen_kappa(assigned: list[str], predicted: list[str]) -> float:
    """Simple Cohen's Kappa for binary labels (L2/L4)."""
    assert len(assigned) == len(predicted)
    n = len(assigned)
    categories = ["L2", "L4"]
    # Observed agreement
    p_o = sum(1 for a, p in zip(assigned, predicted) if a == p) / n
    # Expected agreement
    p_e = sum(
        (assigned.count(c) / n) * (predicted.count(c) / n)
        for c in categories
    )
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def run_irr_check(cfg: dict, state: ExperimentState) -> dict:
    if state.is_phase_complete("irr_check"):
        log.info("IRRAgent", "IRR check already complete, loading results")
        return _load_json(cfg["paths"]["irr_results"])

    variants = _load_json(cfg["paths"]["style_variants_normalized"])
    prompt_template = cfg["prompts"]["irr_formality_check"]

    assigned_labels = []
    predicted_labels = []
    detailed = []

    for variant in variants:
        variant_id = variant["variant_id"]
        log.info("IRRAgent", f"Classifying formality: {variant_id}")
        prompt = _load_prompt(prompt_template, response_text=variant["response_text"])
        r = openai_client.call_with_retry(
            prompt=prompt,
            system_prompt="You are a text formality classifier. Reply with only a single digit.",
            model=cfg["models"]["irr_classifier"],
            temperature=cfg["temperature"]["irr"],
            max_tokens=cfg["max_tokens"]["irr_classification"],
        )
        raw = r["content"].strip()
        # Extract digit — only 2 or 4 are valid for binary classification
        digit = next((c for c in raw if c in "24"), None)
        predicted_level = f"L{digit}" if digit else "UNKNOWN"

        assigned_labels.append(variant["formality_level"])
        predicted_labels.append(predicted_level)
        detailed.append({
            "variant_id": variant_id,
            "assigned_level": variant["formality_level"],
            "irr_predicted_level": predicted_level,
            "match": variant["formality_level"] == predicted_level,
        })

    kappa = _cohen_kappa(assigned_labels, predicted_labels)
    accuracy = sum(1 for a, p in zip(assigned_labels, predicted_labels) if a == p) / len(assigned_labels)
    threshold = cfg["quality_gates"]["irr_kappa_threshold"]
    passed = kappa >= threshold

    result = {
        "cohen_kappa": round(kappa, 4),
        "accuracy": round(accuracy, 4),
        "threshold": threshold,
        "passed": passed,
        "n_variants": len(variants),
        "details": detailed,
    }
    _save_json(cfg["paths"]["irr_results"], result)

    if passed:
        state.mark_phase_complete("irr_check")
        log.info("IRRAgent", f"IRR PASSED: kappa={kappa:.3f} >= {threshold}")
    else:
        log.warn("IRRAgent", f"IRR FAILED: kappa={kappa:.3f} < {threshold}. Revise rewriting prompts.")
        # Identify confused pairs
        confused = [d for d in detailed if not d["match"]]
        log.warn("IRRAgent", f"{len(confused)} variants misclassified: {confused[:5]}")

    return result
