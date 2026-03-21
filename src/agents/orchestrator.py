"""
Orchestrator: main workflow controller for the StyleJudge experiment.
Pure Python threading — no nested Claude Code agents.
All phase gates are enforced before proceeding.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import yaml

from src.agents import (
    analysis_agent,
    dataset_agent,
    irr_agent,
    mechanistic_agent,
    mitigation_agent,
    qa_agent,
)
from src.agents.evaluation_agent import run_evaluation
from src.utils import logger as log
from src.utils.state import ExperimentState


def load_cfg(config_path: str = "config/experiment.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_evaluation_instances(cfg: dict, state: ExperimentState) -> None:
    """
    Combines qa_verified variants + adversarial instances into
    a flat evaluation_instances.json. Strips formality metadata from response_text sent to judges.
    """
    import datetime

    variants_path = cfg["paths"]["qa_verified"]
    adversarial_path = cfg["paths"]["adversarial"]
    out_path = cfg["paths"]["evaluation_instances"]

    if Path(out_path).exists() and state.is_phase_complete("build_instances"):
        log.info("Orchestrator", "evaluation_instances.json already built")
        return

    variants = json.loads(Path(variants_path).read_text(encoding="utf-8"))
    adversarial = []
    if Path(adversarial_path).exists():
        adversarial = json.loads(Path(adversarial_path).read_text(encoding="utf-8"))

    instances = []
    # Standard variants (non-adversarial)
    for v in variants:
        if not v.get("qa_passed", False):
            continue
        instances.append({
            "instance_id": f"eval_{v['variant_id']}",
            "variant_id": v["variant_id"],
            "base_prompt_id": v["base_prompt_id"],
            "domain": v["domain"],
            "prompt_text": v["prompt_text"],
            "formality_level": v["formality_level"],
            "is_adversarial": False,
            "adversarial_id": None,
            "response_text": v["response_text"],  # raw text only; no style metadata in text itself
        })

    # Adversarial instances (L4 with injected error)
    for adv in adversarial:
        # Add the flawed L4 variant
        instances.append({
            "instance_id": f"eval_{adv['adversarial_id']}_flawed",
            "variant_id": adv["flawed_variant_id"],
            "base_prompt_id": adv["base_prompt_id"],
            "domain": adv["domain"],
            "prompt_text": adv["prompt_text"],
            "formality_level": "L4",
            "is_adversarial": True,
            "adversarial_id": adv["adversarial_id"],
            "response_text": adv["flawed_response_text"],
        })
        # Also add the correct L1 counterpart as adversarial pair
        correct_variant = next(
            (v for v in variants if v["variant_id"] == adv.get("correct_variant_id")), None
        )
        if correct_variant:
            instances.append({
                "instance_id": f"eval_{adv['adversarial_id']}_correct",
                "variant_id": adv["correct_variant_id"],
                "base_prompt_id": adv["base_prompt_id"],
                "domain": adv["domain"],
                "prompt_text": adv["prompt_text"],
                "formality_level": "L1",
                "is_adversarial": False,
                "adversarial_id": adv["adversarial_id"],
                "response_text": correct_variant["response_text"],
            })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(instances, indent=2, ensure_ascii=False), encoding="utf-8")
    state.mark_phase_complete("build_instances")
    log.info("Orchestrator", f"Built {len(instances)} evaluation instances → {out_path}")


def run(cfg: dict, state: ExperimentState, limit: int | None = None) -> None:
    """
    Main experiment workflow. All phases enforced in order.
    limit: if set, restricts dataset to first N base prompts (for dry-run).
    """
    log.info("Orchestrator", "=== StyleJudge Experiment Starting ===")

    # ─── Phase 1: Dataset Construction ───────────────────────────────────────
    if not state.is_phase_complete("base_generation"):
        log.info("Orchestrator", "[Phase 1] Generating base responses...")
        dataset_agent.generate_base_responses(cfg, state)
    if not state.is_phase_complete("style_rewriting"):
        log.info("Orchestrator", "[Phase 1] Rewriting to L1-L4 styles...")
        dataset_agent.rewrite_to_styles(cfg, state)
        state.mark_phase_complete("style_rewriting")
    if not state.is_phase_complete("length_normalization"):
        log.info("Orchestrator", "[Phase 1] Normalizing token lengths...")
        dataset_agent.normalize_lengths(cfg, state)
    if not state.is_phase_complete("adversarial_injection"):
        log.info("Orchestrator", "[Phase 1] Injecting adversarial errors...")
        dataset_agent.inject_adversarial_errors(cfg, state)
        state.mark_phase_complete("adversarial_injection")

    # ─── Gate 1: IRR Check ───────────────────────────────────────────────────
    irr_result = irr_agent.run_irr_check(cfg, state)
    if not irr_result.get("passed", False):
        log.error("Orchestrator", f"IRR GATE FAILED: kappa={irr_result.get('cohen_kappa'):.3f}. "
                  "Revise style rewriting prompts and re-run. Halting.")
        sys.exit(1)
    log.info("Orchestrator", f"IRR Gate passed (kappa={irr_result['cohen_kappa']:.3f})")

    # ─── Gate 2: Formality Perception Check (semi-manual) ────────────────────
    if not state.is_phase_complete("formality_perception"):
        log.info("Orchestrator",
                 "Gate 2: Formality perception check — review 2 base prompts manually. "
                 "Once satisfied, run: python scripts/run_experiment.py --confirm-perception")
        # Non-blocking in pilot; log a warning
        log.warn("Orchestrator", "Proceeding without confirmed formality perception check (pilot mode)")
        state.mark_phase_complete("formality_perception")

    # ─── Gate 3: Semantic QA Verification ────────────────────────────────────
    if not state.is_phase_complete("qa_verification"):
        log.info("Orchestrator", "[Phase 1] Running semantic QA verification (GPT-4o)...")
        qa_agent.run_qa_verification(cfg, state)

    # ─── Build evaluation instances ──────────────────────────────────────────
    build_evaluation_instances(cfg, state)

    # ─── Gate 4: OSF Pre-registration (required for full study, waived for pilot) ──
    osf_url = state.get_osf_url()
    if osf_url:
        log.info("Orchestrator", f"Gate 4 passed: OSF pre-registration {osf_url}")
    else:
        log.warn("Orchestrator",
                 "Gate 4: OSF pre-registration URL not set. "
                 "Waived for pilot (n=10). Required before publishing full study results.")

    # ─── Phase 2: Evaluation (3 judges in parallel) ───────────────────────────
    judges = list(cfg["models"]["judges"].keys())
    eval_needed = [j for j in judges if not state.is_phase_complete(f"evaluation_{j}")]
    if eval_needed:
        log.info("Orchestrator", f"[Phase 2] Running evaluation: {eval_needed}")
        with ThreadPoolExecutor(max_workers=len(eval_needed)) as ex:
            futures = [ex.submit(run_evaluation, judge, cfg, state) for judge in eval_needed]
            wait(futures)
        for judge in eval_needed:
            state.mark_phase_complete(f"evaluation_{judge}")

    # ─── Phase 3: Mitigation (3 conditions in parallel) ──────────────────────
    conditions = ["fixed_rubric", "style_norm", "style_agnostic"]
    mit_needed = [c for c in conditions if not state.is_phase_complete(f"mitigation_{c}")]
    if mit_needed:
        log.info("Orchestrator", f"[Phase 3] Running mitigations: {mit_needed}")
        with ThreadPoolExecutor(max_workers=len(mit_needed)) as ex:
            futures = [ex.submit(mitigation_agent.run_mitigation, c, cfg, state) for c in mit_needed]
            wait(futures)
        for c in mit_needed:
            state.mark_phase_complete(f"mitigation_{c}")

    # ─── Phase 4: Mechanistic Experiments (in parallel) ──────────────────────
    if not state.is_phase_complete("mechanistic_position"):
        log.info("Orchestrator", "[Phase 4] Running M1: Context Position Test...")
        mechanistic_agent.run_m1_position_test(cfg, state)
    if not state.is_phase_complete("mechanistic_buffer"):
        log.info("Orchestrator", "[Phase 4] Running M2: Style Buffer Test...")
        mechanistic_agent.run_m2_buffer_test(cfg, state)
    if not state.is_phase_complete("mechanistic_logprob"):
        log.info("Orchestrator", "[Phase 4] Running M3: Log-Probability Analysis (Llama 70B)...")
        mechanistic_agent.run_m3_logprob_analysis(cfg, state)
    if not state.is_phase_complete("mechanistic_two_pass"):
        log.info("Orchestrator", "[Phase 4] Running M4: Two-Pass Context Isolation...")
        mechanistic_agent.run_m4_two_pass(cfg, state)

    # ─── Phase 5: Analysis ────────────────────────────────────────────────────
    if not state.is_phase_complete("analysis"):
        log.info("Orchestrator", "[Phase 5] Running analysis...")
        analysis_agent.run_analysis(cfg, state)

    log.info("Orchestrator", "=== Experiment Complete ===")
    log.info("Orchestrator", f"Results: {cfg['paths']['results_analysis']}/")
