"""
Orchestrator v3: Full StyleJudge study pipeline.

Phases:
  1. Dataset construction
       1a. generate_rubric            → data/raw/rubric_nonfactual.json
       1b. generate_base_responses    → data/dataset/base_responses.json
       1c. rewrite_to_styles          → data/dataset/style_variants.json (Artificial mode)
       1d. generate_natural_variants  → data/dataset/natural_variants.json (Natural mode)
       1e. inject_adversarial_errors  → data/dataset/adversarial.json
       1f. build_evaluation_instances → data/processed/evaluation_instances*.json (3 files)

  2. Rubric evaluation (3 judges × 3 modes)
       claude/gpt4o/llama70b × artificial/natural/adversarial

  3. Pairwise evaluation (2 judges × 2 modes)
       claude/gpt4o × artificial/natural
       (Llama excluded: Groq 6k TPM too low for long pairwise prompts)

  4. Mitigation evaluation (Claude only, Artificial mode)
       format_agnostic | style_norm | fixed_rubric

  5. CoT analysis (mechanistic, H7)
       → results/mechanistic/cot_analysis.json

  6. Analysis → results/analysis/metrics_summary.json + findings_v3.md

Questions come from data/raw/base_prompts.json, built by scripts/crawl_benchmarks.py.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from src.agents import (
    dataset_agent,
    evaluation_agent,
    pairwise_agent,
    analysis_agent,
)
from src.agents.natural_generator import generate_natural_variants
from src.utils import logger as log
from src.utils.state import ExperimentState


def load_cfg(config_path: str = "config/experiment.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_json(path: str, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Build evaluation instance files
# ---------------------------------------------------------------------------

def build_evaluation_instances(cfg: dict, state: ExperimentState) -> list[dict]:
    """Build Artificial mode evaluation instances from style_variants.json."""
    if state.is_phase_complete("build_instances"):
        log.info("Orchestrator", "Evaluation instances already built — skipping")
        return _load_json(cfg["paths"]["evaluation_instances"])

    variants = _load_json(cfg["paths"]["style_variants"])
    instances = []
    for v in variants:
        if not v.get("response_text"):
            continue
        instances.append({
            "instance_id":    f"eval_{v['variant_id']}",
            "variant_id":     v["variant_id"],
            "base_prompt_id": v["base_prompt_id"],
            "stream":         v["stream"],
            "variant_type":   v["variant_type"],
            "mode":           "artificial",
            "question_text":  v["question_text"],
            "response_text":  v["response_text"],
        })

    _save_json(cfg["paths"]["evaluation_instances"], instances)
    state.mark_phase_complete("build_instances")
    log.info("Orchestrator",
             f"Built {len(instances)} artificial evaluation instances")
    return instances


def build_natural_evaluation_instances(cfg: dict, state: ExperimentState) -> list[dict]:
    """Build Natural mode evaluation instances from natural_variants.json."""
    phase_key = "build_instances_natural"
    if state.is_phase_complete(phase_key):
        log.info("Orchestrator", "Natural evaluation instances already built — skipping")
        return _load_json(cfg["paths"]["evaluation_instances_natural"])

    variants = _load_json(cfg["paths"]["natural_variants"])
    instances = []
    for v in variants:
        if not v.get("response_text"):
            continue
        instances.append({
            "instance_id":    f"eval_{v['variant_id']}",
            "variant_id":     v["variant_id"],
            "base_prompt_id": v["base_prompt_id"],
            "stream":         v["stream"],
            "variant_type":   v["variant_type"],
            "mode":           "natural",
            "question_text":  v["question_text"],
            "response_text":  v["response_text"],
        })

    _save_json(cfg["paths"]["evaluation_instances_natural"], instances)
    state.mark_phase_complete(phase_key)
    log.info("Orchestrator",
             f"Built {len(instances)} natural evaluation instances")
    return instances


def build_adversarial_evaluation_instances(cfg: dict, state: ExperimentState) -> list[dict]:
    """Build Adversarial mode evaluation instances from adversarial.json."""
    phase_key = "build_instances_adversarial"
    if state.is_phase_complete(phase_key):
        log.info("Orchestrator", "Adversarial evaluation instances already built — skipping")
        return _load_json(cfg["paths"]["evaluation_instances_adversarial"])

    adversarial = _load_json(cfg["paths"]["adversarial"])
    instances = []
    for a in adversarial:
        if not a.get("response_text"):
            continue
        instances.append({
            "instance_id":      f"eval_{a['adversarial_id']}",
            "adversarial_id":   a["adversarial_id"],
            "variant_id":       a["variant_id"],
            "base_prompt_id":   a["base_prompt_id"],
            "stream":           a["stream"],
            "variant_type":     a["variant_type"],
            "mode":             "adversarial",
            "is_adversarial":   True,
            "question_text":    a["question_text"],
            "response_text":    a["response_text"],
            "error_description": a.get("error_description", ""),
        })

    _save_json(cfg["paths"]["evaluation_instances_adversarial"], instances)
    state.mark_phase_complete(phase_key)
    log.info("Orchestrator",
             f"Built {len(instances)} adversarial evaluation instances")
    return instances


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(cfg: dict, state: ExperimentState, limit: int = None) -> None:
    log.info("Orchestrator", "=== StyleJudge v3 Full Study Starting ===")

    # Verify base_prompts.json exists
    bp_path = Path(cfg["paths"]["base_prompts"])
    if not bp_path.exists():
        log.error("Orchestrator",
                  f"base_prompts.json not found at {bp_path}. "
                  "Run scripts/crawl_benchmarks.py first.")
        sys.exit(1)

    base_prompts = _load_json(str(bp_path))
    if limit:
        limited_ids = {q["question_id"] for q in base_prompts[:limit]}
        log.info("Orchestrator",
                 f"[--limit {limit}] restricting to first {limit} questions")
    else:
        limited_ids = None

    # -----------------------------------------------------------------------
    # Phase 1a: Non-factual rubric
    # -----------------------------------------------------------------------
    log.info("Orchestrator", "[Phase 1a] Generating non-factual rubric...")
    dataset_agent.generate_rubric(cfg, state)

    # -----------------------------------------------------------------------
    # Phase 1b: Base responses
    # -----------------------------------------------------------------------
    log.info("Orchestrator", "[Phase 1b] Generating base responses (DeepSeek-V3)...")
    dataset_agent.generate_base_responses(cfg, state)

    # -----------------------------------------------------------------------
    # Phase 1c: Artificial style variants (V-simple, V-abstract)
    # -----------------------------------------------------------------------
    log.info("Orchestrator", "[Phase 1c] Rewriting to V-simple / V-abstract...")
    dataset_agent.rewrite_to_styles(cfg, state)

    # -----------------------------------------------------------------------
    # Phase 1d: Natural variants (V-natural-simple via V3, V-natural-abstract via R1)
    # -----------------------------------------------------------------------
    log.info("Orchestrator", "[Phase 1d] Generating natural variants (V3 + R1)...")
    generate_natural_variants(cfg, state)

    # -----------------------------------------------------------------------
    # Phase 1e: Adversarial error injection
    # -----------------------------------------------------------------------
    log.info("Orchestrator", "[Phase 1e] Injecting adversarial errors...")
    dataset_agent.inject_adversarial_errors(cfg, state)

    # -----------------------------------------------------------------------
    # Phase 1f: Build evaluation instance files (3 separate files)
    # -----------------------------------------------------------------------
    log.info("Orchestrator", "[Phase 1f] Building evaluation instances...")
    build_evaluation_instances(cfg, state)
    build_natural_evaluation_instances(cfg, state)
    build_adversarial_evaluation_instances(cfg, state)

    # -----------------------------------------------------------------------
    # Phase 2: Rubric evaluation — 3 judges × 3 modes
    # -----------------------------------------------------------------------
    log.info("Orchestrator", "[Phase 2] Running rubric evaluation (3 judges × 3 modes)...")

    judges = list(cfg["models"]["judges"].keys())   # ["claude", "gpt4o", "llama70b"]
    rubric_modes = ["artificial", "natural", "adversarial"]

    for judge_id in judges:
        for mode in rubric_modes:
            log.info("Orchestrator",
                     f"[Phase 2] {judge_id}/{mode} rubric evaluation...")
            evaluation_agent.run_evaluation(cfg, state, judge_id=judge_id, mode=mode)

    # -----------------------------------------------------------------------
    # Phase 3: Pairwise evaluation — 2 judges × 2 modes
    # -----------------------------------------------------------------------
    log.info("Orchestrator", "[Phase 3] Running pairwise evaluation (2 judges × 2 modes)...")

    pairwise_judges = cfg["models"].get("pairwise_judges", ["claude", "gpt4o"])
    pairwise_modes  = ["artificial", "natural"]

    for judge_id in pairwise_judges:
        for mode in pairwise_modes:
            log.info("Orchestrator",
                     f"[Phase 3] {judge_id}/{mode} pairwise evaluation...")
            pairwise_agent.run_pairwise_evaluation(cfg, state,
                                                   judge_id=judge_id, mode=mode)

    # -----------------------------------------------------------------------
    # Phase 4: Mitigation evaluation (Claude only, Artificial mode)
    # -----------------------------------------------------------------------
    log.info("Orchestrator", "[Phase 4] Running mitigation conditions (Claude)...")

    mitigation_conditions = ["format_agnostic", "style_norm", "fixed_rubric"]
    for condition in mitigation_conditions:
        log.info("Orchestrator", f"[Phase 4] Mitigation condition: {condition}...")
        evaluation_agent.run_mitigation_evaluation(cfg, state, condition=condition)

    # -----------------------------------------------------------------------
    # Phase 5: CoT analysis (mechanistic — H7)
    # -----------------------------------------------------------------------
    log.info("Orchestrator", "[Phase 5] Running CoT trace analysis (H7)...")
    try:
        from src.agents.cot_analysis_agent import run_cot_analysis
        run_cot_analysis(cfg, state)
    except ImportError:
        log.warn("Orchestrator",
                 "cot_analysis_agent not found — skipping Phase 5 (H7 analysis)")

    # -----------------------------------------------------------------------
    # Phase 6: Analysis + findings report
    # -----------------------------------------------------------------------
    log.info("Orchestrator", "[Phase 6] Running analysis and writing findings report...")
    analysis_agent.run_analysis(cfg, state)

    log.info("Orchestrator", "=== StyleJudge v3 Experiment Complete ===")
    log.info("Orchestrator", f"Findings: {cfg['paths']['findings_report']}")
