"""
MitigationAgent: tests 5 mitigation conditions using GPT-4o as judge.
Conditions:
  1. fixed_rubric      — pre-established CoT structure before candidate exposure
  2. style_norm        — strip markdown from candidate before judging
  3. style_agnostic    — explicit ignore-formatting instruction
  4. two_pass          — candidate never enters scoring context (see mechanistic_agent.py)
  5. structured_output — API-level forced JSON format (bonus condition)
"""
import json
import re
from pathlib import Path

from src.agents.evaluation_agent import run_evaluation
from src.api import openai_client
from src.utils import logger as log
from src.utils.state import ExperimentState


def _strip_markdown(text: str) -> str:
    """Remove markdown headers, bullets, numbered lists from candidate text."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = re.sub(r"^\s*#{1,6}\s+", "", line)   # headers
        line = re.sub(r"^\s*[-*•]\s+", "", line)     # bullets
        line = re.sub(r"^\s*\d+\.\s+", "", line)     # numbered lists
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line) # bold
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_json(path: str, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run_mitigation(
    condition: str,
    cfg: dict,
    state: ExperimentState,
) -> list[dict]:
    """
    Run one mitigation condition. condition must be one of:
    'fixed_rubric', 'style_norm', 'style_agnostic', 'structured_output'
    """
    phase_key = f"mitigation_{condition}"
    out_path = f"{cfg['paths']['results_mitigation']}/{condition}_scores.json"

    prompt_map = {
        "fixed_rubric": cfg["prompts"]["judge_fixed_rubric"],
        "style_agnostic": cfg["prompts"]["judge_style_agnostic"],
        "structured_output": cfg["prompts"]["judge_structured_output"],
    }

    if condition == "style_norm":
        return _run_style_norm(cfg, state, phase_key, out_path)
    elif condition in prompt_map:
        return run_evaluation(
            judge_name="gpt4o",
            cfg=cfg,
            state=state,
            prompt_template_path=prompt_map[condition],
            output_subdir=f"{cfg['paths']['results_mitigation']}/{condition}",
        )
    else:
        log.warn("MitigationAgent", f"Unknown condition: {condition}")
        return []


def _run_style_norm(
    cfg: dict,
    state: ExperimentState,
    phase_key: str,
    out_path: str,
) -> list[dict]:
    """
    Style normalization: strip markdown from response_text before judging.
    Creates modified instances on-the-fly; does not mutate evaluation_instances.json.
    """
    instances = _load_json(cfg["paths"]["evaluation_instances"])
    normalized_instances = []
    for inst in instances:
        inst_copy = dict(inst)
        inst_copy["response_text"] = _strip_markdown(inst["response_text"])
        normalized_instances.append(inst_copy)

    # Write temp file for evaluation agent
    tmp_path = "data/processed/evaluation_instances_style_norm.json"
    _save_json(tmp_path, normalized_instances)

    # Temporarily patch cfg to use the normalized instances
    cfg_copy = dict(cfg)
    paths_copy = dict(cfg["paths"])
    paths_copy["evaluation_instances"] = tmp_path
    cfg_copy["paths"] = paths_copy

    return run_evaluation(
        judge_name="gpt4o",
        cfg=cfg_copy,
        state=state,
        prompt_template_path=cfg["prompts"]["judge_pointwise"],
        output_subdir=f"{cfg['paths']['results_mitigation']}/style_norm",
    )
