"""
MechanisticAgent: four experiments to test WHY StyleBias occurs.

M1: Context Position Test  — does proximity of candidate to CoT generation matter?
M2: Style Buffer Injection — can pre/post-candidate style reset reduce bias?
M3: Log-Probability Analysis — does L4 candidate increase P(structured first token)?
M4: Two-Pass Context Isolation — does removing candidate from scoring context → SBS ≈ 0?
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.agents.evaluation_agent import evaluate_instance, extract_score, JUDGE_MODEL_IDS
from src.api import openai_client, together_client
from src.utils import logger as log
from src.utils.state import ExperimentState


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_json(path: str, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_prompt(path: str, **kwargs) -> str:
    text = Path(path).read_text(encoding="utf-8")
    for k, v in kwargs.items():
        text = text.replace("{" + k + "}", str(v))
    return text


# ─── M1: Context Position Test ───────────────────────────────────────────────

def run_m1_position_test(cfg: dict, state: ExperimentState) -> dict:
    """Tests 3 candidate positions in the judge prompt."""
    instances = _load_json(cfg["paths"]["evaluation_instances"])
    conditions = {
        "position_A": cfg["prompts"]["judge_position_A"],
        "position_B": cfg["prompts"]["judge_position_B"],
        "position_C": cfg["prompts"]["judge_position_C"],
    }
    results = {}
    for cond_name, prompt_path in conditions.items():
        phase_key = f"mechanistic_position_{cond_name}"
        out_path = f"{cfg['paths']['results_mechanistic']}/position_test/{cond_name}_scores.json"
        scores = _run_condition(
            instances, "gpt4o", cfg, prompt_path, phase_key, out_path, state
        )
        results[cond_name] = scores
    state.mark_phase_complete("mechanistic_position")
    return results


# ─── M2: Style Buffer Injection ──────────────────────────────────────────────

def run_m2_buffer_test(cfg: dict, state: ExperimentState) -> dict:
    """Tests pre- and post-candidate style reset buffers."""
    instances = _load_json(cfg["paths"]["evaluation_instances"])
    conditions = {
        "buffer_baseline": cfg["prompts"]["judge_pointwise"],
        "buffer_B": cfg["prompts"]["judge_buffer_B"],  # post-candidate reset
        "buffer_C": cfg["prompts"]["judge_buffer_C"],  # pre-established style
    }
    results = {}
    for cond_name, prompt_path in conditions.items():
        phase_key = f"mechanistic_buffer_{cond_name}"
        out_path = f"{cfg['paths']['results_mechanistic']}/buffer_test/{cond_name}_scores.json"
        scores = _run_condition(
            instances, "gpt4o", cfg, prompt_path, phase_key, out_path, state
        )
        results[cond_name] = scores
    state.mark_phase_complete("mechanistic_buffer")
    return results


# ─── M3: Log-Probability Analysis (Llama 70B only) ───────────────────────────

STRUCTURE_TOKENS = {"First", "1.", "##", "Criterion", "Aspect", "- ", "**"}


def run_m3_logprob_analysis(cfg: dict, state: ExperimentState) -> list[dict]:
    """
    Extracts log-probabilities of structured tokens at the first N CoT tokens.
    Requires Groq Llama 3.3 70B with logprobs=True (top_logprobs=5).
    """
    if state.is_phase_complete("mechanistic_logprob"):
        return _load_json(f"{cfg['paths']['results_mechanistic']}/logprob/logprob_results.json")

    instances = _load_json(cfg["paths"]["evaluation_instances"])
    out_path = f"{cfg['paths']['results_mechanistic']}/logprob/logprob_results.json"
    n_tokens = cfg["analysis"]["mechanistic_logprob_tokens"]
    results = []

    for inst in instances:
        instance_id = inst["instance_id"]
        if state.is_complete("mechanistic_logprob", instance_id):
            continue
        log.info("MechanisticAgent", f"M3 logprob: {instance_id}")
        prompt = _load_prompt(
            cfg["prompts"]["judge_pointwise"],
            question=inst["prompt_text"],
            response_text=inst["response_text"],
        )
        try:
            r = together_client.call_with_retry(
                prompt=prompt,
                system_prompt="You are an expert evaluator. Follow the evaluation instructions precisely.",
                model=JUDGE_MODEL_IDS["llama70b"],
                temperature=0.0,
                max_tokens=cfg["max_tokens"]["judge_cot"],
                logprobs=5,
            )
            # Compute P_structure from first n_tokens logprob data
            p_structure = _compute_p_structure(r.get("logprobs"), n_tokens)
            results.append({
                "instance_id": instance_id,
                "formality_level": inst["formality_level"],
                "domain": inst["domain"],
                "base_prompt_id": inst["base_prompt_id"],
                "p_structure": p_structure,
                "cot_preview": r["content"][:200],
            })
            _save_json(out_path, results)
            state.mark_complete("mechanistic_logprob", instance_id)
        except Exception as e:
            log.error("MechanisticAgent", f"M3 failed for {instance_id}: {e}")
            state.log_error("mechanistic_logprob", instance_id, str(e))

    state.mark_phase_complete("mechanistic_logprob")
    return results


def _compute_p_structure(logprobs_data, n_tokens: int) -> float | None:
    """
    Compute the sum of log-probabilities for structure tokens
    in the first n_tokens of the generated CoT.
    Returns None if logprobs not available.
    """
    if not logprobs_data:
        return None
    try:
        # Groq logprobs format: list of token dicts with top_logprobs
        total_p_structure = 0.0
        tokens_examined = 0
        token_list = logprobs_data if isinstance(logprobs_data, list) else []
        for token_data in token_list[:n_tokens]:
            top_logprobs = token_data.get("top_logprobs", {})
            for token_str, logprob in top_logprobs.items():
                if any(st in token_str for st in STRUCTURE_TOKENS):
                    total_p_structure += logprob
            tokens_examined += 1
        return round(total_p_structure / max(tokens_examined, 1), 4)
    except Exception:
        return None


# ─── M4: Two-Pass Context Window Isolation ────────────────────────────────────

def run_m4_two_pass(cfg: dict, state: ExperimentState) -> list[dict]:
    """
    Pass 1: Extract factual claims from candidate (GPT-4o, temp=0)
    Pass 2: Score the extracted claims only — original candidate never in scoring context
    """
    if state.is_phase_complete("mechanistic_two_pass"):
        return _load_json(f"{cfg['paths']['results_mechanistic']}/two_pass/two_pass_scores.json")

    instances = _load_json(cfg["paths"]["evaluation_instances"])
    out_path = f"{cfg['paths']['results_mechanistic']}/two_pass/two_pass_scores.json"
    results = []

    for inst in instances:
        instance_id = inst["instance_id"]
        if state.is_complete("mechanistic_two_pass", instance_id):
            continue
        log.info("MechanisticAgent", f"M4 two-pass: {instance_id}")
        try:
            # Pass 1: extract factual claims
            extract_prompt = _load_prompt(
                cfg["prompts"]["judge_two_pass_extract"],
                response_text=inst["response_text"],
            )
            r1 = openai_client.call_with_retry(
                prompt=extract_prompt,
                system_prompt="Extract factual claims as a plain numbered list. No formatting.",
                model="gpt-4o",
                temperature=0.0,
                max_tokens=cfg["max_tokens"]["mechanistic_extraction"],
            )
            factual_summary = r1["content"]

            # Pass 2: score the summary (fresh context — original not present)
            score_prompt = _load_prompt(
                cfg["prompts"]["judge_pointwise"],
                question=inst["prompt_text"],
                response_text=factual_summary,  # summary, not original
            )
            r2 = openai_client.call_with_retry(
                prompt=score_prompt,
                system_prompt="You are an expert evaluator. Follow the evaluation instructions precisely.",
                model="gpt-4o",
                temperature=0.0,
                max_tokens=cfg["max_tokens"]["judge_cot"],
            )
            cot = r2["content"]
            score = extract_score(cot)

            results.append({
                "instance_id": instance_id,
                "formality_level": inst["formality_level"],
                "domain": inst["domain"],
                "base_prompt_id": inst["base_prompt_id"],
                "is_adversarial": inst.get("is_adversarial", False),
                "score": score,
                "factual_summary": factual_summary,
                "cot_trace": cot,
            })
            _save_json(out_path, results)
            state.mark_complete("mechanistic_two_pass", instance_id)
        except Exception as e:
            log.error("MechanisticAgent", f"M4 failed for {instance_id}: {e}")
            state.log_error("mechanistic_two_pass", instance_id, str(e))

    state.mark_phase_complete("mechanistic_two_pass")
    return results


# ─── Shared helper ────────────────────────────────────────────────────────────

def _run_condition(
    instances: list[dict],
    judge_name: str,
    cfg: dict,
    prompt_path: str,
    phase_key: str,
    out_path: str,
    state: ExperimentState,
) -> list[dict]:
    existing = []
    if Path(out_path).exists():
        existing = _load_json(out_path)
    results = list(existing)
    pending = [i for i in instances if not state.is_complete(phase_key, i["instance_id"])]

    max_workers = cfg["evaluation"]["max_workers"].get(judge_name, 3)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(evaluate_instance, inst, judge_name, cfg, prompt_path): inst["instance_id"]
            for inst in pending
        }
        for fut in as_completed(futures):
            iid = futures[fut]
            result = fut.result()
            if result:
                results.append(result)
                _save_json(out_path, results)
                state.mark_complete(phase_key, iid)
    return results
