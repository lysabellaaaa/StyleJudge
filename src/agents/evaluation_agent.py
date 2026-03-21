"""
EvaluationAgent: runs pointwise scoring for a specific judge model.
Parameterized by judge name. Parallel execution via ThreadPoolExecutor.
Stores full CoT traces. Extracts scores via regex with fallback.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.api import openai_client, together_client, zhipu_client
from src.utils import logger as log
from src.utils.state import ExperimentState

SCORE_PATTERNS = [
    r"(?:Score|Rating|Final Score|Overall Score):\s*([1-5])",
    r"\b([1-5])\s*/\s*5\b",
    r"(?:give.*?|score of|rated?)\s*([1-5])\b",
]

JUDGE_CLIENTS = {
    "gpt4o": openai_client,
    "llama70b": together_client,
    "glm5": zhipu_client,
}

JUDGE_MODEL_IDS = {
    "gpt4o": "gpt-4o",
    "llama70b": "llama-3.3-70b-versatile",
    "glm5": "research-model",
}


def extract_score(text: str) -> int:
    for pattern in SCORE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    raise ValueError(f"Score not found in response: {text[:200]}")


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


def evaluate_instance(
    instance: dict,
    judge_name: str,
    cfg: dict,
    prompt_template_path: str,
) -> dict | None:
    instance_id = instance["instance_id"]
    client = JUDGE_CLIENTS[judge_name]
    model_id = JUDGE_MODEL_IDS[judge_name]
    prompt = _load_prompt(
        prompt_template_path,
        question=instance["prompt_text"],
        response_text=instance["response_text"],
    )
    system_prompt = "You are an expert evaluator. Follow the evaluation instructions precisely."

    try:
        r = client.call_with_retry(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model_id,
            temperature=cfg["temperature"]["evaluation"],
            max_tokens=cfg["max_tokens"]["judge_cot"],
        )
        cot = r["content"]
        try:
            score = extract_score(cot)
        except ValueError:
            # Retry once at temp=0.1
            log.warn("EvaluationAgent", f"Score extraction failed for {instance_id}, retrying at temp=0.1")
            r2 = client.call_with_retry(
                prompt=prompt, system_prompt=system_prompt,
                model=model_id, temperature=0.1,
                max_tokens=cfg["max_tokens"]["judge_cot"],
            )
            cot = r2["content"]
            score = extract_score(cot)

        return {
            "instance_id": instance_id,
            "judge_model": judge_name,
            "judge_model_id": model_id,
            "score": score,
            "cot_trace": cot,
            "formality_level": instance["formality_level"],
            "domain": instance["domain"],
            "base_prompt_id": instance["base_prompt_id"],
            "is_adversarial": instance.get("is_adversarial", False),
            "tokens_used": r["output_tokens"],
            "evaluated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        log.error("EvaluationAgent", f"Failed {instance_id}: {e}")
        return None


def run_evaluation(
    judge_name: str,
    cfg: dict,
    state: ExperimentState,
    prompt_template_path: str | None = None,
    output_subdir: str | None = None,
) -> list[dict]:
    if judge_name not in JUDGE_CLIENTS:
        log.warn("EvaluationAgent", f"Unknown judge: {judge_name}. Skipping.")
        return []

    instances = _load_json(cfg["paths"]["evaluation_instances"])
    if not prompt_template_path:
        prompt_template_path = cfg["prompts"]["judge_pointwise"]

    phase_key = f"evaluation_{judge_name}"
    out_dir = output_subdir or f"{cfg['paths']['results_evaluations']}/{judge_name}"
    scores_path = f"{out_dir}/raw_scores.json"

    existing = []
    if Path(scores_path).exists():
        existing = _load_json(scores_path)
    results = list(existing)

    pending = [
        inst for inst in instances
        if not state.is_complete(phase_key, inst["instance_id"])
    ]
    log.info("EvaluationAgent", f"[{judge_name}] {len(pending)} instances to evaluate")

    max_workers = cfg["evaluation"]["max_workers"].get(judge_name, 3)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(evaluate_instance, inst, judge_name, cfg, prompt_template_path): inst["instance_id"]
            for inst in pending
        }
        for fut in as_completed(futures):
            instance_id = futures[fut]
            result = fut.result()
            if result:
                results.append(result)
                _save_json(scores_path, results)
                state.mark_complete(phase_key, instance_id)
            else:
                state.log_error(phase_key, instance_id, "evaluation failed")

    log.info("EvaluationAgent", f"[{judge_name}] complete: {len(results)} scores")
    return results
