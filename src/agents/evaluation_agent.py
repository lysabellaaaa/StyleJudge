"""
EvaluationAgent v3: Rubric-based scoring with multi-judge support.

Judges: claude (primary), gpt4o (cross-family), llama70b (open-source, Groq)
Modes: artificial, natural, adversarial

Each judge × mode combination writes to:
  results/evaluations/{judge_id}/{mode}_scores.json

Output schema per record:
{
  "instance_id":    "eval_fq_001_V-simple",
  "judge_id":       "claude",
  "judge_model":    "claude-sonnet-4-6",
  "mode":           "artificial",
  "stream":         "factual",
  "variant_type":   "V-simple",
  "base_prompt_id": "fq_001",
  "criteria_scores": {...},
  "aggregate_score": 3.7,
  "cot_trace":      "full raw judge response",
  "tokens_used":    387,
  "evaluated_at":   "ISO8601Z"
}
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.api import anthropic_client, openai_client, together_client
from src.utils import logger as log
from src.utils.state import ExperimentState


def _load_prompt(path: str, **kwargs) -> str:
    text = Path(path).read_text(encoding="utf-8")
    for k, v in kwargs.items():
        text = text.replace("{" + k + "}", str(v))
    return text


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_json(path: str, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_json_from_response(text: str):
    """Extract JSON from judge response, handling markdown code fences."""
    if text is None:
        raise ValueError("Judge returned empty content")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def _rubric_to_text(rubric: dict) -> str:
    """Format rubric criteria as a readable string for injection into judge prompts."""
    lines = []
    for c in rubric.get("criteria", []):
        anchors = c.get("anchors", {})
        lines.append(
            f"**{c['name']}** ({c['id']}): {c['description']}\n"
            f"  - Score 1: {anchors.get('1', '')}\n"
            f"  - Score 3: {anchors.get('3', '')}\n"
            f"  - Score 5: {anchors.get('5', '')}"
        )
    return "\n\n".join(lines)


def _get_client(judge_id: str):
    return {
        "claude":   anthropic_client,
        "gpt4o":    openai_client,
        "llama70b": together_client,
    }[judge_id]


def _get_model(judge_id: str, cfg: dict) -> str:
    return cfg["models"]["judges"][judge_id]


def _extract_scores_factual(parsed: dict) -> tuple[dict, float]:
    criteria_keys = ["factual_accuracy", "logical_coherence", "completeness"]
    scores = {}
    total = 0.0
    count = 0
    for k in criteria_keys:
        entry = parsed.get(k, {})
        score = float(entry.get("score", 3))
        scores[k] = {"score": score, "reasoning": entry.get("reasoning", "")}
        total += score
        count += 1
    agg = parsed.get("aggregate_score", round(total / count, 1) if count else 3.0)
    return scores, float(agg)


def _extract_scores_nonfactual(parsed: dict, rubric: dict) -> tuple[dict, float]:
    criteria_ids = [c["id"] for c in rubric.get("criteria", [])]
    raw = parsed.get("criteria_scores", {})
    scores = {}
    total = 0.0
    count = 0
    for cid in criteria_ids:
        entry = raw.get(cid, {})
        score = float(entry.get("score", 3))
        scores[cid] = {"score": score, "reasoning": entry.get("reasoning", "")}
        total += score
        count += 1
    agg = parsed.get("aggregate_score", round(total / count, 1) if count else 3.0)
    return scores, float(agg)


def _evaluate_instance(instance: dict, judge_id: str, cfg: dict,
                       rubric: dict, retry_on_fail: bool = True) -> dict:
    client = _get_client(judge_id)
    model  = _get_model(judge_id, cfg)
    stream = instance["stream"]

    if stream == "factual":
        prompt = _load_prompt(
            cfg["prompts"]["judge_rubric_factual"],
            question_text=instance["question_text"],
            response_text=instance["response_text"],
        )
    else:
        prompt = _load_prompt(
            cfg["prompts"]["judge_rubric_nonfactual"],
            question_text=instance["question_text"],
            response_text=instance["response_text"],
            rubric_text=_rubric_to_text(rubric),
        )

    system_prompt = (
        "You are an expert evaluator. Assess the response carefully and output only valid JSON."
    )

    r = client.call_with_retry(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=cfg["temperature"]["evaluation"],
        max_tokens=cfg["max_tokens"]["judge_cot"],
    )
    cot = r.get("content", "") or ""

    try:
        parsed = _parse_json_from_response(cot)
    except Exception:
        if retry_on_fail:
            # Retry once with an explicit JSON reminder appended
            r2 = client.call_with_retry(
                prompt=prompt + "\n\nIMPORTANT: Output ONLY the JSON object. No other text.",
                system_prompt=system_prompt,
                model=model,
                temperature=0.0,
                max_tokens=cfg["max_tokens"]["judge_cot"],
            )
            cot = r2.get("content", "") or ""
            parsed = _parse_json_from_response(cot)
        else:
            raise

    if stream == "factual":
        criteria_scores, aggregate = _extract_scores_factual(parsed)
    else:
        criteria_scores, aggregate = _extract_scores_nonfactual(parsed, rubric)

    return {
        "instance_id":    instance["instance_id"],
        "judge_id":       judge_id,
        "judge_model":    model,
        "mode":           instance.get("mode", "artificial"),
        "stream":         stream,
        "variant_type":   instance["variant_type"],
        "base_prompt_id": instance["base_prompt_id"],
        "criteria_scores": criteria_scores,
        "aggregate_score": aggregate,
        "cot_trace":      cot,
        "tokens_used":    r.get("output_tokens", 0),
        "evaluated_at":   _now(),
    }


def run_evaluation(cfg: dict, state: ExperimentState,
                   judge_id: str, mode: str = "artificial") -> list[dict]:
    """
    Run rubric evaluation for one judge × mode combination.

    mode: "artificial" | "natural" | "adversarial"
    """
    phase_key = f"evaluation_{judge_id}_{mode}"

    if state.is_phase_complete(phase_key):
        log.info("EvaluationAgent", f"[{judge_id}/{mode}] Already complete — skipping")
        out_path = f"{cfg['paths']['results_evaluations']}/{judge_id}/{mode}_scores.json"
        return _load_json(out_path) if Path(out_path).exists() else []

    instances_path_key = {
        "artificial":  "evaluation_instances",
        "natural":     "evaluation_instances_natural",
        "adversarial": "evaluation_instances_adversarial",
    }[mode]
    instances = _load_json(cfg["paths"][instances_path_key])

    rubric = {}
    if Path(cfg["paths"]["rubric_nonfactual"]).exists():
        rubric = _load_json(cfg["paths"]["rubric_nonfactual"])

    out_path = f"{cfg['paths']['results_evaluations']}/{judge_id}/{mode}_scores.json"
    existing = []
    if Path(out_path).exists():
        existing = _load_json(out_path)
    done_ids = {s["instance_id"] for s in existing}
    results = list(existing)

    todo = [i for i in instances if i["instance_id"] not in done_ids
            and not state.is_complete(phase_key, i["instance_id"])]

    log.info("EvaluationAgent", f"[{judge_id}/{mode}] {len(todo)} instances to evaluate")

    max_workers = cfg["evaluation"].get("max_workers", 4)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_evaluate_instance, inst, judge_id, cfg, rubric): inst["instance_id"]
            for inst in todo
        }
        for future in as_completed(futures):
            iid = futures[future]
            try:
                record = future.result()
                results.append(record)
                _save_json(out_path, results)
                state.mark_complete(phase_key, iid)
                log.info("EvaluationAgent",
                         f"[{judge_id}/{mode}] {iid} → agg={record['aggregate_score']}")
            except Exception as e:
                log.warn("EvaluationAgent", f"[{judge_id}/{mode}] Failed {iid}: {e}")
                state.log_error(phase_key, iid, str(e))

    state.mark_phase_complete(phase_key)
    log.info("EvaluationAgent", f"[{judge_id}/{mode}] complete: {len(results)} scores")
    return results


def run_mitigation_evaluation(cfg: dict, state: ExperimentState,
                               condition: str) -> list[dict]:
    """
    Run evaluation under a mitigation condition (Claude only, Artificial mode).
    condition: "format_agnostic" | "style_norm" | "fixed_rubric"
    """
    phase_key = f"mitigation_{condition}"
    if state.is_phase_complete(phase_key):
        log.info("EvaluationAgent", f"[mitigation/{condition}] Already complete — skipping")
        out_path = f"{cfg['paths']['results_mitigation']}/{condition}_scores.json"
        return _load_json(out_path) if Path(out_path).exists() else []

    instances = _load_json(cfg["paths"]["evaluation_instances"])
    rubric = {}
    if Path(cfg["paths"]["rubric_nonfactual"]).exists():
        rubric = _load_json(cfg["paths"]["rubric_nonfactual"])

    out_path = f"{cfg['paths']['results_mitigation']}/{condition}_scores.json"
    existing = []
    if Path(out_path).exists():
        existing = _load_json(out_path)
    done_ids = {s["instance_id"] for s in existing}
    results = list(existing)

    # Map condition → prompt key override
    prompt_key_override = {
        "format_agnostic": "judge_mitigation_format_agnostic",
        "style_norm":      None,   # uses standard prompt but strips markdown from response
        "fixed_rubric":    "judge_mitigation_fixed_rubric",
    }[condition]

    todo = [i for i in instances
            if i["instance_id"] not in done_ids
            and not state.is_complete(phase_key, i["instance_id"])]

    log.info("EvaluationAgent", f"[mitigation/{condition}] {len(todo)} instances")

    judge_id = "claude"
    model = _get_model(judge_id, cfg)
    client = _get_client(judge_id)

    max_workers = cfg["evaluation"].get("max_workers", 4)

    def _eval_mitigation(instance: dict) -> dict:
        stream = instance["stream"]
        response_text = instance["response_text"]

        # style_norm: strip markdown before judging
        if condition == "style_norm":
            response_text = _strip_markdown(response_text)

        if prompt_key_override:
            if stream == "factual":
                # format_agnostic uses same structure but different rubric phrasing
                prompt = _load_prompt(
                    cfg["prompts"][prompt_key_override],
                    question_text=instance["question_text"],
                    response_text=response_text,
                )
            else:
                prompt = _load_prompt(
                    cfg["prompts"][prompt_key_override],
                    question_text=instance["question_text"],
                    response_text=response_text,
                    rubric_text=_rubric_to_text(rubric),
                )
        else:
            # style_norm: reuse standard prompt with stripped text
            if stream == "factual":
                prompt = _load_prompt(
                    cfg["prompts"]["judge_rubric_factual"],
                    question_text=instance["question_text"],
                    response_text=response_text,
                )
            else:
                prompt = _load_prompt(
                    cfg["prompts"]["judge_rubric_nonfactual"],
                    question_text=instance["question_text"],
                    response_text=response_text,
                    rubric_text=_rubric_to_text(rubric),
                )

        r = client.call_with_retry(
            prompt=prompt,
            system_prompt="You are an expert evaluator. Output only valid JSON.",
            model=model,
            temperature=0.0,
            max_tokens=cfg["max_tokens"]["judge_cot"],
        )
        cot = r.get("content", "") or ""
        try:
            parsed = _parse_json_from_response(cot)
        except Exception:
            r2 = client.call_with_retry(
                prompt=prompt + "\n\nOutput ONLY the JSON object.",
                system_prompt="You are an expert evaluator. Output only valid JSON.",
                model=model,
                temperature=0.0,
                max_tokens=cfg["max_tokens"]["judge_cot"],
            )
            cot = r2.get("content", "") or ""
            parsed = _parse_json_from_response(cot)

        if stream == "factual":
            criteria_scores, aggregate = _extract_scores_factual(parsed)
        else:
            criteria_scores, aggregate = _extract_scores_nonfactual(parsed, rubric)

        return {
            "instance_id":     instance["instance_id"],
            "judge_id":        "claude",
            "judge_model":     model,
            "condition":       condition,
            "mode":            "artificial",
            "stream":          stream,
            "variant_type":    instance["variant_type"],
            "base_prompt_id":  instance["base_prompt_id"],
            "criteria_scores": criteria_scores,
            "aggregate_score": aggregate,
            "cot_trace":       cot,
            "tokens_used":     r.get("output_tokens", 0),
            "evaluated_at":    _now(),
        }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_eval_mitigation, inst): inst["instance_id"] for inst in todo}
        for future in as_completed(futures):
            iid = futures[future]
            try:
                record = future.result()
                results.append(record)
                _save_json(out_path, results)
                state.mark_complete(phase_key, iid)
                log.info("EvaluationAgent",
                         f"[mitigation/{condition}] {iid} → agg={record['aggregate_score']}")
            except Exception as e:
                log.warn("EvaluationAgent", f"[mitigation/{condition}] Failed {iid}: {e}")
                state.log_error(phase_key, iid, str(e))

    state.mark_phase_complete(phase_key)
    return results


def _strip_markdown(text: str) -> str:
    """Remove markdown headers, bullets, and numbered lists from response text."""
    text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)   # headers
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)     # bullets
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)     # numbered lists
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                     # bold
    text = re.sub(r"\*(.+?)\*", r"\1", text)                         # italic
    text = re.sub(r"\n{3,}", "\n\n", text)                            # excess newlines
    return text.strip()
