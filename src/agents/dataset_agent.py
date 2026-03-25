"""
DatasetAgent v3: Generates rubric, base responses, and style variants (Artificial mode).
Also injects adversarial errors into V-simple and V-abstract variants.

Generator: DeepSeek-V3 (deepseek-chat) via deepseek_client.
Questions are sourced externally via crawl_benchmarks.py (not generated here).
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.api import deepseek_client
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


def _parse_json_from_response(text):
    """Extract JSON from model response, handling markdown code fences and None."""
    if text is None:
        raise ValueError("Model returned empty content")
    text = str(text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for end_char, start_char in [("]", "["), ("}", "{")]:
            last = text.rfind(end_char)
            if last != -1:
                start = text.find(start_char)
                if start != -1 and start < last:
                    try:
                        return json.loads(text[start:last + 1])
                    except json.JSONDecodeError:
                        pass
        raise


# ---------------------------------------------------------------------------
# Phase 1a: Generate rubric for non-factual stream
# ---------------------------------------------------------------------------

def generate_rubric(cfg: dict, state: ExperimentState) -> dict:
    if state.is_phase_complete("rubric_generation"):
        log.info("DatasetAgent", "Rubric generation already complete — skipping")
        return _load_json(cfg["paths"]["rubric_nonfactual"])

    log.info("DatasetAgent", "Generating non-factual rubric via DeepSeek-V3...")
    prompt = _load_prompt(cfg["prompts"]["generate_rubric_nonfactual"])
    r = deepseek_client.call_with_retry(
        prompt=prompt,
        system_prompt="You are a research assistant. Output only valid JSON as instructed.",
        model=deepseek_client.MODEL_V3,
        temperature=cfg["temperature"]["generation"],
        max_tokens=cfg["max_tokens"]["rubric_generation"],
    )
    try:
        rubric = _parse_json_from_response(r["content"])
    except Exception as e:
        log.warn("DatasetAgent", f"JSON parse failed for rubric: {e}")
        log.warn("DatasetAgent", f"Raw: {r['content'][:300]}")
        raise

    _save_json(cfg["paths"]["rubric_nonfactual"], rubric)
    state.mark_phase_complete("rubric_generation")
    log.info("DatasetAgent", f"Rubric saved: {len(rubric.get('criteria', []))} criteria")
    return rubric


# ---------------------------------------------------------------------------
# Phase 1b: Generate base responses
# ---------------------------------------------------------------------------

def generate_base_responses(cfg: dict, state: ExperimentState) -> list[dict]:
    questions = _load_json(cfg["paths"]["base_prompts"])
    out_path = cfg["paths"]["base_responses"]

    existing = []
    if Path(out_path).exists():
        existing = _load_json(out_path)
    done_ids = {r["prompt_id"] for r in existing if r.get("response_text")}

    results = list(existing)
    for q in questions:
        qid = q["question_id"]
        if qid in done_ids or state.is_complete("base_generation", qid):
            log.info("DatasetAgent", f"Base response exists: {qid} — skipping")
            continue

        log.info("DatasetAgent", f"Generating base response for {qid}")
        prompt = _load_prompt(
            cfg["prompts"]["generate_base_response"],
            question_text=q["question_text"],
        )
        r = deepseek_client.call_with_retry(
            prompt=prompt,
            system_prompt="You are a knowledgeable assistant. Answer thoroughly and accurately.",
            model=deepseek_client.MODEL_V3,
            temperature=cfg["temperature"]["generation"],
            max_tokens=cfg["max_tokens"]["base_generation"],
        )
        if not r.get("content"):
            log.warn("DatasetAgent", f"Empty content for {qid} — skipping")
            state.log_error("base_generation", qid, "empty content from API")
            continue

        record = {
            "response_id":       f"{qid}_base",
            "prompt_id":         qid,
            "stream":            q["stream"],
            "question_text":     q["question_text"],
            "response_text":     r["content"],
            "token_count":       r.get("output_tokens", 0),
            "generation_model":  deepseek_client.MODEL_V3,
            "created_at":        _now(),
        }
        results.append(record)
        _save_json(out_path, results)
        state.mark_complete("base_generation", qid)
        log.info("DatasetAgent", f"Base response saved: {qid} ({record['token_count']} tokens)")

    state.mark_phase_complete("base_generation")
    return results


# ---------------------------------------------------------------------------
# Phase 1c: Rewrite to V-simple and V-abstract (Artificial mode)
# ---------------------------------------------------------------------------

def _rewrite_single(q_id: str, stream: str, question_text: str,
                    response_text: str, variant_type: str, cfg: dict) -> dict:
    prompt_key = "rewrite_v_simple" if variant_type == "V-simple" else "rewrite_v_abstract"
    prompt = _load_prompt(cfg["prompts"][prompt_key], response_text=response_text)
    r = deepseek_client.call_with_retry(
        prompt=prompt,
        system_prompt="You are a skilled writing assistant. Follow the style instructions exactly.",
        model=deepseek_client.MODEL_V3,
        temperature=cfg["temperature"]["rewriting"],
        max_tokens=cfg["max_tokens"]["style_rewrite"],
    )
    return {
        "variant_id":      f"{q_id}_{variant_type}",
        "base_prompt_id":  q_id,
        "stream":          stream,
        "question_text":   question_text,
        "variant_type":    variant_type,
        "mode":            "artificial",
        "response_text":   r["content"],
        "token_count":     r.get("output_tokens", 0),
        "rewrite_model":   deepseek_client.MODEL_V3,
        "created_at":      _now(),
    }


def rewrite_to_styles(cfg: dict, state: ExperimentState) -> list[dict]:
    base_responses = _load_json(cfg["paths"]["base_responses"])
    questions_map = {q["question_id"]: q for q in _load_json(cfg["paths"]["base_prompts"])}
    out_path = cfg["paths"]["style_variants"]
    variant_types = cfg["dataset"]["variant_types"]

    existing = []
    if Path(out_path).exists():
        existing = _load_json(out_path)
    done_ids = {v["variant_id"] for v in existing}
    results = list(existing)

    tasks = []
    for base in base_responses:
        if not base.get("response_text"):
            continue
        qid = base["prompt_id"]
        stream = base.get("stream") or questions_map.get(qid, {}).get("stream", "factual")
        question_text = base.get("question_text") or questions_map.get(qid, {}).get("question_text", "")
        for vt in variant_types:
            vid = f"{qid}_{vt}"
            if vid in done_ids or state.is_complete("style_rewriting", vid):
                log.info("DatasetAgent", f"Variant exists: {vid} — skipping")
                continue
            tasks.append((qid, stream, question_text, base["response_text"], vt))

    if not tasks:
        log.info("DatasetAgent", "All variants already written")
        state.mark_phase_complete("style_rewriting")
        return results

    max_workers = cfg["evaluation"].get("max_workers", 4)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_rewrite_single, qid, stream, qt, rt, vt, cfg): (qid, vt)
            for qid, stream, qt, rt, vt in tasks
        }
        for future in as_completed(futures):
            qid, vt = futures[future]
            vid = f"{qid}_{vt}"
            try:
                record = future.result()
                results.append(record)
                _save_json(out_path, results)
                state.mark_complete("style_rewriting", vid)
                log.info("DatasetAgent", f"Variant saved: {vid} ({record['token_count']} tokens)")
            except Exception as e:
                log.warn("DatasetAgent", f"Rewrite failed {vid}: {e}")
                state.log_error("style_rewriting", vid, str(e))

    state.mark_phase_complete("style_rewriting")
    return results


# ---------------------------------------------------------------------------
# Phase 1d: Inject adversarial errors (both V-simple and V-abstract)
# ---------------------------------------------------------------------------

def _inject_single(variant: dict, cfg: dict) -> dict:
    """Inject a factual error into one variant response."""
    prompt = _load_prompt(
        cfg["prompts"]["adversarial_inject"],
        question_text=variant["question_text"],
        response_text=variant["response_text"],
    )
    r = deepseek_client.call_with_retry(
        prompt=prompt,
        system_prompt=(
            "You are a research assistant. Follow the injection instructions exactly. "
            "Output only valid JSON as instructed."
        ),
        model=deepseek_client.MODEL_V3,
        temperature=cfg["temperature"]["adversarial"],
        max_tokens=cfg["max_tokens"]["adversarial"],
    )
    result = _parse_json_from_response(r["content"])
    adv_id = f"adv_{variant['variant_id']}"
    return {
        "adversarial_id":   adv_id,
        "base_prompt_id":   variant["base_prompt_id"],
        "variant_id":       variant["variant_id"],
        "variant_type":     variant["variant_type"],
        "stream":           variant["stream"],
        "question_text":    variant["question_text"],
        "response_text":    result.get("flawed_response", ""),
        "error_type":       result.get("error_type", "factual"),
        "error_description": result.get("error_description", ""),
        "error_location":   result.get("error_location", ""),
        "injection_model":  deepseek_client.MODEL_V3,
        "created_at":       _now(),
    }


def inject_adversarial_errors(cfg: dict, state: ExperimentState) -> list[dict]:
    """
    For each stream, select n_per_type variants of each type (V-simple, V-abstract)
    and inject a factual error. Saves to data/dataset/adversarial.json.
    """
    if state.is_phase_complete("adversarial_injection"):
        log.info("DatasetAgent", "Adversarial injection already complete — skipping")
        return _load_json(cfg["paths"]["adversarial"])

    variants = _load_json(cfg["paths"]["style_variants"])
    out_path = cfg["paths"]["adversarial"]
    adversarial_per_stream = cfg["dataset"].get("adversarial_per_stream", 10)
    n_per_type = adversarial_per_stream // 2  # split equally between V-simple and V-abstract

    existing = []
    if Path(out_path).exists():
        existing = _load_json(out_path)
    done_ids = {a["adversarial_id"] for a in existing}
    results = list(existing)

    # Select candidates: n_per_type from each (stream × variant_type) cell
    import random
    rng = random.Random(42)
    tasks = []
    for stream in ("factual", "non_factual"):
        for vt in ("V-simple", "V-abstract"):
            pool = [v for v in variants
                    if v["stream"] == stream
                    and v["variant_type"] == vt
                    and v.get("response_text")]
            selected = rng.sample(pool, min(n_per_type, len(pool)))
            for variant in selected:
                adv_id = f"adv_{variant['variant_id']}"
                if adv_id in done_ids or state.is_complete("adversarial_injection", adv_id):
                    log.info("DatasetAgent", f"Adversarial exists: {adv_id} — skipping")
                    continue
                tasks.append(variant)

    if not tasks:
        log.info("DatasetAgent", "All adversarial instances already written")
        state.mark_phase_complete("adversarial_injection")
        return results

    max_workers = cfg["evaluation"].get("max_workers", 4)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_inject_single, v, cfg): v["variant_id"] for v in tasks}
        for future in as_completed(futures):
            vid = futures[future]
            try:
                record = future.result()
                results.append(record)
                _save_json(out_path, results)
                state.mark_complete("adversarial_injection", record["adversarial_id"])
                log.info("DatasetAgent", f"Adversarial saved: {record['adversarial_id']}")
            except Exception as e:
                log.warn("DatasetAgent", f"Adversarial injection failed {vid}: {e}")
                state.log_error("adversarial_injection", f"adv_{vid}", str(e))

    state.mark_phase_complete("adversarial_injection")
    return results
