"""
NaturalGenerator v3: Generates Natural mode variants (Mode B).

For each question:
  - V-natural-simple: DeepSeek-V3 with a plain-prose system prompt
  - V-natural-abstract: DeepSeek-R1 with a structured reasoning system prompt
    The <think> block is automatically stripped by deepseek_client; only the
    final structured answer is stored. think_block_words is logged for H7 analysis.

Writes to: data/dataset/natural_variants.json
"""
import json
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


def _generate_single(question: dict, variant_type: str, cfg: dict) -> dict:
    """Generate one natural variant for a question."""
    qid = question["question_id"]

    if variant_type == "V-natural-simple":
        prompt_key = "generate_natural_simple"
        model = cfg["models"]["natural_simple_generator"]   # deepseek-chat (V3)
        temperature = cfg["temperature"]["natural"]
    else:
        prompt_key = "generate_natural_abstract"
        model = cfg["models"]["natural_abstract_generator"] # deepseek-reasoner (R1)
        temperature = cfg["temperature"]["natural"]         # ignored for R1 by client

    system_prompt = _load_prompt(cfg["prompts"][prompt_key])
    user_prompt = question["question_text"]

    r = deepseek_client.call_with_retry(
        prompt=user_prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        max_tokens=cfg["max_tokens"]["natural_generation"],
    )

    vid = f"{qid}_{variant_type}"
    return {
        "variant_id":         vid,
        "base_prompt_id":     qid,
        "stream":             question["stream"],
        "question_text":      question["question_text"],
        "variant_type":       variant_type,
        "mode":               "natural",
        "generator_model":    model,
        "response_text":      r.get("content", ""),
        "token_count":        r.get("output_tokens", 0),
        "think_block_words":  r.get("think_block_words", 0),  # >0 only for R1
        "created_at":         _now(),
    }


def generate_natural_variants(cfg: dict, state: ExperimentState) -> list[dict]:
    """
    Generate V-natural-simple and V-natural-abstract for every question.
    Checkpoint-safe: skips already-completed variants.
    """
    if state.is_phase_complete("natural_generation"):
        log.info("NaturalGenerator", "Natural generation already complete — skipping")
        return _load_json(cfg["paths"]["natural_variants"])

    questions = _load_json(cfg["paths"]["base_prompts"])
    out_path = cfg["paths"]["natural_variants"]

    existing = []
    if Path(out_path).exists():
        existing = _load_json(out_path)
    done_ids = {v["variant_id"] for v in existing if v.get("response_text")}
    results = list(existing)

    tasks = []
    natural_types = cfg["dataset"]["natural_variant_types"]
    for question in questions:
        for vt in natural_types:
            vid = f"{question['question_id']}_{vt}"
            if vid in done_ids or state.is_complete("natural_generation", vid):
                log.info("NaturalGenerator", f"Natural variant exists: {vid} — skipping")
                continue
            tasks.append((question, vt))

    if not tasks:
        log.info("NaturalGenerator", "All natural variants already written")
        state.mark_phase_complete("natural_generation")
        return results

    log.info("NaturalGenerator", f"{len(tasks)} natural variants to generate")

    # Run V-natural-simple and V-natural-abstract in parallel
    # Note: R1 calls are slower (think phase); 4 workers is fine
    max_workers = cfg["evaluation"].get("max_workers", 4)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_generate_single, q, vt, cfg): (q["question_id"], vt)
            for q, vt in tasks
        }
        for future in as_completed(futures):
            qid, vt = futures[future]
            vid = f"{qid}_{vt}"
            try:
                record = future.result()
                if not record.get("response_text"):
                    log.warn("NaturalGenerator", f"Empty response for {vid} — skipping")
                    state.log_error("natural_generation", vid, "empty content")
                    continue
                results.append(record)
                _save_json(out_path, results)
                state.mark_complete("natural_generation", vid)
                think_note = (
                    f" (think_block_words={record['think_block_words']})"
                    if record["think_block_words"] > 0 else ""
                )
                log.info("NaturalGenerator",
                         f"Saved: {vid} ({record['token_count']} tokens){think_note}")
            except Exception as e:
                log.warn("NaturalGenerator", f"Generation failed {vid}: {e}")
                state.log_error("natural_generation", vid, str(e))

    state.mark_phase_complete("natural_generation")
    log.info("NaturalGenerator", f"Natural generation complete: {len(results)} variants")
    return results
