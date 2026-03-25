"""
PairwiseAgent v3: Blind pairwise comparison with multi-judge and multi-mode support.

Isolation guarantee:
- Fresh API call per comparison with no prior scoring context
- Prompt contains only: question + Response A + Response B + criteria
- A/B assignment randomised per comparison; recorded for post-hoc decoding
- Runs only after rubric scoring phase is complete

Output path: results/pairwise/{judge_id}_{mode}.json
"""
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.api import anthropic_client, openai_client
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


def _parse_json_from_response(text: str) -> dict:
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


def _get_client(judge_id: str):
    return {"claude": anthropic_client, "gpt4o": openai_client}[judge_id]


def _compare_pair(question_id: str, stream: str, question_text: str,
                  variants: dict, judge_id: str, mode: str, cfg: dict) -> dict:
    """
    variants: {"V-simple": "<text>", "V-abstract": "<text>"}
              or {"V-natural-simple": ..., "V-natural-abstract": ...}

    Randomises A/B assignment to control position bias.
    Returns result with preferred_variant decoded.
    """
    variant_keys = list(variants.keys())
    rng = random.Random()  # thread-local RNG; seeded by system
    rng.shuffle(variant_keys)
    assignment = {"A": variant_keys[0], "B": variant_keys[1]}

    prompt = _load_prompt(
        cfg["prompts"]["judge_pairwise"],
        question_text=question_text,
        response_a=variants[assignment["A"]],
        response_b=variants[assignment["B"]],
    )

    client = _get_client(judge_id)
    model = cfg["models"]["judges"][judge_id]
    system_prompt = "You are an expert evaluator. Output only valid JSON as instructed."

    r = client.call_with_retry(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=cfg["temperature"]["pairwise"],
        max_tokens=cfg["max_tokens"]["pairwise"],
    )
    raw = r.get("content", "") or ""

    try:
        parsed = _parse_json_from_response(raw)
    except Exception:
        r2 = client.call_with_retry(
            prompt=prompt + "\n\nOutput ONLY the JSON object. No text before or after.",
            system_prompt=system_prompt,
            model=model,
            temperature=0.0,
            max_tokens=cfg["max_tokens"]["pairwise"],
        )
        raw = r2.get("content", "") or ""
        parsed = _parse_json_from_response(raw)

    preferred_label   = parsed.get("preferred", "A")
    preferred_variant = assignment.get(preferred_label, assignment["A"])

    return {
        "question_id":        question_id,
        "stream":             stream,
        "judge_id":           judge_id,
        "judge_model":        model,
        "mode":               mode,
        "assignment":         assignment,
        "preferred_label":    preferred_label,
        "preferred_variant":  preferred_variant,
        "preference_strength": parsed.get("preference_strength", ""),
        "primary_reason":     parsed.get("primary_reason", ""),
        "a_strengths":        parsed.get("a_strengths", ""),
        "b_strengths":        parsed.get("b_strengths", ""),
        "cot_trace":          raw,
        "evaluated_at":       _now(),
    }


def run_pairwise_evaluation(cfg: dict, state: ExperimentState,
                             judge_id: str, mode: str = "artificial") -> list[dict]:
    """
    Run pairwise evaluation for one judge × mode combination.

    judge_id: "claude" | "gpt4o"   (Llama excluded: Groq TPM too low for long prompts)
    mode: "artificial" | "natural"
    """
    phase_key = f"pairwise_{judge_id}_{mode}"

    if state.is_phase_complete(phase_key):
        log.info("PairwiseAgent", f"[{judge_id}/{mode}] Already complete — skipping")
        out_path = Path(cfg["paths"]["results_pairwise"]) / f"{judge_id}_{mode}.json"
        return _load_json(str(out_path)) if out_path.exists() else []

    # Load variant data for this mode
    variants_path_key = {
        "artificial": "style_variants",
        "natural":    "natural_variants",
    }[mode]
    variants_data = _load_json(cfg["paths"][variants_path_key])

    # Group by question_id — need exactly 2 variants per question
    groups: dict[str, dict] = {}
    for v in variants_data:
        qid = v["base_prompt_id"]
        if not v.get("response_text"):
            continue
        if qid not in groups:
            groups[qid] = {
                "stream":        v["stream"],
                "question_text": v["question_text"],
                "variants":      {},
            }
        groups[qid]["variants"][v["variant_type"]] = v["response_text"]

    # Keep only complete pairs
    complete_groups = {
        qid: info for qid, info in groups.items()
        if len(info["variants"]) == 2
    }

    out_path = Path(cfg["paths"]["results_pairwise"]) / f"{judge_id}_{mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if out_path.exists():
        existing = _load_json(str(out_path))
    done_qids = {r["question_id"] for r in existing}
    results = list(existing)

    pending = [
        (qid, info) for qid, info in complete_groups.items()
        if qid not in done_qids and not state.is_complete(phase_key, qid)
    ]

    log.info("PairwiseAgent", f"[{judge_id}/{mode}] {len(pending)} comparisons to run")

    max_workers = cfg["evaluation"].get("max_workers", 4)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                _compare_pair,
                qid, info["stream"], info["question_text"],
                info["variants"], judge_id, mode, cfg
            ): qid
            for qid, info in pending
        }
        for future in as_completed(futures):
            qid = futures[future]
            try:
                result = future.result()
                results.append(result)
                _save_json(str(out_path), results)
                state.mark_complete(phase_key, qid)
                log.info("PairwiseAgent",
                         f"[{judge_id}/{mode}] {qid} → "
                         f"prefers {result['preferred_variant']} ({result['preference_strength']})")
            except Exception as e:
                log.warn("PairwiseAgent", f"[{judge_id}/{mode}] Failed {qid}: {e}")
                state.log_error(phase_key, qid, str(e))

    state.mark_phase_complete(phase_key)
    log.info("PairwiseAgent", f"[{judge_id}/{mode}] complete: {len(results)} comparisons")
    return results
