"""
DatasetAgent: handles all dataset construction sub-tasks.
  1. generate_base_responses   — Claude Sonnet 4.6, temp=0
  2. rewrite_to_styles         — Claude Sonnet 4.6, temp=0.3, parallel across L1-L4
  3. normalize_lengths         — local, no API (tiktoken)
  4. inject_adversarial_errors — Claude Sonnet 4.6, temp=0
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from src.api import anthropic_client
from src.utils import logger as log
from src.utils.length_normalizer import count_tokens, normalize_variants
from src.utils.state import ExperimentState


def _load_prompt(path: str, **kwargs) -> str:
    text = Path(path).read_text(encoding="utf-8")
    for key, val in kwargs.items():
        text = text.replace("{" + key + "}", str(val))
    return text


def _load_json(path: str) -> list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_json(path: str, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_base_responses(cfg: dict, state: ExperimentState) -> list[dict]:
    base_prompts = _load_json(cfg["paths"]["base_prompts"])
    out_path = cfg["paths"]["base_responses"]

    existing = []
    if Path(out_path).exists():
        existing = _load_json(out_path)
    done_ids = {r["prompt_id"] for r in existing}

    results = list(existing)
    for prompt in base_prompts:
        pid = prompt["prompt_id"]
        if state.is_complete("base_generation", pid):
            continue
        log.info("DatasetAgent", f"Generating base response for {pid}")
        prompt_text = _load_prompt(
            cfg["prompts"]["base_generation"],
            prompt_text=prompt["prompt_text"],
        )
        r = anthropic_client.call_with_retry(
            prompt=prompt_text,
            system_prompt="You are an expert respondent. Write factually accurate, complete responses.",
            model=cfg["models"]["base_generator"],
            temperature=cfg["temperature"]["generation"],
            max_tokens=cfg["max_tokens"]["base_generation"],
        )
        record = {
            "response_id": f"{pid}_base",
            "prompt_id": pid,
            "domain": prompt["domain"],
            "prompt_text": prompt["prompt_text"],
            "response_text": r["content"],
            "token_count": count_tokens(r["content"]),
            "generation_model": cfg["models"]["base_generator"],
            "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        }
        results.append(record)
        _save_json(out_path, results)
        state.mark_complete("base_generation", pid)
        log.info("DatasetAgent", f"Base response saved: {pid} ({record['token_count']} tokens)")

    return results


def rewrite_to_styles(cfg: dict, state: ExperimentState) -> list[dict]:
    base_responses = _load_json(cfg["paths"]["base_responses"])
    base_prompts_map = {p["prompt_id"]: p for p in _load_json(cfg["paths"]["base_prompts"])}
    out_path = cfg["paths"]["style_variants"]

    existing = []
    if Path(out_path).exists():
        existing = _load_json(out_path)
    done_ids = {v["variant_id"] for v in existing}
    results = list(existing)

    def rewrite_one(base: dict, level: str) -> dict | None:
        variant_id = f"{base['prompt_id']}_{level}"
        if variant_id in done_ids or state.is_complete("style_rewriting", variant_id):
            return None
        log.info("DatasetAgent", f"Rewriting {variant_id}")
        prompt = _load_prompt(
            cfg["prompts"]["style_rewrite"][level],
            response_text=base["response_text"],
        )
        r = anthropic_client.call_with_retry(
            prompt=prompt,
            system_prompt="You are a style rewriter. Rewrite text exactly as instructed, preserving all factual content.",
            model=cfg["models"]["style_rewriter"],
            temperature=cfg["temperature"]["rewriting"],
            max_tokens=cfg["max_tokens"]["style_rewrite"],
        )
        return {
            "variant_id": variant_id,
            "base_prompt_id": base["prompt_id"],
            "domain": base["domain"],
            "prompt_text": base["prompt_text"],
            "formality_level": level,
            "response_text": r["content"],
            "token_count": count_tokens(r["content"]),
            "normalized": False,
            "qa_passed": None,
            "irr_label": None,
            "irr_kappa": None,
            "generation_model": cfg["models"]["base_generator"],
            "rewrite_model": cfg["models"]["style_rewriter"],
            "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        }

    levels = cfg["dataset"]["formality_levels"]
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            ex.submit(rewrite_one, base, level): (base["prompt_id"], level)
            for base in base_responses
            for level in levels
        }
        for fut in as_completed(futures):
            variant = fut.result()
            if variant:
                results.append(variant)
                _save_json(out_path, results)
                state.mark_complete("style_rewriting", variant["variant_id"])

    return results


def normalize_lengths(cfg: dict, state: ExperimentState) -> list[dict]:
    variants = _load_json(cfg["paths"]["style_variants"])
    out_path = cfg["paths"]["style_variants_normalized"]
    tolerance = cfg["dataset"]["length_normalization_tolerance"]
    max_iter = cfg["dataset"]["length_normalization_max_iterations"]

    log.info("DatasetAgent", "Running length normalization (trimming to L1 baseline)...")
    normalized = normalize_variants(variants, tolerance=tolerance, max_iterations=max_iter)
    _save_json(out_path, normalized)
    state.mark_complete("length_normalization", "done")
    log.info("DatasetAgent", f"Length normalization complete: {len(normalized)} variants")
    return normalized


def inject_adversarial_errors(cfg: dict, state: ExperimentState) -> list[dict]:
    variants = _load_json(cfg["paths"]["style_variants_normalized"])
    out_path = cfg["paths"]["adversarial"]

    existing = []
    if Path(out_path).exists():
        existing = _load_json(out_path)
    done_ids = {a["adversarial_id"] for a in existing}
    results = list(existing)

    # Select L4 variants for adversarial injection
    l4_variants = [v for v in variants if v["formality_level"] == "L4" and v.get("qa_passed") is not False]
    target_n = cfg["dataset"]["n_adversarial"]
    candidates = l4_variants[:target_n]

    for i, variant in enumerate(candidates):
        adv_id = f"adv_{i+1:03d}"
        if adv_id in done_ids or state.is_complete("adversarial_injection", adv_id):
            continue
        log.info("DatasetAgent", f"Injecting adversarial error into {variant['variant_id']} → {adv_id}")
        prompt = _load_prompt(
            cfg["prompts"]["adversarial_inject"],
            question=variant["prompt_text"],
            response_text=variant["response_text"],
        )
        r = anthropic_client.call_with_retry(
            prompt=prompt,
            system_prompt="You are a research assistant creating controlled test stimuli. Follow instructions precisely.",
            model=cfg["models"]["adversarial_injector"],
            temperature=cfg["temperature"]["generation"],
            max_tokens=cfg["max_tokens"]["adversarial_injection"],
        )
        content = r["content"]
        # Parse the response: modified text + metadata
        error_type = _extract_field(content, "ERROR_TYPE")
        error_desc = _extract_field(content, "ERROR_DESCRIPTION")
        error_loc = _extract_field(content, "ERROR_LOCATION")
        # The modified response is everything before ERROR_TYPE line
        split_idx = content.find("ERROR_TYPE:")
        flawed_text = content[:split_idx].strip() if split_idx > 0 else content

        # Find correct L1 counterpart
        correct_l1 = next(
            (v for v in variants
             if v["base_prompt_id"] == variant["base_prompt_id"] and v["formality_level"] == "L1"),
            None,
        )
        record = {
            "adversarial_id": adv_id,
            "base_prompt_id": variant["base_prompt_id"],
            "domain": variant["domain"],
            "prompt_text": variant["prompt_text"],
            "correct_variant_id": correct_l1["variant_id"] if correct_l1 else None,
            "flawed_variant_id": f"{variant['variant_id']}_adv",
            "flawed_response_text": flawed_text,
            "error_type": error_type or "unknown",
            "error_description": error_desc or "see response",
            "error_location_hint": error_loc or "see response",
            "injection_model": cfg["models"]["adversarial_injector"],
        }
        results.append(record)
        _save_json(out_path, results)
        state.mark_complete("adversarial_injection", adv_id)

    log.info("DatasetAgent", f"Adversarial injection complete: {len(results)} instances")
    return results


def _extract_field(text: str, field: str) -> str | None:
    match = re.search(rf"{field}:\s*(.+)", text)
    return match.group(1).strip() if match else None
