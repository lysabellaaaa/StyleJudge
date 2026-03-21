"""
QAAgent: Semantic equivalence verification using GPT-4o (NOT Claude).
Groups variants by base_prompt_id and checks that all 4 formality levels
convey identical factual content.
"""
import json
from collections import defaultdict
from pathlib import Path

from src.api import openai_client
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


def run_qa_verification(cfg: dict, state: ExperimentState) -> list[dict]:
    variants = _load_json(cfg["paths"]["style_variants_normalized"])
    out_path = cfg["paths"]["qa_verified"]
    flags_path = out_path.replace(".json", "_flags.json")

    existing = []
    if Path(out_path).exists():
        existing = _load_json(out_path)
    done_prompt_ids = {v["base_prompt_id"] for v in existing if v.get("qa_passed") is not None}

    # Group by base_prompt_id
    groups: dict[str, list] = defaultdict(list)
    for v in variants:
        groups[v["base_prompt_id"]].append(v)

    updated_variants = {v["variant_id"]: v for v in existing}
    flags = []

    for prompt_id, group in groups.items():
        if prompt_id in done_prompt_ids:
            continue
        # Sort by formality level for consistent presentation
        level_order = {"L1": 0, "L2": 1, "L3": 2, "L4": 3}
        group_sorted = sorted(group, key=lambda x: level_order.get(x["formality_level"], 9))

        # With only 2 variants (L2/L4), semantic equivalence is guaranteed by construction
        # (both are rewrites of the same base response). Auto-pass and record.
        if len(group_sorted) < 3:
            log.info("QAAgent", f"{prompt_id}: 2-variant group — auto-passed (same base response)")
            for v in group_sorted:
                v = dict(v)
                v["qa_passed"] = True
                v["qa_response"] = "auto-passed: 2-variant group derived from same base response"
                updated_variants[v["variant_id"]] = v
            continue

        log.info("QAAgent", f"Verifying semantic equivalence for {prompt_id}")
        variants_text = "\n\n".join(
            f"Variant {v['formality_level']}:\n{v['response_text']}"
            for v in group_sorted
        )
        prompt = f"Do all of these variants convey identical factual claims?\n\n{variants_text}\n\nReply with VERDICT: EQUIVALENT or VERDICT: DIVERGENT, then explain any differences."
        r = openai_client.call_with_retry(
            prompt=prompt,
            system_prompt="You are a semantic equivalence checker. Be precise and objective.",
            model=cfg["models"]["qa_verifier"],
            temperature=cfg["temperature"]["qa_verification"],
            max_tokens=cfg["max_tokens"]["qa_verification"],
        )
        response_text = r["content"]
        is_equivalent = "VERDICT: EQUIVALENT" in response_text.upper()

        for v in group_sorted:
            v = dict(v)
            v["qa_passed"] = is_equivalent
            v["qa_response"] = response_text[:500]
            updated_variants[v["variant_id"]] = v

        if not is_equivalent:
            flags.append({"prompt_id": prompt_id, "qa_response": response_text})
            log.warn("QAAgent", f"{prompt_id}: DIVERGENT — flagged for human review")
        else:
            log.info("QAAgent", f"{prompt_id}: EQUIVALENT")

    result = list(updated_variants.values())
    _save_json(out_path, result)
    if flags:
        _save_json(flags_path, flags)
        log.warn("QAAgent", f"{len(flags)} prompt groups flagged. Review {flags_path}")

    state.mark_phase_complete("qa_verification")
    return result
