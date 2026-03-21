"""
Token-count length normalizer.
Normalizes style variants DOWN to the L2 baseline (shortest variant).
Never inflates shorter variants to match longer ones.
"""
import re
from collections import defaultdict

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def get_l2_baseline(variants: list[dict]) -> int:
    """Return token count of the L2 variant in a group."""
    for v in variants:
        if v["formality_level"] == "L2":
            return v["token_count"]
    raise ValueError("No L2 variant found in group")


def group_by_prompt(variants: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for v in variants:
        groups[v["base_prompt_id"]].append(v)
    return dict(groups)


def is_within_tolerance(token_count: int, baseline: int, tolerance: float = 0.10) -> bool:
    return abs(token_count - baseline) / baseline <= tolerance


def trim_to_tokens(text: str, max_tokens: int) -> str:
    """
    Trim text to approximately max_tokens by truncating at sentence boundaries.
    Removes trailing incomplete sentences.
    """
    tokens = _ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    trimmed_tokens = tokens[:max_tokens]
    trimmed_text = _ENCODER.decode(trimmed_tokens)
    # Cut at last sentence boundary
    last_sentence_end = max(
        trimmed_text.rfind("."),
        trimmed_text.rfind("!"),
        trimmed_text.rfind("?"),
    )
    if last_sentence_end > 0:
        return trimmed_text[: last_sentence_end + 1].strip()
    return trimmed_text.strip()


def check_and_flag(
    groups: dict[str, list[dict]],
    tolerance: float = 0.10,
) -> dict[str, list[str]]:
    """
    Returns a dict of prompt_id → list of variant_ids that exceed tolerance.
    Does NOT modify variants; only reports what needs trimming.
    """
    needs_trimming: dict[str, list[str]] = {}
    for prompt_id, variants in groups.items():
        baseline = get_l2_baseline(variants)
        flagged = [
            v["variant_id"]
            for v in variants
            if v["formality_level"] != "L2"
            and not is_within_tolerance(v["token_count"], baseline, tolerance)
        ]
        if flagged:
            needs_trimming[prompt_id] = flagged
    return needs_trimming


def normalize_variants(
    variants: list[dict],
    tolerance: float = 0.10,
    max_iterations: int = 3,
) -> list[dict]:
    """
    Trim L2/L3/L4 variants to within tolerance of the L2 baseline.
    Updates token_count and response_text in-place on copies.
    Flags any variant that cannot be trimmed after max_iterations.
    """
    groups = group_by_prompt(variants)
    result = []

    for prompt_id, group in groups.items():
        baseline = get_l2_baseline(group)
        max_allowed = int(baseline * (1 + tolerance))

        normalized_group = []
        for v in group:
            v = dict(v)  # copy
            v["token_count"] = count_tokens(v["response_text"])

            if v["formality_level"] == "L2":
                v["normalized"] = True
                normalized_group.append(v)
                continue

            for _ in range(max_iterations):
                if is_within_tolerance(v["token_count"], baseline, tolerance):
                    break
                if v["token_count"] > max_allowed:
                    v["response_text"] = trim_to_tokens(v["response_text"], max_allowed)
                    v["token_count"] = count_tokens(v["response_text"])

            v["normalized"] = is_within_tolerance(v["token_count"], baseline, tolerance)
            if not v["normalized"]:
                print(
                    f"[WARN] {v['variant_id']}: could not normalize "
                    f"(tokens={v['token_count']}, baseline={baseline}). "
                    "Flag for manual revision."
                )
            normalized_group.append(v)

        result.extend(normalized_group)

    return result
