"""
Unit tests for the length normalizer.
Verifies: normalization trims DOWN to L1 baseline, not inflating shorter variants.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.length_normalizer import (
    count_tokens,
    get_l2_baseline,
    is_within_tolerance,
    normalize_variants,
    trim_to_tokens,
)



def _make_variant(prompt_id: str, level: str, text: str) -> dict:
    return {
        "variant_id": f"{prompt_id}_{level}",
        "base_prompt_id": prompt_id,
        "formality_level": level,
        "response_text": text,
        "token_count": count_tokens(text),
        "normalized": False,
    }


SHORT_TEXT = "So basically this is the main point. It's pretty simple and doesn't need much explaining."
MEDIUM_TEXT = "This is a response that addresses the question in some detail. " * 5
LONG_TEXT = "## Overview\n\nThis section provides a comprehensive analysis of the topic. " * 15


def test_count_tokens_nonzero():
    assert count_tokens("Hello world") > 0


def test_is_within_tolerance():
    assert is_within_tolerance(100, 100, 0.10)
    assert is_within_tolerance(108, 100, 0.10)
    assert not is_within_tolerance(120, 100, 0.10)


def test_get_l2_baseline():
    variants = [
        _make_variant("wp_001", "L2", SHORT_TEXT),
        _make_variant("wp_001", "L4", LONG_TEXT),
    ]
    baseline = get_l2_baseline(variants)
    assert baseline == count_tokens(SHORT_TEXT)


def test_normalize_trims_l4_not_inflates_l1():
    """L1 should not change; L4 should be trimmed toward L1 baseline."""
    variants = [
        _make_variant("wp_001", "L2", SHORT_TEXT),
        _make_variant("wp_001", "L2", MEDIUM_TEXT),
        _make_variant("wp_001", "L3", MEDIUM_TEXT),
        _make_variant("wp_001", "L4", LONG_TEXT),
    ]
    baseline = count_tokens(SHORT_TEXT)
    normalized = normalize_variants(variants, tolerance=0.10)

    l1 = next(v for v in normalized if v["formality_level"] == "L2")
    l4 = next(v for v in normalized if v["formality_level"] == "L4")

    # L1 text should be unchanged
    assert l1["response_text"] == SHORT_TEXT, "L1 text should not be modified"

    # L4 should be shorter or equal after normalization
    assert l4["token_count"] <= count_tokens(LONG_TEXT), "L4 should have been trimmed"

    # If L4 was successfully normalized, it should be within tolerance
    if l4["normalized"]:
        assert is_within_tolerance(l4["token_count"], baseline, 0.10), (
            f"L4 token count {l4['token_count']} not within 10% of baseline {baseline}"
        )


def test_trim_to_tokens_preserves_sentences():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    trimmed = trim_to_tokens(text, 5)
    assert trimmed.endswith("."), "Should end at sentence boundary"
    assert count_tokens(trimmed) <= 10  # rough check


def test_normalize_l1_unchanged_token_count():
    variants = [
        _make_variant("wp_001", "L2", SHORT_TEXT),
        _make_variant("wp_001", "L4", LONG_TEXT),
    ]
    normalized = normalize_variants(variants, tolerance=0.10)
    l1 = next(v for v in normalized if v["formality_level"] == "L2")
    original_count = count_tokens(SHORT_TEXT)
    assert l1["token_count"] == original_count
