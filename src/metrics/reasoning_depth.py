"""
Reasoning Depth and Structure Index computation.
Applied to judge Chain-of-Thought traces.
"""
import re

import textstat

# Authoritative regex patterns for structure detection in judge CoT.
# Validated in tests/test_structure_index.py before use as a metric.
STRUCTURE_PATTERNS = [
    r"^\s*[-*•]\s",                              # bullet markers
    r"^\s*\d+\.\s",                              # numbered lists
    r"^\s*#{1,3}\s",                             # markdown headers
    r"\b(criterion|criteria|dimension|aspect|rubric|framework)\b",
    r"\b(first|second|third|finally),\s",        # enumeration language
    r"\b(however|therefore|consequently|furthermore)\b",
]


def compute_structure_index(cot_text: str) -> int:
    """Sum of regex matches across all STRUCTURE_PATTERNS in the CoT."""
    total = 0
    for pattern in STRUCTURE_PATTERNS:
        flags = re.IGNORECASE | re.MULTILINE
        total += len(re.findall(pattern, cot_text, flags=flags))
    return total


def compute_reasoning_depth(cot_text: str) -> dict:
    """
    Returns {"word_count": int, "structure_index": int, "flesch_kincaid_grade": float}.
    """
    word_count = len(cot_text.split())
    structure_index = compute_structure_index(cot_text)
    try:
        fk_grade = textstat.flesch_kincaid_grade(cot_text)
    except Exception:
        fk_grade = -1.0
    return {
        "word_count": word_count,
        "structure_index": structure_index,
        "flesch_kincaid_grade": round(fk_grade, 2),
    }
