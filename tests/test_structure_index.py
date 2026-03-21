"""
Critical validation: Structure Index regex must discriminate between
structured and casual CoT examples BEFORE it is used as a metric.
If this test fails, revise STRUCTURE_PATTERNS in reasoning_depth.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.reasoning_depth import compute_structure_index

# Known-structured CoT examples (judge responses to structured candidates)
STRUCTURED_COTS = [
    """## Factual Accuracy
- The response correctly identifies the key criterion for assessment.
- First, correctness: the claim is well-supported. Second, completeness: several dimensions are addressed.
The framework employed here is sound. Therefore, the response meets the required standard.
Consequently, I would rate this highly.""",

    """The criterion for evaluation here involves three dimensions:
1. Factual accuracy — the response addresses this aspect correctly.
2. Logical coherence — furthermore, the argument follows a clear framework.
3. Completeness — however, one aspect is missing.
Therefore, the final assessment reflects these considerations.""",

    """## Assessment
**Criterion 1:** Factual accuracy — the claims are correct.
**Criterion 2:** Logical coherence — the argument is structured well.
- First, the premise is established correctly.
- Second, the conclusion follows logically.
Consequently, this response demonstrates strong performance across all dimensions.""",
]

# Known-casual CoT examples (judge responses to casual candidates)
CASUAL_COTS = [
    "This response seems pretty good overall. It gets the main idea right and doesn't miss anything major. I think it's solid.",

    "So the answer here is basically correct. The person explains it well enough and it's easy to follow. Nothing really wrong with it.",

    "Seems like a decent answer to me. It covers what's needed without going off track. I'd say it's a good response.",
]


def test_structure_index_discriminates():
    structured_scores = [compute_structure_index(cot) for cot in STRUCTURED_COTS]
    casual_scores = [compute_structure_index(cot) for cot in CASUAL_COTS]

    mean_structured = sum(structured_scores) / len(structured_scores)
    mean_casual = sum(casual_scores) / len(casual_scores)

    print(f"Structured CoT Structure Index scores: {structured_scores} (mean={mean_structured:.1f})")
    print(f"Casual CoT Structure Index scores:     {casual_scores} (mean={mean_casual:.1f})")

    assert mean_structured > mean_casual, (
        f"Structure Index failed to discriminate: "
        f"structured mean ({mean_structured:.1f}) should be > casual mean ({mean_casual:.1f}). "
        "Revise STRUCTURE_PATTERNS in src/metrics/reasoning_depth.py"
    )
    # Each individual structured CoT should score higher than each casual CoT
    for s_score in structured_scores:
        for c_score in casual_scores:
            assert s_score >= c_score, (
                f"Structured score {s_score} not >= casual score {c_score}. "
                "Patterns may be too narrow."
            )


def test_structure_index_zero_on_empty():
    assert compute_structure_index("") == 0


def test_structure_index_bullets():
    text = "- Item one\n- Item two\n- Item three"
    assert compute_structure_index(text) >= 3


def test_structure_index_rubric_language():
    text = "The criterion for evaluation is clear. Each dimension should be assessed separately."
    score = compute_structure_index(text)
    assert score >= 2, f"Expected >=2 for rubric language, got {score}"


def test_structure_index_enumeration():
    text = "First, we assess accuracy. Second, we consider completeness. Therefore, the score is 4."
    score = compute_structure_index(text)
    assert score >= 2
