"""
API client unit tests (mock-based — no real API calls).
Tests: retry logic, score extraction, error handling.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.agents.evaluation_agent import extract_score


# ─── Score Extraction Tests ──────────────────────────────────────────────────

def test_extract_score_standard():
    assert extract_score("The response is good. Score: 4") == 4


def test_extract_score_rating_format():
    assert extract_score("Rating: 3\nThis is my evaluation.") == 3


def test_extract_score_fraction_format():
    assert extract_score("I give this 5/5 for its precision.") == 5


def test_extract_score_natural_language():
    assert extract_score("I would give a score of 2 for this response.") == 2


def test_extract_score_final_score_label():
    assert extract_score("Final Score: 4\nOverall well done.") == 4


def test_extract_score_raises_on_missing():
    with pytest.raises(ValueError):
        extract_score("This is a great response with no score marker.")


def test_extract_score_takes_first_match():
    # Only valid 1-5 scores should match; "10" should not match single-digit pattern
    result = extract_score("Score: 3. Earlier I mentioned 10 points.")
    assert result == 3


def test_extract_score_out_of_range_not_matched():
    """Score: 6 should not match [1-5] pattern."""
    with pytest.raises(ValueError):
        extract_score("Score: 6")


# ─── State Tests (no API calls needed) ───────────────────────────────────────

def test_state_atomic_write(tmp_path):
    from src.utils.state import ExperimentState
    state_file = str(tmp_path / "state.json")
    state = ExperimentState.from_file(state_file)
    state.mark_complete("evaluation_gpt4o", "eval_001")
    assert state.is_complete("evaluation_gpt4o", "eval_001")
    assert not state.is_complete("evaluation_gpt4o", "eval_002")


def test_state_phase_complete(tmp_path):
    from src.utils.state import ExperimentState
    state_file = str(tmp_path / "state.json")
    state = ExperimentState.from_file(state_file)
    assert not state.is_phase_complete("evaluation_gpt4o")
    state.mark_phase_complete("evaluation_gpt4o")
    assert state.is_phase_complete("evaluation_gpt4o")


def test_state_persistence(tmp_path):
    from src.utils.state import ExperimentState
    state_file = str(tmp_path / "state.json")
    s1 = ExperimentState.from_file(state_file)
    s1.mark_complete("base_generation", "wp_001")

    # Load fresh instance from same file
    s2 = ExperimentState.from_file(state_file)
    assert s2.is_complete("base_generation", "wp_001")


def test_state_thread_safety(tmp_path):
    """Multiple threads marking different items should not corrupt state."""
    import threading
    from src.utils.state import ExperimentState
    state_file = str(tmp_path / "state.json")
    state = ExperimentState.from_file(state_file)

    def mark(item_id):
        state.mark_complete("evaluation_gpt4o", item_id)

    threads = [threading.Thread(target=mark, args=(f"eval_{i:03d}",)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    completed = state.get_completed("evaluation_gpt4o")
    assert len(completed) == 20, f"Expected 20 items, got {len(completed)}"
