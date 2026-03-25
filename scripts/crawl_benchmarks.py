"""
crawl_benchmarks.py — One-time dataset builder for StyleJudge v3.

Pulls from established HuggingFace benchmarks and converts to open-ended question format.
Output: data/raw/base_prompts.json (100 questions: 50 factual + 50 non-factual)

Usage:
    py -3 scripts/crawl_benchmarks.py [--limit N] [--seed 42]

Factual sources  (n=50): MMLU (30) + ARC-Challenge (20)
Non-factual sources (n=50): ETHICS (35) + SCRUPLES (8) + Moral Stories (7)
"""
import argparse
import json
import random
import re
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# MMLU subjects sampled: mix of science, humanities, professional knowledge
# Chosen so that no single domain dominates and questions require ≥3 key facts
# ---------------------------------------------------------------------------
MMLU_SUBJECTS = [
    ("cais/mmlu", "college_biology"),
    ("cais/mmlu", "high_school_microeconomics"),
    ("cais/mmlu", "formal_logic"),
    ("cais/mmlu", "medical_genetics"),
    ("cais/mmlu", "high_school_world_history"),
    ("cais/mmlu", "high_school_biology"),
]
MMLU_N = 5  # per subject → 30 total

ARC_N = 20

# Non-factual: MMLU ethical/philosophical subtasks (same library, Parquet-native)
MMLU_ETHICS_SUBJECTS = [
    ("cais/mmlu", "moral_scenarios",  15),
    ("cais/mmlu", "moral_disputes",   10),
    ("cais/mmlu", "business_ethics",  10),
    ("cais/mmlu", "philosophy",       10),
    ("cais/mmlu", "jurisprudence",     5),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(dataset_id: str, config: str | None, split: str):
    """Load a HuggingFace dataset, with a clear error if `datasets` not installed."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "The `datasets` package is required. Install with:\n"
            "  pip install datasets"
        )
    if config:
        return load_dataset(dataset_id, config, split=split)
    return load_dataset(dataset_id, split=split)


def _clean(text: str) -> str:
    """Strip excess whitespace and normalise."""
    return re.sub(r"\s+", " ", text).strip()


def _make_id(prefix: str, n: int) -> str:
    return f"{prefix}_{n:03d}"


# ---------------------------------------------------------------------------
# Factual sourcing — MMLU
# ---------------------------------------------------------------------------

def _mmlu_to_openended(row: dict) -> dict | None:
    """Convert one MMLU row to an open-ended question dict, or None if skipped."""
    q = _clean(row["question"])
    choices = row.get("choices", [])
    answer_idx = row.get("answer", 0)

    if not choices or answer_idx >= len(choices):
        return None

    correct = _clean(choices[answer_idx])
    if len(correct) < 20:
        return None  # trivially short answer

    # Rephrase: "Which of the following..." → "Explain..." / "What is..." etc.
    stem = re.sub(r"^which of the following\s+", "Explain ", q, flags=re.IGNORECASE)
    stem = re.sub(r"^which\s+", "Explain which ", stem, flags=re.IGNORECASE)
    if stem == q:
        # Generic fallback: just ask for an explanation
        stem = q.rstrip("?") + "? Explain your reasoning."

    # expected_points: correct answer claim + hint from wrong options as "misconceptions"
    distractors = [_clean(c) for i, c in enumerate(choices) if i != answer_idx and len(_clean(c)) > 15]
    expected_points = [correct] + distractors[:2]

    return {
        "stream": "factual",
        "question_text": stem,
        "source_dataset": "mmlu",
        "source_subject": None,  # filled by caller
        "difficulty": "medium",
        "expected_points": expected_points,
    }


def crawl_mmlu(n_per_subject: int, rng: random.Random) -> list[dict]:
    rows = []
    for dataset_id, subject in MMLU_SUBJECTS:
        ds = _load(dataset_id, subject, "test")
        candidates = [r for r in ds if r.get("choices") and len(r.get("choices", [])) == 4]
        sample = rng.sample(candidates, min(n_per_subject * 3, len(candidates)))
        accepted = []
        for row in sample:
            converted = _mmlu_to_openended(row)
            if converted:
                converted["source_subject"] = subject
                accepted.append(converted)
                if len(accepted) == n_per_subject:
                    break
        rows.extend(accepted)
        print(f"  MMLU {subject}: {len(accepted)} questions")
    return rows


# ---------------------------------------------------------------------------
# Factual sourcing — ARC-Challenge
# ---------------------------------------------------------------------------

def _arc_to_openended(row: dict) -> dict | None:
    q = _clean(row["question"])
    choices = row.get("choices", {})
    labels = choices.get("label", [])
    texts  = choices.get("text",  [])
    answer_key = row.get("answerKey", "")

    if not labels or answer_key not in labels:
        return None

    idx = labels.index(answer_key)
    correct = _clean(texts[idx]) if idx < len(texts) else ""
    if len(correct) < 15:
        return None

    stem = re.sub(r"^which of the following\s+", "Explain ", q, flags=re.IGNORECASE)
    if stem == q:
        stem = q.rstrip("?") + "? Provide a detailed explanation."

    distractors = [_clean(texts[i]) for i, l in enumerate(labels)
                   if l != answer_key and i < len(texts) and len(_clean(texts[i])) > 10]
    expected_points = [correct] + distractors[:2]

    return {
        "stream": "factual",
        "question_text": stem,
        "source_dataset": "arc_challenge",
        "source_subject": "science_reasoning",
        "difficulty": "medium",
        "expected_points": expected_points,
    }


def crawl_arc(n: int, rng: random.Random) -> list[dict]:
    ds = _load("allenai/ai2_arc", "ARC-Challenge", "test")
    sample = rng.sample(list(ds), min(n * 3, len(ds)))
    accepted = []
    for row in sample:
        converted = _arc_to_openended(row)
        if converted:
            accepted.append(converted)
            if len(accepted) == n:
                break
    print(f"  ARC-Challenge: {len(accepted)} questions")
    return accepted


# ---------------------------------------------------------------------------
# Non-factual sourcing — MMLU ethical/philosophical subtasks
# ---------------------------------------------------------------------------

def _mmlu_ethics_to_openended(row: dict, subject: str) -> dict | None:
    q = _clean(row["question"])
    choices = row.get("choices", [])
    answer_idx = row.get("answer", 0)

    if not choices or answer_idx >= len(choices):
        return None

    correct = _clean(choices[answer_idx])
    if len(correct) < 10:
        return None

    # Rephrase MC stem into an open-ended reasoning question
    stem = re.sub(r"^which of the following\s+", "Explain ", q, flags=re.IGNORECASE)
    stem = re.sub(r"^which\s+", "Explain which ", stem, flags=re.IGNORECASE)
    if stem == q:
        stem = q.rstrip("?") + "? Discuss the relevant ethical considerations and reasoning."

    distractors = [_clean(c) for i, c in enumerate(choices)
                   if i != answer_idx and len(_clean(c)) > 10]
    key_tensions = [correct] + distractors[:2]

    return {
        "stream": "non_factual",
        "question_text": stem,
        "source_dataset": "mmlu",
        "source_subject": subject,
        "difficulty": "medium",
        "key_tensions": key_tensions,
    }


def crawl_mmlu_ethics(rng: random.Random) -> list[dict]:
    rows = []
    for dataset_id, subject, n in MMLU_ETHICS_SUBJECTS:
        ds = _load(dataset_id, subject, "test")
        candidates = [r for r in ds if r.get("choices") and len(r.get("choices", [])) == 4]
        sample = rng.sample(candidates, min(n * 3, len(candidates)))
        accepted = []
        for row in sample:
            converted = _mmlu_ethics_to_openended(row, subject)
            if converted:
                accepted.append(converted)
                if len(accepted) == n:
                    break
        rows.extend(accepted)
        print(f"  MMLU {subject}: {len(accepted)} questions")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_prompts(limit: int | None, seed: int) -> list[dict]:
    rng = random.Random(seed)

    print("Crawling factual sources...")
    factual = crawl_mmlu(MMLU_N, rng) + crawl_arc(ARC_N, rng)
    rng.shuffle(factual)
    factual = factual[:50]

    print("Crawling non-factual sources...")
    non_factual = crawl_mmlu_ethics(rng)
    rng.shuffle(non_factual)
    non_factual = non_factual[:50]

    combined = factual + non_factual
    if limit:
        # Take proportionally from each stream
        half = limit // 2
        combined = factual[:half] + non_factual[:half]

    # Assign stable question IDs
    result = []
    fq_i = nf_i = 1
    for q in combined:
        if q["stream"] == "factual":
            q["question_id"] = _make_id("fq", fq_i)
            fq_i += 1
        else:
            q["question_id"] = _make_id("nf", nf_i)
            nf_i += 1
        result.append(q)

    print(f"\nTotal: {len(result)} questions "
          f"({sum(1 for q in result if q['stream']=='factual')} factual, "
          f"{sum(1 for q in result if q['stream']=='non_factual')} non-factual)")
    return result


def main():
    parser = argparse.ArgumentParser(description="Crawl benchmarks for StyleJudge v3")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap total questions (e.g. 10 for a quick test)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompts = build_prompts(args.limit, args.seed)

    out_path = Path("data/raw/base_prompts.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
