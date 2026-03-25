"""
CoTAnalysisAgent v3: Mechanistic analysis of judge CoT traces (H7).

Tests whether judge chain-of-thought length and rubric-vocabulary density mediate
the StyleBias Score — i.e., whether the format of the candidate response causes
the judge to write a longer, more rubric-like CoT, which in turn drives score penalties.

Baron-Kenny 3-step mediation (OLS):
  Step 1: Format → Score (total effect c)
  Step 2: Format → CoT_length (a path)
  Step 3: Format + CoT_length → Score (b path, c' = direct effect)
  Indirect effect = a × b; proportion mediated = (c - c') / c

Output: results/mechanistic/cot_analysis.json
"""
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.utils import logger as log
from src.utils.state import ExperimentState


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_json(path: str, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Text metrics
# ---------------------------------------------------------------------------

RUBRIC_VOCAB = re.compile(
    r"\b(criterion|criteria|dimension|aspect|rubric|framework|completeness|"
    r"coherence|accuracy|structure|organized|organization|logical|first|"
    r"second|third|finally|however|therefore|consequently|furthermore|"
    r"analysis|evaluate|assess|consider)\b",
    re.IGNORECASE,
)


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _rubric_vocab_density(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    matches = len(RUBRIC_VOCAB.findall(text))
    return round(matches / len(words), 4)


def _flesch_kincaid_grade(text: str) -> float:
    """Approximate Flesch-Kincaid grade level without external library."""
    sentences = max(1, len(re.split(r"[.!?]+", text)))
    words = text.split()
    n_words = max(1, len(words))
    # Approximate syllables: count vowel groups
    syllables = sum(
        max(1, len(re.findall(r"[aeiouAEIOU]+", w))) for w in words
    )
    asl = n_words / sentences            # average sentence length
    asw = syllables / n_words            # average syllables per word
    return round(0.39 * asl + 11.8 * asw - 15.59, 2)


# ---------------------------------------------------------------------------
# Simple OLS regression (no scipy required — uses normal equations)
# ---------------------------------------------------------------------------

def _ols(X: list[list[float]], y: list[float]) -> list[float]:
    """Ordinary least squares. X includes a leading 1 for intercept."""
    n = len(y)
    k = len(X[0])
    # XtX and Xty
    XtX = [[sum(X[i][r] * X[i][c] for i in range(n)) for c in range(k)] for r in range(k)]
    Xty = [sum(X[i][r] * y[i] for i in range(n)) for r in range(k)]
    # Solve via Gaussian elimination
    aug = [XtX[r][:] + [Xty[r]] for r in range(k)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        if aug[col][col] == 0:
            return [float("nan")] * k
        factor = aug[col][col]
        aug[col] = [v / factor for v in aug[col]]
        for row in range(k):
            if row != col:
                mul = aug[row][col]
                aug[row] = [aug[row][j] - mul * aug[col][j] for j in range(k + 1)]
    return [aug[r][k] for r in range(k)]


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_cot_analysis(cfg: dict, state: ExperimentState) -> dict:
    phase_key = "cot_analysis"
    if state.is_phase_complete(phase_key):
        log.info("CoTAnalysisAgent", "CoT analysis already complete — skipping")
        out_path = Path(cfg["paths"]["results_mechanistic"]) / "cot_analysis.json"
        return _load_json(str(out_path)) if out_path.exists() else {}

    # Load all evaluation scores from all judges and modes
    results_dir = Path(cfg["paths"]["results_evaluations"])
    all_scores = []
    for judge_dir in results_dir.iterdir():
        if not judge_dir.is_dir():
            continue
        for score_file in judge_dir.glob("*_scores.json"):
            mode = score_file.stem.replace("_scores", "")
            if mode == "adversarial":
                continue  # Adversarial has different semantics; exclude
            try:
                records = _load_json(str(score_file))
                for r in records:
                    r["_mode"] = mode
                all_scores.extend(records)
            except Exception as e:
                log.warn("CoTAnalysisAgent", f"Could not load {score_file}: {e}")

    if not all_scores:
        log.warn("CoTAnalysisAgent", "No evaluation scores found — skipping CoT analysis")
        return {}

    log.info("CoTAnalysisAgent", f"Loaded {len(all_scores)} score records for CoT analysis")

    # Separate into abstract (1) and simple (0) by variant_type
    simple_types   = {"V-simple",         "V-natural-simple"}
    abstract_types = {"V-abstract",       "V-natural-abstract"}

    records_with_cot = []
    for r in all_scores:
        vt = r.get("variant_type", "")
        cot = r.get("cot_trace", "") or ""
        score = r.get("aggregate_score")
        if vt not in simple_types | abstract_types:
            continue
        if score is None:
            continue
        is_abstract = 1 if vt in abstract_types else 0
        records_with_cot.append({
            "variant_type":    vt,
            "is_abstract":     is_abstract,
            "score":           float(score),
            "cot_words":       _word_count(cot),
            "rubric_density":  _rubric_vocab_density(cot),
            "fk_grade":        _flesch_kincaid_grade(cot),
            "judge_id":        r.get("judge_id", "unknown"),
            "stream":          r.get("stream", ""),
            "_mode":           r.get("_mode", ""),
        })

    if not records_with_cot:
        log.warn("CoTAnalysisAgent", "No records with valid CoT traces found")
        return {}

    # -----------------------------------------------------------------------
    # Aggregate CoT metrics by variant type
    # -----------------------------------------------------------------------
    by_type: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in records_with_cot:
        vt_group = "abstract" if r["is_abstract"] else "simple"
        by_type[vt_group]["cot_words"].append(r["cot_words"])
        by_type[vt_group]["rubric_density"].append(r["rubric_density"])
        by_type[vt_group]["fk_grade"].append(r["fk_grade"])
        by_type[vt_group]["score"].append(r["score"])

    def _agg(vals):
        return {"mean": round(_mean(vals), 3), "n": len(vals)}

    summary_by_type = {
        vt_group: {
            "mean_cot_words":          round(_mean(d["cot_words"]), 1),
            "mean_rubric_vocab_density": round(_mean(d["rubric_density"]), 4),
            "mean_fk_grade":           round(_mean(d["fk_grade"]), 2),
            "mean_score":              round(_mean(d["score"]), 3),
            "n":                       len(d["score"]),
        }
        for vt_group, d in by_type.items()
    }

    # -----------------------------------------------------------------------
    # Baron-Kenny mediation: Format → CoT_length → Score
    # -----------------------------------------------------------------------
    X_format   = [r["is_abstract"] for r in records_with_cot]
    Y_score    = [r["score"]       for r in records_with_cot]
    M_cot      = [r["cot_words"]   for r in records_with_cot]
    n          = len(records_with_cot)

    # Step 1: Format → Score (total effect c)
    coefs_c = _ols([[1, X_format[i]] for i in range(n)], Y_score)
    c = coefs_c[1] if len(coefs_c) > 1 else float("nan")

    # Step 2: Format → CoT_length (a path)
    coefs_a = _ols([[1, X_format[i]] for i in range(n)], M_cot)
    a = coefs_a[1] if len(coefs_a) > 1 else float("nan")

    # Step 3: Format + CoT_length → Score (b path, c' direct effect)
    coefs_bc = _ols([[1, X_format[i], M_cot[i]] for i in range(n)], Y_score)
    c_prime = coefs_bc[1] if len(coefs_bc) > 1 else float("nan")
    b        = coefs_bc[2] if len(coefs_bc) > 2 else float("nan")

    indirect = a * b if not (math.isnan(a) or math.isnan(b)) else float("nan")
    prop_mediated = ((c - c_prime) / c) if (c != 0 and not math.isnan(c) and not math.isnan(c_prime)) else float("nan")

    mediation = {
        "total_effect_c":        round(c, 4) if not math.isnan(c) else None,
        "a_path_format_to_cot":  round(a, 4) if not math.isnan(a) else None,
        "b_path_cot_to_score":   round(b, 6) if not math.isnan(b) else None,
        "direct_effect_c_prime": round(c_prime, 4) if not math.isnan(c_prime) else None,
        "indirect_effect_ab":    round(indirect, 4) if not math.isnan(indirect) else None,
        "proportion_mediated":   round(prop_mediated, 3) if not math.isnan(prop_mediated) else None,
        "n_records":             n,
        "interpretation": (
            f"CoT length mediates {round(prop_mediated*100, 1) if not math.isnan(prop_mediated) else '?'}% "
            f"of the format→score effect. "
            f"Total effect: {round(c, 3) if not math.isnan(c) else '?'} score units per abstract variant. "
            f"Format increases CoT length by {round(a, 1) if not math.isnan(a) else '?'} words on average."
        ),
    }

    # -----------------------------------------------------------------------
    # Per-judge SBS summary (to cross-check with main analysis)
    # -----------------------------------------------------------------------
    by_judge: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in records_with_cot:
        by_judge[r["judge_id"]]["abstract" if r["is_abstract"] else "simple"].append(r["score"])

    sbs_by_judge = {}
    for jid, d in by_judge.items():
        s_scores = d.get("simple", [])
        a_scores = d.get("abstract", [])
        sbs_by_judge[jid] = {
            "mean_simple":   round(_mean(s_scores), 3),
            "mean_abstract": round(_mean(a_scores), 3),
            "sbs":           round(_mean(a_scores) - _mean(s_scores), 3) if s_scores and a_scores else None,
            "n_simple":      len(s_scores),
            "n_abstract":    len(a_scores),
        }

    result = {
        "generated_at":  _now(),
        "n_records":     len(records_with_cot),
        "by_variant_type": summary_by_type,
        "mediation":     mediation,
        "sbs_by_judge":  sbs_by_judge,
    }

    out_path = Path(cfg["paths"]["results_mechanistic"]) / "cot_analysis.json"
    _save_json(str(out_path), result)
    state.mark_phase_complete(phase_key)
    log.info("CoTAnalysisAgent",
             f"CoT analysis complete. Proportion mediated: {mediation['proportion_mediated']}")
    return result
