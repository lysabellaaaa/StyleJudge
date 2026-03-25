"""
AnalysisAgent v3: Full H4–H9 analysis across multi-judge, multi-mode results.

Reads:
  - results/evaluations/{judge_id}/{mode}_scores.json   (rubric)
  - results/pairwise/{judge_id}_{mode}.json             (pairwise)
  - results/mitigation/{condition}_scores.json          (mitigation)
  - results/mechanistic/cot_analysis.json               (H7)

Writes:
  - results/analysis/metrics_summary.json
  - results/analysis/findings_v3.md
"""
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.utils import logger as log
from src.utils.state import ExperimentState


def _load_json(path: str):
    return json.loads(Path(path).read_bytes().decode("utf-8", errors="replace"))


def _save_json(path: str, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mean(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def _cohens_d(a: list, b: list) -> float:
    if not a or not b:
        return float("nan")
    pooled = math.sqrt((_std(a) ** 2 + _std(b) ** 2) / 2)
    if pooled == 0:
        return float("nan")
    return (_mean(a) - _mean(b)) / pooled


def _bootstrap_ci(a: list, b: list, n: int = 1000) -> tuple[float, float]:
    import random
    diffs = []
    for _ in range(n):
        sa = [random.choice(a) for _ in range(len(a))]
        sb = [random.choice(b) for _ in range(len(b))]
        diffs.append(_mean(sa) - _mean(sb))
    diffs.sort()
    lo = diffs[int(0.025 * n)]
    hi = diffs[int(0.975 * n)]
    return round(lo, 3), round(hi, 3)


def _sbs_row(simple_scores: list, abstract_scores: list) -> dict:
    sbs = round(_mean(abstract_scores) - _mean(simple_scores), 3) if simple_scores and abstract_scores else None
    d   = round(_cohens_d(abstract_scores, simple_scores), 3)     if simple_scores and abstract_scores else None
    ci  = _bootstrap_ci(abstract_scores, simple_scores)           if simple_scores and abstract_scores else (None, None)
    return {
        "n_simple":      len(simple_scores),
        "n_abstract":    len(abstract_scores),
        "mean_simple":   round(_mean(simple_scores), 3),
        "mean_abstract": round(_mean(abstract_scores), 3),
        "sbs":           sbs,
        "cohens_d":      d,
        "ci_lower":      ci[0],
        "ci_upper":      ci[1],
    }


def _fmt_row(label: str, row: dict) -> str:
    sbs = f"{row['sbs']:+.2f}" if row["sbs"] is not None else "N/A"
    d   = f"{row['cohens_d']:.2f}" if row["cohens_d"] is not None and not math.isnan(row["cohens_d"]) else "NaN"
    ci  = f"[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]" if row["ci_lower"] is not None else "N/A"
    return (f"| {label:<38} | {row['mean_simple']:.2f} | {row['mean_abstract']:.2f} "
            f"| {sbs} | {d} | {ci} |")


def _load_rubric_scores(results_dir: Path, judge_id: str, mode: str) -> list[dict]:
    path = results_dir / judge_id / f"{mode}_scores.json"
    return _load_json(str(path)) if path.exists() else []


def _load_pairwise(pairwise_dir: Path, judge_id: str, mode: str) -> list[dict]:
    path = pairwise_dir / f"{judge_id}_{mode}.json"
    return _load_json(str(path)) if path.exists() else []


# ---------------------------------------------------------------------------
# Build per-stream SBS tables
# ---------------------------------------------------------------------------

def _index_scores(scores: list) -> dict:
    """Return {(stream, variant_type): [aggregate_scores]}."""
    idx: dict[tuple, list] = defaultdict(list)
    for s in scores:
        vt = s.get("variant_type", "")
        stream = s.get("stream", "")
        idx[(stream, vt)].append(s["aggregate_score"])
    return idx


def _sbs_table_for_judge(scores: list) -> dict:
    """Compute SBS by stream for a single judge's rubric scores."""
    simple_types   = {"V-simple",         "V-natural-simple"}
    abstract_types = {"V-abstract",       "V-natural-abstract"}

    by_stream_simple: dict[str, list] = defaultdict(list)
    by_stream_abstract: dict[str, list] = defaultdict(list)

    for s in scores:
        vt     = s.get("variant_type", "")
        stream = s.get("stream", "")
        score  = s.get("aggregate_score")
        if score is None:
            continue
        if vt in simple_types:
            by_stream_simple[stream].append(float(score))
        elif vt in abstract_types:
            by_stream_abstract[stream].append(float(score))

    result = {}
    for stream in set(by_stream_simple) | set(by_stream_abstract):
        result[stream] = _sbs_row(by_stream_simple[stream], by_stream_abstract[stream])
    # Overall
    all_simple   = [s for vals in by_stream_simple.values()   for s in vals]
    all_abstract = [s for vals in by_stream_abstract.values() for s in vals]
    result["overall"] = _sbs_row(all_simple, all_abstract)
    return result


# ---------------------------------------------------------------------------
# Pairwise stats
# ---------------------------------------------------------------------------

def _pairwise_stats(pairwise: list) -> dict:
    by_stream: dict[str, dict] = defaultdict(lambda: {"V-simple": 0, "V-abstract": 0,
                                                       "V-natural-simple": 0, "V-natural-abstract": 0, "total": 0})
    for p in pairwise:
        stream = p.get("stream", "all")
        pref   = p.get("preferred_variant", "")
        by_stream[stream][pref] = by_stream[stream].get(pref, 0) + 1
        by_stream[stream]["total"] += 1

    overall = {"V-simple": 0, "V-abstract": 0,
               "V-natural-simple": 0, "V-natural-abstract": 0, "total": 0}
    for p in pairwise:
        pref = p.get("preferred_variant", "")
        overall[pref] = overall.get(pref, 0) + 1
        overall["total"] += 1

    def _abstract_pref_rate(d):
        abstract_count = d.get("V-abstract", 0) + d.get("V-natural-abstract", 0)
        total = d.get("total", 0)
        return round(abstract_count / total, 3) if total else None

    return {
        "by_stream":            dict(by_stream),
        "overall":              overall,
        "abstract_pref_rate":   _abstract_pref_rate(overall),
    }


# ---------------------------------------------------------------------------
# Adversarial EDR/FPR
# ---------------------------------------------------------------------------

def _adversarial_stats(scores: list, threshold: int) -> dict:
    """Error Detection Rate and False Penalty Rate from adversarial scores."""
    by_type: dict[str, dict] = defaultdict(lambda: {"detected": 0, "total": 0})
    for s in scores:
        vt    = s.get("variant_type", "")
        score = s.get("aggregate_score")
        if score is None:
            continue
        by_type[vt]["total"] += 1
        if float(score) <= threshold:
            by_type[vt]["detected"] += 1

    result = {}
    for vt, d in by_type.items():
        total = d["total"]
        detected = d["detected"]
        result[vt] = {
            "n": total,
            "detected": detected,
            "edr": round(detected / total, 3) if total else None,
        }
    return result


# ---------------------------------------------------------------------------
# Mitigation delta SBS
# ---------------------------------------------------------------------------

def _mitigation_stats(baseline_scores: list, mitigation_scores: list) -> dict:
    """Compute SBS for baseline and mitigation; return delta."""
    def _sbs(scores):
        simple_types   = {"V-simple"}
        abstract_types = {"V-abstract"}
        s_vals = [float(s["aggregate_score"]) for s in scores if s.get("variant_type") in simple_types]
        a_vals = [float(s["aggregate_score"]) for s in scores if s.get("variant_type") in abstract_types]
        return round(_mean(a_vals) - _mean(s_vals), 3) if s_vals and a_vals else None

    baseline_sbs   = _sbs(baseline_scores)
    mitigation_sbs = _sbs(mitigation_scores)
    delta = round(mitigation_sbs - baseline_sbs, 3) if baseline_sbs is not None and mitigation_sbs is not None else None
    return {
        "baseline_sbs":   baseline_sbs,
        "mitigation_sbs": mitigation_sbs,
        "delta_sbs":      delta,
        "interpretation": (
            "Mitigation reduced bias" if delta is not None and delta < 0 else
            "Mitigation increased bias" if delta is not None and delta > 0 else
            "No change"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(cfg: dict, state: ExperimentState) -> None:
    results_dir    = Path(cfg["paths"]["results_evaluations"])
    pairwise_dir   = Path(cfg["paths"]["results_pairwise"])
    mitigation_dir = Path(cfg["paths"]["results_mitigation"])
    mechanistic_dir = Path(cfg["paths"]["results_mechanistic"])
    rubric_path    = Path(cfg["paths"]["rubric_nonfactual"])
    questions_path = cfg["paths"]["base_prompts"]

    rubric    = _load_json(str(rubric_path)) if rubric_path.exists() else {}
    questions = _load_json(questions_path)
    rubric_criteria = {c["id"]: c["name"] for c in rubric.get("criteria", [])}

    judges         = list(cfg["models"]["judges"].keys())
    pairwise_judges = cfg["models"].get("pairwise_judges", ["claude", "gpt4o"])
    adv_threshold  = cfg["evaluation"].get("adversarial_threshold", 2)

    # -----------------------------------------------------------------------
    # Collect all rubric scores per judge × mode
    # -----------------------------------------------------------------------
    all_rubric: dict[str, dict[str, list]] = {}
    for judge_id in judges:
        all_rubric[judge_id] = {
            "artificial": _load_rubric_scores(results_dir, judge_id, "artificial"),
            "natural":    _load_rubric_scores(results_dir, judge_id, "natural"),
            "adversarial":_load_rubric_scores(results_dir, judge_id, "adversarial"),
        }

    # -----------------------------------------------------------------------
    # Collect pairwise
    # -----------------------------------------------------------------------
    all_pairwise: dict[str, dict[str, list]] = {}
    for judge_id in pairwise_judges:
        all_pairwise[judge_id] = {
            "artificial": _load_pairwise(pairwise_dir, judge_id, "artificial"),
            "natural":    _load_pairwise(pairwise_dir, judge_id, "natural"),
        }

    # -----------------------------------------------------------------------
    # Collect mitigation (Claude only)
    # -----------------------------------------------------------------------
    baseline_artificial = all_rubric.get("claude", {}).get("artificial", [])
    mitigation_results = {}
    for condition in ["format_agnostic", "style_norm", "fixed_rubric"]:
        path = mitigation_dir / f"{condition}_scores.json"
        mit_scores = _load_json(str(path)) if path.exists() else []
        mitigation_results[condition] = _mitigation_stats(baseline_artificial, mit_scores)

    # -----------------------------------------------------------------------
    # Collect CoT analysis (H7)
    # -----------------------------------------------------------------------
    cot_path = mechanistic_dir / "cot_analysis.json"
    cot_data = _load_json(str(cot_path)) if cot_path.exists() else {}

    # -----------------------------------------------------------------------
    # Compute all SBS tables
    # -----------------------------------------------------------------------
    sbs_rubric: dict[str, dict[str, dict]] = {}   # [judge_id][mode] → sbs_table
    for judge_id in judges:
        sbs_rubric[judge_id] = {}
        for mode in ["artificial", "natural"]:
            sbs_rubric[judge_id][mode] = _sbs_table_for_judge(all_rubric[judge_id][mode])

    pairwise_stats: dict[str, dict[str, dict]] = {}
    for judge_id in pairwise_judges:
        pairwise_stats[judge_id] = {}
        for mode in ["artificial", "natural"]:
            pairwise_stats[judge_id][mode] = _pairwise_stats(all_pairwise[judge_id][mode])

    adv_stats: dict[str, dict] = {}
    for judge_id in judges:
        adv_scores = all_rubric[judge_id]["adversarial"]
        if adv_scores:
            adv_stats[judge_id] = _adversarial_stats(adv_scores, adv_threshold)

    # -----------------------------------------------------------------------
    # Save metrics JSON
    # -----------------------------------------------------------------------
    metrics = {
        "generated_at":    _now(),
        "n_questions":     len(questions),
        "sbs_rubric":      sbs_rubric,
        "pairwise_stats":  pairwise_stats,
        "adversarial":     adv_stats,
        "mitigation":      mitigation_results,
        "cot_analysis":    cot_data.get("mediation", {}),
    }
    _save_json(str(Path(cfg["paths"]["results_analysis"]) / "metrics_summary.json"), metrics)

    # -----------------------------------------------------------------------
    # Write findings_v3.md
    # -----------------------------------------------------------------------
    out_path = cfg["paths"]["findings_report"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    lines = []
    n_factual    = sum(1 for q in questions if q.get("stream") == "factual")
    n_nonfactual = sum(1 for q in questions if q.get("stream") == "non_factual")

    lines.append("# StyleJudge v3 — Full Study Findings\n")
    lines.append(f"**Date:** {_now()[:10]}  ")
    lines.append(f"**Judges:** {', '.join(judges)}  ")
    lines.append(f"**Questions:** {len(questions)} "
                 f"(factual: {n_factual}, non-factual: {n_nonfactual})  ")
    lines.append(f"**Modes:** Artificial (rewrites) + Natural (V3 vs R1)  \n")

    # --- H4: Paradigm Flip ---
    lines.append("---\n")
    lines.append("## H4 — The Evaluation Paradigm Flip\n")
    lines.append("*Prediction: SBS_rubric < 0 AND pairwise preference for V-abstract > 50% simultaneously.*\n")
    lines.append("| Judge | Mode | V-simple mean | V-abstract mean | SBS (rubric) | Pairwise P(abstract) | Flip? |")
    lines.append("|---|---|---|---|---|---|---|")

    for judge_id in judges:
        for mode in ["artificial", "natural"]:
            sbs_row = sbs_rubric.get(judge_id, {}).get(mode, {}).get("overall", {})
            sbs_val = sbs_row.get("sbs")
            p_stats = pairwise_stats.get(judge_id, {}).get(mode, {})
            p_abstract = p_stats.get("abstract_pref_rate")
            if sbs_val is None and p_abstract is None:
                continue
            sbs_str   = f"{sbs_val:+.3f}" if sbs_val is not None else "—"
            p_str     = f"{p_abstract:.1%}" if p_abstract is not None else "—"
            flip      = "✓ FLIP" if (sbs_val is not None and sbs_val < 0 and
                                     p_abstract is not None and p_abstract > 0.5) else "—"
            vs_mean   = f"{sbs_row.get('mean_simple', 0):.2f}" if sbs_row else "—"
            va_mean   = f"{sbs_row.get('mean_abstract', 0):.2f}" if sbs_row else "—"
            lines.append(f"| {judge_id} | {mode} | {vs_mean} | {va_mean} | {sbs_str} | {p_str} | {flip} |")

    lines.append("")

    # --- H5: Domain-Conditional Directionality ---
    lines.append("---\n")
    lines.append("## H5 — Domain-Conditional Directionality\n")
    lines.append("*Prediction: Factual SBS < 0 (structure audited strictly); Non-factual SBS ≈ 0 (halo effect)*\n")
    lines.append("| Judge | Mode | Stream | V-simple | V-abstract | SBS | Cohen's d | 95% CI |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for judge_id in judges:
        for mode in ["artificial", "natural"]:
            sbs_table = sbs_rubric.get(judge_id, {}).get(mode, {})
            for stream in ["factual", "non_factual"]:
                row = sbs_table.get(stream, {})
                if not row or row.get("n_simple", 0) == 0:
                    continue
                cd = row["cohens_d"]
                cd_str = "NaN" if cd is None or (isinstance(cd, float) and math.isnan(cd)) else f"{cd:.2f}"
                sbs = row.get("sbs")
                ci_lo = row.get("ci_lower")
                ci_hi = row.get("ci_upper")
                sbs_str = f"{sbs:+.3f}" if sbs is not None else "—"
                ci_str = f"[{ci_lo:.2f}, {ci_hi:.2f}]" if ci_lo is not None and ci_hi is not None else "—"
                lines.append(
                    f"| {judge_id} | {mode} | {stream} "
                    f"| {row['mean_simple']:.2f} | {row['mean_abstract']:.2f} "
                    f"| {sbs_str} | {cd_str} "
                    f"| {ci_str} |"
                )

    lines.append("")

    # --- H6: Natural vs Artificial ---
    lines.append("---\n")
    lines.append("## H6 — Natural vs Artificial Mode Comparison\n")
    lines.append("*Prediction: SBS_natural ≈ SBS_artificial (format alone explains gap). "
                 "If SBS_natural >> SBS_artificial, quality confound exists.*\n")
    lines.append("| Judge | SBS Artificial | SBS Natural | Delta (N−A) | Interpretation |")
    lines.append("|---|---|---|---|---|")

    for judge_id in judges:
        sbs_art = sbs_rubric.get(judge_id, {}).get("artificial", {}).get("overall", {}).get("sbs")
        sbs_nat = sbs_rubric.get(judge_id, {}).get("natural",    {}).get("overall", {}).get("sbs")
        if sbs_art is None and sbs_nat is None:
            continue
        art_str = f"{sbs_art:+.3f}" if sbs_art is not None else "—"
        nat_str = f"{sbs_nat:+.3f}" if sbs_nat is not None else "—"
        if sbs_art is not None and sbs_nat is not None:
            delta = round(sbs_nat - sbs_art, 3)
            delta_str = f"{delta:+.3f}"
            interp = ("Quality confound likely" if abs(delta) > 0.3 else "Converging — format isolates well")
        else:
            delta_str = "—"
            interp = "Insufficient data"
        lines.append(f"| {judge_id} | {art_str} | {nat_str} | {delta_str} | {interp} |")

    lines.append("")

    # --- H7: CoT Echo Length ---
    lines.append("---\n")
    lines.append("## H7 — CoT Echo Length (Mechanistic)\n")
    lines.append("*Prediction: V-abstract candidates produce longer judge CoT, mediating the score penalty.*\n")

    cot_by_type = cot_data.get("by_variant_type", {})
    if cot_by_type:
        lines.append("| Variant Type | Mean CoT Words | Rubric Vocab Density | F-K Grade | Mean Score |")
        lines.append("|---|---|---|---|---|")
        for vt_group, stats in cot_by_type.items():
            lines.append(
                f"| {vt_group} | {stats.get('mean_cot_words', '—')} "
                f"| {stats.get('mean_rubric_vocab_density', '—')} "
                f"| {stats.get('mean_fk_grade', '—')} "
                f"| {stats.get('mean_score', '—')} |"
            )

    med = cot_data.get("mediation", {})
    if med:
        lines.append("")
        lines.append("**Mediation analysis (Baron-Kenny OLS):**")
        lines.append(f"- Total effect (c): {med.get('total_effect_c', '—')}")
        lines.append(f"- Format → CoT length (a path): {med.get('a_path_format_to_cot', '—')} words")
        lines.append(f"- CoT length → Score (b path): {med.get('b_path_cot_to_score', '—')}")
        lines.append(f"- Direct effect (c'): {med.get('direct_effect_c_prime', '—')}")
        lines.append(f"- Indirect effect (a×b): {med.get('indirect_effect_ab', '—')}")
        pm = med.get('proportion_mediated')
        pm_str = f"{pm:.1%}" if pm is not None else "—"
        lines.append(f"- **Proportion mediated: {pm_str}**")

    lines.append("")

    # --- H8: Credibility Deference ---
    lines.append("---\n")
    lines.append("## H8 — Credibility Deference (Adversarial)\n")
    lines.append(f"*Prediction: EDR(V-abstract) < EDR(V-simple) — errors in structured responses "
                 f"are detected at a lower rate. Threshold for detection: score ≤ {adv_threshold}.*\n")

    if adv_stats:
        lines.append("| Judge | Variant Type | n | Errors Detected | EDR |")
        lines.append("|---|---|---|---|---|")
        for judge_id, stats in adv_stats.items():
            for vt, d in sorted(stats.items()):
                edr_str = f"{d['edr']:.1%}" if d["edr"] is not None else "—"
                lines.append(f"| {judge_id} | {vt} | {d['n']} | {d['detected']} | {edr_str} |")
    else:
        lines.append("*Adversarial evaluation results not yet available.*")

    lines.append("")

    # --- H9: Cross-Family Replication ---
    lines.append("---\n")
    lines.append("## H9 — Cross-Family Paradigm Flip Replication\n")
    lines.append("*Prediction: H4 (SBS_rubric < 0 AND pairwise preference for abstract > 50%) holds "
                 "across all 3 judge families.*\n")
    lines.append("| Judge | Family | SBS (Artificial) | Pairwise P(abstract) | H4 Confirmed? |")
    lines.append("|---|---|---|---|---|")

    judge_families = {"claude": "Anthropic", "gpt4o": "OpenAI", "llama70b": "Meta/Groq"}
    for judge_id in judges:
        sbs_val  = sbs_rubric.get(judge_id, {}).get("artificial", {}).get("overall", {}).get("sbs")
        p_stats  = pairwise_stats.get(judge_id, {}).get("artificial", {})
        p_abstract = p_stats.get("abstract_pref_rate")
        family   = judge_families.get(judge_id, "Unknown")
        sbs_str  = f"{sbs_val:+.3f}" if sbs_val is not None else "—"
        p_str    = f"{p_abstract:.1%}" if p_abstract is not None else "—"
        confirmed = ("Yes" if (sbs_val is not None and sbs_val < 0 and
                               p_abstract is not None and p_abstract > 0.5)
                     else "No" if (sbs_val is not None or p_abstract is not None)
                     else "Pending")
        lines.append(f"| {judge_id} | {family} | {sbs_str} | {p_str} | {confirmed} |")

    lines.append("")

    # --- Mitigation ---
    lines.append("---\n")
    lines.append("## Mitigation Results\n")
    lines.append("*Claude only, Artificial mode. Positive delta = bias increased; negative = reduced.*\n")
    lines.append("| Condition | Baseline SBS | Mitigation SBS | Δ SBS | Interpretation |")
    lines.append("|---|---|---|---|---|")

    for condition, stats in mitigation_results.items():
        b_str = f"{stats['baseline_sbs']:+.3f}"   if stats["baseline_sbs"]   is not None else "—"
        m_str = f"{stats['mitigation_sbs']:+.3f}" if stats["mitigation_sbs"] is not None else "—"
        d_str = f"{stats['delta_sbs']:+.3f}"       if stats["delta_sbs"]       is not None else "—"
        lines.append(f"| {condition} | {b_str} | {m_str} | {d_str} | {stats['interpretation']} |")

    lines.append("")

    # --- Key Findings Summary ---
    lines.append("---\n")
    lines.append("## Key Findings\n")

    finding_num = 1

    # H4 check
    for judge_id in judges:
        sbs_art = sbs_rubric.get(judge_id, {}).get("artificial", {}).get("overall", {}).get("sbs")
        p_art   = pairwise_stats.get(judge_id, {}).get("artificial", {}).get("abstract_pref_rate")
        if sbs_art is not None and p_art is not None:
            if sbs_art < 0 and p_art > 0.5:
                lines.append(
                    f"{finding_num}. **H4 CONFIRMED ({judge_id})**: Rubric SBS = {sbs_art:+.3f} (structured scored lower) "
                    f"but pairwise preference for V-abstract = {p_art:.1%} — the Evaluation Paradigm Flip holds."
                )
                finding_num += 1
            elif sbs_art >= 0:
                lines.append(
                    f"{finding_num}. **H4 NOT CONFIRMED ({judge_id})**: Rubric SBS = {sbs_art:+.3f} — "
                    f"structured responses were NOT scored lower in rubric mode."
                )
                finding_num += 1

    # H6 check
    for judge_id in judges:
        sbs_art = sbs_rubric.get(judge_id, {}).get("artificial", {}).get("overall", {}).get("sbs")
        sbs_nat = sbs_rubric.get(judge_id, {}).get("natural",    {}).get("overall", {}).get("sbs")
        if sbs_art is not None and sbs_nat is not None:
            delta = abs(round(sbs_nat - sbs_art, 3))
            if delta <= 0.15:
                lines.append(
                    f"{finding_num}. **H6 SUPPORTED ({judge_id})**: SBS_artificial = {sbs_art:+.3f}, "
                    f"SBS_natural = {sbs_nat:+.3f} — modes converge (Δ={delta:+.3f}), "
                    f"suggesting format alone (not quality) drives the bias."
                )
            else:
                lines.append(
                    f"{finding_num}. **H6 REJECTED ({judge_id})**: SBS_artificial = {sbs_art:+.3f}, "
                    f"SBS_natural = {sbs_nat:+.3f} — modes diverge (Δ={delta:+.3f}), "
                    f"suggesting a quality confound in the Natural mode."
                )
            finding_num += 1

    # H7 summary
    med = cot_data.get("mediation", {})
    pm = med.get("proportion_mediated")
    if pm is not None:
        lines.append(
            f"{finding_num}. **H7 (CoT mediation)**: {pm:.1%} of the format→score effect "
            f"is mediated by judge CoT length. "
            f"Indirect effect: {med.get('indirect_effect_ab', '?')} (a={med.get('a_path_format_to_cot', '?')}, "
            f"b={med.get('b_path_cot_to_score', '?')})."
        )
        finding_num += 1

    # Implications
    lines.append("\n---\n")
    lines.append("## Implications\n")
    lines.append(
        "- **RLHF training incoherence**: If rubric-based reward models assign lower scores to structured "
        "responses while pairwise preference models prefer them, the two training signals contradict each other."
    )
    lines.append(
        "- **Evaluation paradigm choice is not neutral**: The direction of format bias depends on whether "
        "evaluators use rubric scoring or pairwise comparison — a methodological confound in LLM evaluation benchmarks."
    )
    lines.append(
        "- **CoT echo as mechanism**: When judges write longer, more structured CoTs in response to structured "
        "candidates, they self-impose stricter rubric auditing — format of the input primes the evaluation process."
    )
    lines.append(
        "- **Credibility deference risk**: If structured responses receive lower error detection rates, "
        "adversarially crafted high-structure responses may evade rubric scrutiny."
    )

    lines.append("\n---\n")
    lines.append("## Limitations\n")
    lines.append(f"- n={len(questions)} questions from HuggingFace benchmarks — confidence intervals wide at this scale.")
    lines.append("- Mediation analysis (H7) is OLS-based; causal interpretation requires longitudinal or experimental design.")
    lines.append("- Adversarial injection quality not independently human-verified.")
    lines.append("- Groq Llama 3.3 70B excluded from pairwise evaluation due to TPM limit — H9 is partial.")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    state.mark_phase_complete("analysis")
    log.info("AnalysisAgent", f"v3 findings report written: {out_path}")
    log.info("AnalysisAgent", "Metrics saved to results/analysis/metrics_summary.json")
