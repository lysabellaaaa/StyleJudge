"""
AnalysisAgent: computes all metrics and generates figures.
Loads all results by convention (no hardcoded judge names).
Reports Cohen's d + 95% CI. No bare p-values.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from src.metrics.effect_decomposition import run_full_decomposition
from src.metrics.error_detection import compute_edr, compute_fpr
from src.metrics.reasoning_depth import compute_reasoning_depth
from src.metrics.style_bias_score import compute_sbs_matrix
from src.utils import logger as log
from src.utils.state import ExperimentState


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_json(path: str, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_all_scores(results_dir: str) -> list[dict]:
    """Convention-based loader: reads raw_scores.json from each judge subdir."""
    scores = []
    for judge_dir in Path(results_dir).iterdir():
        if not judge_dir.is_dir():
            continue
        scores_file = judge_dir / "raw_scores.json"
        if scores_file.exists():
            batch = _load_json(str(scores_file))
            scores.extend(batch)
    return scores


def load_cot_traces(results_dir: str) -> list[dict]:
    """Load all CoT traces across all judges. Returns flat list with judge_model field."""
    traces = []
    for judge_dir in Path(results_dir).iterdir():
        if not judge_dir.is_dir():
            continue
        scores_file = judge_dir / "raw_scores.json"
        if scores_file.exists():
            for rec in _load_json(str(scores_file)):
                if "cot_trace" in rec:
                    traces.append(rec)
    return traces


def compute_reasoning_depths(traces: list[dict]) -> list[dict]:
    enriched = []
    for t in traces:
        cot = t.get("cot_trace", "")
        rd = compute_reasoning_depth(cot)
        enriched.append({**t, **rd})
    return enriched


def run_anova(scores: list[dict], judge_model: str, domain: str | None = None) -> dict:
    """One-way ANOVA across L1/L2/L3/L4 formality levels."""
    filtered = [s for s in scores if s["judge_model"] == judge_model
                and not s.get("is_adversarial", False)]
    if domain:
        filtered = [s for s in filtered if s.get("domain") == domain]
    groups = {}
    for level in ["L1", "L2", "L3", "L4"]:
        groups[level] = [s["score"] for s in filtered if s["formality_level"] == level]
    non_empty = [g for g in groups.values() if len(g) >= 2]
    if len(non_empty) < 2:
        return {"f_statistic": None, "p_value": None, "significant": False}
    f, p = stats.f_oneway(*non_empty)
    return {
        "judge_model": judge_model,
        "domain": domain or "all",
        "f_statistic": round(float(f), 4),
        "p_value": round(float(p), 4),
        "significant": float(p) < 0.05,
        "group_sizes": {k: len(v) for k, v in groups.items()},
    }


def run_pearson(enriched_traces: list[dict], judge_model: str) -> dict:
    """Pearson correlation: Structure Index vs. score."""
    filtered = [t for t in enriched_traces if t.get("judge_model") == judge_model]
    x = [t["structure_index"] for t in filtered]
    y = [t["score"] for t in filtered]
    if len(x) < 3:
        return {"correlation": None, "p_value": None, "n": len(x)}
    r, p = stats.pearsonr(x, y)
    return {
        "judge_model": judge_model,
        "correlation": round(float(r), 4),
        "p_value": round(float(p), 4),
        "n": len(x),
        "interpretation": "Structure Index negatively correlated with score" if r < -0.1 else
                          "Structure Index positively correlated with score" if r > 0.1 else "No clear correlation",
    }


def generate_figures(
    scores: list[dict],
    enriched_traces: list[dict],
    sbs_matrix: list[dict],
    figures_dir: str,
) -> list[str]:
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    saved = []

    # Figure 1: SBS by domain (bar chart)
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        judges = list({r["judge_model"] for r in sbs_matrix})
        domains = list({r["domain"] for r in sbs_matrix if r["domain"] != "all"})
        x = np.arange(len(domains))
        width = 0.25
        for i, judge in enumerate(judges):
            sbs_vals = [
                next((r["sbs"] for r in sbs_matrix
                      if r["judge_model"] == judge and r["domain"] == d), 0)
                for d in domains
            ]
            ci_errs = [
                abs(next((r.get("ci_upper", r["sbs"]) - r["sbs"] for r in sbs_matrix
                          if r["judge_model"] == judge and r["domain"] == d), 0))
                for d in domains
            ]
            ax.bar(x + i * width, sbs_vals, width, label=judge, yerr=ci_errs, capsize=4)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Domain")
        ax.set_ylabel("StyleBias Score (SBS = mean_L4 - mean_L1)")
        ax.set_title("StyleBias Score by Domain and Judge\n(Negative = structured responses scored lower)")
        ax.set_xticks(x + width)
        ax.set_xticklabels(domains, rotation=15)
        ax.legend()
        fig.tight_layout()
        path = f"{figures_dir}/sbs_by_domain.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
    except Exception as e:
        log.error("AnalysisAgent", f"Figure 1 failed: {e}")

    # Figure 2: Structure Index vs. score scatter
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        for judge in set(t.get("judge_model") for t in enriched_traces):
            pts = [t for t in enriched_traces if t.get("judge_model") == judge]
            x_vals = [t["structure_index"] for t in pts]
            y_vals = [t["score"] for t in pts]
            ax.scatter(x_vals, y_vals, alpha=0.5, label=judge, s=30)
            if len(x_vals) > 2:
                m, b = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(min(x_vals), max(x_vals), 100)
                ax.plot(x_line, m * x_line + b, linewidth=1.5)
        ax.set_xlabel("Structure Index (regex count in judge CoT)")
        ax.set_ylabel("Score (1-5)")
        ax.set_title("Structure Index vs. Score\n(Mechanistic signature of reasoning-style coupling)")
        ax.legend()
        fig.tight_layout()
        path = f"{figures_dir}/structure_index_vs_score.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
    except Exception as e:
        log.error("AnalysisAgent", f"Figure 2 failed: {e}")

    # Figure 3: Reasoning depth by formality level
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        levels = ["L1", "L2", "L3", "L4"]
        word_counts = [[t["word_count"] for t in enriched_traces if t.get("formality_level") == lvl]
                       for lvl in levels]
        ax.boxplot([wc for wc in word_counts if wc], labels=[l for l, wc in zip(levels, word_counts) if wc])
        ax.set_xlabel("Candidate Formality Level")
        ax.set_ylabel("Judge CoT Word Count")
        ax.set_title("Judge CoT Length by Candidate Formality Level\n(H1: L4 → longer, more deliberate reasoning)")
        fig.tight_layout()
        path = f"{figures_dir}/cot_depth_by_formality.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
    except Exception as e:
        log.error("AnalysisAgent", f"Figure 3 failed: {e}")

    return saved


def run_analysis(cfg: dict, state: ExperimentState) -> dict:
    log.info("AnalysisAgent", "Loading all evaluation results...")
    scores = load_all_scores(cfg["paths"]["results_evaluations"])
    traces = load_cot_traces(cfg["paths"]["results_evaluations"])
    enriched = compute_reasoning_depths(traces)

    judges = list({s["judge_model"] for s in scores})
    domains = list({s.get("domain") for s in scores if s.get("domain")})

    # StyleBias Score matrix
    sbs_matrix = compute_sbs_matrix(scores, judges, domains,
                                     n_resamples=cfg["analysis"]["bootstrap_resamples"])

    # ANOVA
    anova_results = []
    for judge in judges:
        anova_results.append(run_anova(scores, judge))
        for domain in domains:
            anova_results.append(run_anova(scores, judge, domain))

    # Pearson correlation (Structure Index vs. score)
    pearson_results = [run_pearson(enriched, judge) for judge in judges]

    # EDR and FPR
    edr = compute_edr(scores, cfg["analysis"]["penalty_threshold"])
    fpr = compute_fpr(scores, cfg["analysis"]["penalty_threshold"])

    # Effect decomposition (halo vs. scrutiny)
    decomposition = run_full_decomposition(scores, judges, domains)

    metrics = {
        "sbs_matrix": sbs_matrix,
        "anova": anova_results,
        "pearson": pearson_results,
        "edr_by_formality": edr,
        "fpr_by_formality": fpr,
        "effect_decomposition": decomposition,
        "n_scores": len(scores),
        "judges": judges,
        "domains": domains,
    }

    _save_json(f"{cfg['paths']['results_analysis']}/metrics_summary.json", metrics)
    log.info("AnalysisAgent", "Metrics saved")

    figures = generate_figures(
        scores, enriched, sbs_matrix,
        f"{cfg['paths']['results_analysis']}/figures",
    )
    log.info("AnalysisAgent", f"Figures saved: {figures}")

    state.mark_phase_complete("analysis")
    return metrics
