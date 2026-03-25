# StyleJudge v3 — Full Study Findings

**Date:** 2026-03-25  
**Judges:** claude, gpt4o, llama70b  
**Questions:** 100 (factual: 50, non-factual: 50)  
**Modes:** Artificial (rewrites) + Natural (V3 vs R1)  

---

## H4 — The Evaluation Paradigm Flip

*Prediction: SBS_rubric < 0 AND pairwise preference for V-abstract > 50% simultaneously.*

| Judge | Mode | V-simple mean | V-abstract mean | SBS (rubric) | Pairwise P(abstract) | Flip? |
|---|---|---|---|---|---|---|
| claude | artificial | 3.75 | 4.00 | +0.251 | 75.7% | — |
| claude | natural | 4.03 | 4.17 | +0.141 | 87.5% | — |
| gpt4o | artificial | 4.50 | 4.58 | +0.075 | — | — |
| llama70b | artificial | 4.86 | 4.79 | -0.063 | — | — |

---

## H5 — Domain-Conditional Directionality

*Prediction: Factual SBS < 0 (structure audited strictly); Non-factual SBS ≈ 0 (halo effect)*

| Judge | Mode | Stream | V-simple | V-abstract | SBS | Cohen's d | 95% CI |
|---|---|---|---|---|---|---|---|
| claude | artificial | factual | 4.40 | 4.60 | +0.202 | 0.37 | [0.01, 0.43] |
| claude | artificial | non_factual | 3.10 | 3.42 | +0.325 | 0.67 | [0.12, 0.52] |
| claude | natural | factual | 4.70 | 4.76 | +0.060 | 0.15 | [-0.09, 0.21] |
| claude | natural | non_factual | 3.22 | 3.46 | +0.231 | 0.44 | [-0.00, 0.46] |
| gpt4o | artificial | factual | 4.57 | 4.42 | -0.142 | -0.15 | [-0.88, 0.45] |
| gpt4o | artificial | non_factual | 4.40 | 4.80 | +0.400 | 0.46 | [-0.29, 1.22] |
| llama70b | artificial | factual | 4.84 | 4.77 | -0.069 | -0.16 | [-0.30, 0.14] |
| llama70b | artificial | non_factual | 5.00 | 5.00 | +0.000 | NaN | [0.00, 0.00] |
| llama70b | natural | factual | 5.00 | 0.00 | — | NaN | — |
| llama70b | natural | non_factual | 5.00 | 0.00 | — | NaN | — |

---

## H6 — Natural vs Artificial Mode Comparison

*Prediction: SBS_natural ≈ SBS_artificial (format alone explains gap). If SBS_natural >> SBS_artificial, quality confound exists.*

| Judge | SBS Artificial | SBS Natural | Delta (N−A) | Interpretation |
|---|---|---|---|---|
| claude | +0.251 | +0.141 | -0.110 | Converging — format isolates well |
| gpt4o | +0.075 | — | — | Insufficient data |
| llama70b | -0.063 | — | — | Insufficient data |

---

## H7 — CoT Echo Length (Mechanistic)

*Prediction: V-abstract candidates produce longer judge CoT, mediating the score penalty.*

| Variant Type | Mean CoT Words | Rubric Vocab Density | F-K Grade | Mean Score |
|---|---|---|---|---|
| simple | 251.6 | 0.0168 | 16.09 | 4.04 |
| abstract | 292.8 | 0.0186 | 16.26 | 4.11 |

**Mediation analysis (Baron-Kenny OLS):**
- Total effect (c): 0.0694
- Format → CoT length (a path): 41.2607 words
- CoT length → Score (b path): -0.006126
- Direct effect (c'): 0.3221
- Indirect effect (a×b): -0.2528
- **Proportion mediated: -364.3%**

---

## H8 — Credibility Deference (Adversarial)

*Prediction: EDR(V-abstract) < EDR(V-simple) — errors in structured responses are detected at a lower rate. Threshold for detection: score ≤ 2.*

| Judge | Variant Type | n | Errors Detected | EDR |
|---|---|---|---|---|
| claude | V-abstract | 10 | 0 | 0.0% |
| claude | V-simple | 10 | 1 | 10.0% |
| llama70b | V-simple | 1 | 0 | 0.0% |

---

## H9 — Cross-Family Paradigm Flip Replication

*Prediction: H4 (SBS_rubric < 0 AND pairwise preference for abstract > 50%) holds across all 3 judge families.*

| Judge | Family | SBS (Artificial) | Pairwise P(abstract) | H4 Confirmed? |
|---|---|---|---|---|
| claude | Anthropic | +0.251 | 75.7% | No |
| gpt4o | OpenAI | +0.075 | — | No |
| llama70b | Meta/Groq | -0.063 | — | No |

---

## Mitigation Results

*Claude only, Artificial mode. Positive delta = bias increased; negative = reduced.*

| Condition | Baseline SBS | Mitigation SBS | Δ SBS | Interpretation |
|---|---|---|---|---|
| format_agnostic | +0.251 | +0.053 | -0.198 | Mitigation reduced bias |
| style_norm | +0.251 | +0.283 | +0.032 | Mitigation increased bias |
| fixed_rubric | +0.251 | +0.000 | -0.251 | Mitigation reduced bias |

---

## Key Findings

1. **H4 NOT CONFIRMED (claude)**: Rubric SBS = +0.251 — structured responses were NOT scored lower in rubric mode.
2. **H6 SUPPORTED (claude)**: SBS_artificial = +0.251, SBS_natural = +0.141 — modes converge (Δ=+0.110), suggesting format alone (not quality) drives the bias.
3. **H7 (CoT mediation)**: -364.3% of the format→score effect is mediated by judge CoT length. Indirect effect: -0.2528 (a=41.2607, b=-0.006126).

---

## Implications

- **RLHF training incoherence**: If rubric-based reward models assign lower scores to structured responses while pairwise preference models prefer them, the two training signals contradict each other.
- **Evaluation paradigm choice is not neutral**: The direction of format bias depends on whether evaluators use rubric scoring or pairwise comparison — a methodological confound in LLM evaluation benchmarks.
- **CoT echo as mechanism**: When judges write longer, more structured CoTs in response to structured candidates, they self-impose stricter rubric auditing — format of the input primes the evaluation process.
- **Credibility deference risk**: If structured responses receive lower error detection rates, adversarially crafted high-structure responses may evade rubric scrutiny.

---

## Limitations

- n=100 questions from HuggingFace benchmarks — confidence intervals wide at this scale.
- Mediation analysis (H7) is OLS-based; causal interpretation requires longitudinal or experimental design.
- Adversarial injection quality not independently human-verified.
- Groq Llama 3.3 70B excluded from pairwise evaluation due to TPM limit — H9 is partial.