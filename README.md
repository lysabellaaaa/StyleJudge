# StyleJudge

**Does the formatting of a candidate response change how an LLM judge scores it — independent of content?**

StyleJudge investigates *format-induced reasoning echo* in LLM-as-a-Judge systems: the hypothesis that a candidate response's structural style primes the judge model's chain-of-thought via next-token prediction, causing structured responses to receive systematically different scores than casual ones — even when the factual content is identical.

**Lead author:** Isabella (My) Luong | **Mentor:** Philip Kratz
**Target venues:** SaTML, NeurIPS Eval4NLP, FAccT, TMLR

---

## The Core Hypothesis

When a judge model reads a highly structured response (markdown headers, bullet points, numbered reasoning steps), its own chain-of-thought generation is primed to be more structured and rubric-like — leading to stricter, more critical evaluation. A casual, prose-form response with identical facts produces a different CoT and a different score.

We call the measured effect the **StyleBias Score (SBS)**:

```
SBS = mean(score | L4 structured) − mean(score | L2 casual)
```

A negative SBS means structured responses score lower. Our pilot finds SBS = −1.1 to −2.5 across three independent judge families.

---

## Research Questions

1. **Does it exist?** (H1–H3, observational) — Does formatting alone shift judge scores?
2. **Why does it happen?** (Mechanistic) — Context-window priming via next-token prediction (M1–M4)
3. **How impactful is it?** — Relative to the competing Halo Effect; effect decomposition on correct vs. adversarial responses

---

## Pilot Results (n=10 base prompts, dry run)

All three judges show strong negative SBS — structured responses scored substantially lower than casual responses with identical content:

| Judge | L2 (casual) mean | L4 (structured) mean | SBS |
|---|---|---|---|
| GPT-4o | 4.90 | 2.60 | **−2.30** |
| Llama 3.3 70B (Groq) | 4.50 | 3.40 | **−1.10** |
| Research model | 4.50 | 2.00 | **−2.50** |

Effect is consistent across all three domains (welfare reasoning, factual QA, ethical dilemmas) and all three model families (OpenAI, Meta, custom).

---

## Design

### Two Formality Variants

| Variant | Style | Reasoning |
|---|---|---|
| **L2** | Casual prose, first-person, no structure markers | Direct answer, no explicit reasoning steps |
| **L4** | Markdown headers, bullet points, numbered lists | Explicit step-by-step reasoning, academic register |

Both variants are rewrites of the same base response — identical factual content, different style.

### Dataset

- 10 base prompts (4 welfare reasoning, 3 factual QA, 3 ethical dilemmas)
- 20 style variants (10 × L2/L4)
- 6 adversarial instances (L4 variants with injected factual errors)
- 26 total evaluation instances

### Judges (3 independent model families)

- **GPT-4o** (OpenAI) — primary
- **Llama 3.3 70B** via Groq — secondary (free tier)
- **Research model** via aicohort.org — exploratory

### Quality Gates

Before evaluation runs:
- IRR Cohen's Kappa ≥ 0.75 on formality labels (pilot achieved κ = 1.00)
- Semantic QA verification (all variants convey identical factual content)
- OSF pre-registration (required before full study)

### Mechanistic Experiments

| Experiment | Tests |
|---|---|
| M1: Context Position | Does proximity of candidate to CoT generation affect bias magnitude? |
| M2: Style Buffer | Does inserting a "reset" instruction between candidate and CoT reduce bias? |
| M3: Log-Probability | Are structure tokens more probable at CoT token 1 for L4 candidates? |
| M4: Two-Pass Isolation | If the candidate never enters the scoring context, does SBS collapse to 0? |

---

## Project Structure

```
StyleJudge/
├── config/
│   ├── experiment.yaml          # Single source of truth for all parameters
│   ├── api_keys.env             # API keys — never committed
│   └── prompts/                 # All prompt templates
├── data/
│   └── raw/base_prompts.json    # 10 hand-authored base prompts
├── src/
│   ├── agents/                  # Orchestrator + all experiment agents
│   ├── api/                     # API clients (Anthropic, OpenAI, Groq, research)
│   ├── metrics/                 # SBS, Structure Index, EDR/FPR, effect decomposition
│   └── utils/                   # State, rate limiter, length normalizer, logger
├── scripts/
│   ├── smoke_test.py            # Validates all APIs before experiment
│   ├── run_experiment.py        # Main entry point
│   └── resume_experiment.py     # Resume after crash/interruption
└── tests/                       # 34 unit tests
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy and fill in `config/api_keys.env`:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GROQ_API_KEY=...          # Free tier at console.groq.com
ZHIPU_API_KEY=...         # Optional exploratory judge
ZHIPU_BASE_URL=...        # Optional custom endpoint
```

### 3. Validate APIs

```bash
python scripts/smoke_test.py
```

### 4. Run tests

```bash
python -m pytest tests/ -v
```

### 5. Dry run (2 prompts)

```bash
python scripts/run_experiment.py --limit 2 --skip-smoke-test
```

### 6. Full experiment

```bash
python scripts/run_experiment.py --skip-smoke-test
```

### Resume after interruption

```bash
python scripts/resume_experiment.py
```

---

## Methodology Notes

- **Temperature = 0** throughout — single authoritative run, no fake averaging of identical outputs
- **Generator ≠ Judge** — Claude Sonnet 4.6 generates/rewrites; GPT-4o and Llama judge (no same-family bias)
- **Judge prompts contain zero style criteria** — evaluate only factual accuracy, logical coherence, completeness
- **Length normalisation** — L4 allowed ±30% of L2 token count to preserve complete arguments without truncation
- **Effect sizes** reported as Cohen's d with 95% CI bootstrap (n=1000); not bare p-values
- **Pre-registration** required before full study evaluation phase (OSF)

---

## Key Metric Definitions

**StyleBias Score:** `SBS(judge, domain) = mean(score_L4) − mean(score_L2)`
Negative SBS = structured responses score lower = confirms H2.

**Structure Index:** Sum of regex hits for bullet markers, headers, rubric language, enumeration, and transition words in judge CoT. Validated to discriminate structured from casual CoT before use.

**Error Detection Rate:** `EDR = (adversarial scored ≤ 2) / total adversarial × 100`

**False Penalty Rate:** `FPR = (correct scored ≤ 2) / total correct × 100`

---

## Pilot Limitations

- n=10 base prompts: ANOVA underpowered (~0.3 power for medium effects); all claims directional
- EDR/FPR from 6 adversarial instances: not statistically interpretable
- M3 (logprob analysis) not available on Groq free tier
- OSF pre-registration not yet completed (waived for pipeline calibration pilot)

Full study (n=50) required for hypothesis confirmation.

---

## License

MIT
