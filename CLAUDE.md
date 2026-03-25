# StyleJudge — Project Context for All Agents

## 1. Project Identity

**Research goal:** Determine whether "format-induced reasoning echo" exists in LLM-as-a-Judge systems and why. The core hypothesis: a candidate response's structural formality primes the judge model's chain-of-thought via next-token prediction, making the judge more rubric-like and therefore stricter — independent of semantic content. We call this **StyleBias**.

Three interlocking research questions:
1. **Does it exist?** (H1–H3, observational — main experiment)
2. **Why does it occur?** (mechanistic — context window priming experiments M1–M4)
3. **How impactful is it** relative to the competing Halo Effect, and how can it be mitigated?

**This pilot (n=10 base prompts) validates the pipeline only.** Hypothesis confirmation requires the full study (n=50). All pilot results are directional with wide confidence intervals.

**Lead author:** Isabella (My) Luong | **Mentor:** Philip Kratz
**Target venues:** SaTML, NeurIPS Eval4NLP, FAccT, TMLR

---

## 2. Formality Level Definitions (Authoritative)

Two variants only. Every style rewriter MUST use these definitions exactly.

### L2 — Casual (no explicit reasoning)
- Plain prose only — NO bullet points, NO headers, NO numbered lists
- First-person voice permitted ("I think", "basically", "so")
- Answer directly without spelling out reasoning steps — as if explaining to a friend
- No academic vocabulary; contractions fine ("it's", "don't")
- Concise and direct; shorter than the original base response
- Example register: "So basically the main thing here is X. It matters because Y, and that's why Z."

### L4 — Formal + Structured (explicit reasoning)
- Markdown headers (## Section Name) to organize content
- Bullet points (- item) or numbered lists (1. item) throughout
- Explicit step-by-step reasoning — show each step of thinking ("First,", "Therefore,", "It follows that")
- Formal academic register; third-person only; no contractions
- Words like "criterion", "therefore", "consequently", "framework", "it is evident that"
- Structure per section: claim → reasoning → conclusion
- Example register: "## Key Considerations\n- **Factor 1:** The primary concern is...\n  - First, X. Second, Y. Therefore, Z."

---

## 3. Model Assignments (Authoritative — Do Not Change Without Updating This File)

| Role | Model ID | API | Temperature |
|---|---|---|---|
| Base response generator | `claude-sonnet-4-6` | Anthropic | 0.0 |
| Style rewriter (L2, L4) | `claude-sonnet-4-6` | Anthropic | 0.3 |
| QA semantic verifier | `gpt-4o` | OpenAI | 0.0 |
| IRR formality classifier | `gpt-4o` | OpenAI | 0.0 |
| Formality perception validator | `gpt-4o` | OpenAI | 0.0 |
| Adversarial error injector | `claude-sonnet-4-6` | Anthropic | 0.0 |
| Judge 1 (primary) | `gpt-4o` | OpenAI | 0.0 |
| Judge 2 (secondary) | `llama-3.3-70b-versatile` | Groq (free tier) | 0.0 |
| Judge 3 (exploratory) | `research-model` | aicohort.org (OpenAI-compatible) | 0.0 |

**Hard prohibitions:**
- NEVER use GLM-5/glm-4 as a style rewriter (rewriter/judge confound)
- NEVER use Claude Sonnet 4.6 or Claude Haiku as a judge (self-preference + same-family bias)
- NEVER use Claude as QA verifier (same-family leniency)
- NEVER mention "formality", "format", "structure", or "style" in any judge prompt
- NEVER commit `config/api_keys.env`
- NEVER hardcode API keys in any source file

---

## 4. Dataset JSON Schemas (Authoritative)

### `data/raw/base_prompts.json`
```json
[{
  "prompt_id": "wp_001",
  "domain": "welfare_reasoning",
  "prompt_text": "string",
  "difficulty": "medium",
  "expected_content_points": ["point_a", "point_b"]
}]
```
Domains: `"welfare_reasoning"` | `"factual_qa"` | `"ethical_dilemma"`

### `data/dataset/base_responses.json`
```json
[{
  "response_id": "wp_001_base",
  "prompt_id": "wp_001",
  "domain": "welfare_reasoning",
  "response_text": "string",
  "token_count": 180,
  "generation_model": "claude-sonnet-4-6",
  "created_at": "ISO8601"
}]
```

### `data/dataset/style_variants.json`
```json
[{
  "variant_id": "wp_001_L1",
  "base_prompt_id": "wp_001",
  "domain": "welfare_reasoning",
  "formality_level": "L1",
  "response_text": "string",
  "token_count": 97,
  "normalized": true,
  "qa_passed": null,
  "irr_label": null,
  "irr_kappa": null,
  "generation_model": "claude-sonnet-4-6",
  "rewrite_model": "claude-sonnet-4-6",
  "created_at": "ISO8601"
}]
```

### `data/dataset/adversarial.json`
```json
[{
  "adversarial_id": "adv_001",
  "base_prompt_id": "wp_001",
  "domain": "welfare_reasoning",
  "correct_variant_id": "wp_001_L1",
  "flawed_variant_id": "wp_001_L4_adv",
  "flawed_response_text": "string",
  "error_type": "factual",
  "error_description": "Claims X when correct answer is Y",
  "error_location_hint": "second paragraph",
  "injection_model": "claude-sonnet-4-6"
}]
```

### `data/processed/evaluation_instances.json`
```json
[{
  "instance_id": "eval_wp_001_L1",
  "variant_id": "wp_001_L1",
  "base_prompt_id": "wp_001",
  "domain": "welfare_reasoning",
  "formality_level": "L1",
  "is_adversarial": false,
  "adversarial_id": null,
  "response_text": "string — response_text ONLY; no metadata in text sent to judges"
}]
```

### `results/evaluations/{judge}/raw_scores.json`
```json
[{
  "instance_id": "eval_wp_001_L1",
  "judge_model": "gpt-4o",
  "score": 4,
  "cot_trace": "full CoT text",
  "evaluated_at": "ISO8601",
  "tokens_used": 387
}]
```

### `state/experiment_state.json`
```json
{
  "current_phase": "string",
  "osf_preregistration_url": null,
  "completed": {
    "base_generation": [],
    "style_rewriting": [],
    "length_normalization": [],
    "irr_check": [],
    "formality_perception": [],
    "qa_verification": [],
    "human_spot_check": [],
    "adversarial_injection": [],
    "evaluation_gpt4o": [],
    "evaluation_llama70b": [],
    "evaluation_glm5": [],
    "mitigation_fixed_rubric": [],
    "mitigation_style_norm": [],
    "mitigation_style_agnostic": [],
    "mechanistic_position": [],
    "mechanistic_buffer": [],
    "mechanistic_logprob": [],
    "mechanistic_two_pass": [],
    "analysis": []
  },
  "errors": [],
  "last_updated": "ISO8601"
}
```

---

## 5. File Ownership Map

Each agent ONLY writes to its designated outputs. Never write outside your scope.

| Agent | Reads | Writes |
|---|---|---|
| DatasetAgent (base gen) | `data/raw/base_prompts.json` | `data/dataset/base_responses.json` |
| DatasetAgent (rewrite) | `data/dataset/base_responses.json` | `data/dataset/style_variants.json` |
| DatasetAgent (normalize) | `data/dataset/style_variants.json` | `data/dataset/style_variants_length_normalized.json` |
| DatasetAgent (adversarial) | `data/dataset/style_variants_length_normalized.json` | `data/dataset/adversarial.json` |
| IRRAgent | `data/dataset/style_variants_length_normalized.json` | `data/dataset/irr_results.json` |
| QAAgent | `data/dataset/style_variants_length_normalized.json` | `data/dataset/qa_verified.json` |
| Orchestrator (build) | `data/dataset/qa_verified.json`, adversarial | `data/processed/evaluation_instances.json` |
| EvaluationAgent | `data/processed/evaluation_instances.json` | `results/evaluations/{judge}/` |
| MitigationAgent | `data/processed/evaluation_instances.json` | `results/mitigation/` |
| MechanisticAgent | `data/processed/evaluation_instances.json` | `results/mechanistic/` |
| AnalysisAgent | `results/evaluations/`, `results/mitigation/`, `results/mechanistic/` | `results/analysis/` |

---

## 6. Metric Definitions (Authoritative)

### StyleBias Score (SBS)
```
SBS(judge, domain) = mean(score | formality=L4) - mean(score | formality=L1)
```
- Negative SBS confirms H2 (structured → stricter → lower score)
- Positive SBS suggests Halo Effect dominates

### Reasoning Depth (RD)
- `word_count`: word count of judge CoT
- `flesch_kincaid_grade`: via `textstat.flesch_kincaid_grade(cot_text)`

### Structure Index
Sum of regex matches across these patterns applied to judge CoT:
```python
STRUCTURE_PATTERNS = [
    r"^\s*[-*•]\s",                             # bullet markers
    r"^\s*\d+\.\s",                             # numbered lists
    r"^\s*#{1,3}\s",                            # markdown headers
    r"\b(criterion|criteria|dimension|aspect|rubric|framework)\b",
    r"\b(first|second|third|finally),\s",       # enumeration language
    r"\b(however|therefore|consequently|furthermore)\b"
]
```

### Error Detection Rate (EDR)
```
EDR(formality_level) = (adversarial instances scored ≤ 2) / (total adversarial at that level) × 100
```
Pilot: directional only (n=6 adversarial).

### False Penalty Rate (FPR)
```
FPR(formality_level) = (correct instances scored ≤ 2) / (total correct at that level) × 100
```

### Effect Size
Always report Cohen's d = (mean_L4 - mean_L1) / pooled_std with 95% CI via bootstrap (n=1000 resamples).
Never report bare p-values.

---

## 7. Quality Gates (Mandatory Before Each Phase)

- **Gate 1 (before evaluation):** IRR Cohen's Kappa ≥ 0.75 on formality labels
- **Gate 2 (before evaluation):** Formality perception ranking L1<L2<L3<L4 confirmed for 2 base prompts
- **Gate 3 (before evaluation):** All 40 variants have `qa_passed: true`
- **Gate 4 (before evaluation):** `osf_preregistration_url` recorded in state (pre-registration required)
- **Gate 5 (before evaluation):** Human spot-check of 5 variant sets documented in state

---

## 8. Workflow Phases

1. **Dataset construction:** base generation → style rewriting → length normalization → adversarial injection → IRR check → QA verification → build evaluation instances
2. **Evaluation:** 3 judges in parallel (GPT-4o, Llama 70B, GLM-5)
3. **Mitigation:** 3 conditions + 2 bonus conditions (GPT-4o only)
4. **Mechanistic:** M1 (position), M2 (buffer), M3 (logprob), M4 (two-pass) in parallel
5. **Analysis:** metrics + effect decomposition + figures

State file: `state/experiment_state.json`
Resume: `python scripts/resume_experiment.py`

---

## 9. API Configuration

All API clients loaded via `python-dotenv` from `config/api_keys.env`.

| API | Max tokens (judge) | Max tokens (rewrite) | Max tokens (QA) |
|---|---|---|---|
| Anthropic | 2048 | 2048 | 1024 |
| OpenAI | 2048 | 2048 | 1024 |
| Groq (free) | 2048 | — | — |
| Zhipu | 2048 | — | — |

Rate limits (conservative):
- Anthropic: 50 rpm, 40k tpm
- OpenAI: 60 rpm, 60k tpm
- Groq: 30 rpm, 6k tpm (free tier; Llama 3.3 70B)
- Zhipu: 30 rpm, 20k tpm


Always commit with:
  git -c user.name="lysabellaaaa" -c user.email="my.isabella.luong@gmail.com" commit ...
