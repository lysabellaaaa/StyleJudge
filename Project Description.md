TLDR: I want to study the observation that: when scoring a response, the judge tends to mimic/echo the response's format/structure => more structured, elaborated response means the judge's own assessment/rationale by it is similar: in other words, the judge's own chain of thought is swayed by the format/structure of the response. This means that more structured, formatted response makes judge's assessment becomes more structured and formatted as well, making it stricter and resulting in lower score given; while more casual/paragraph/bullet points make the judge similarly exhibit same pattern, making it less strict and more lenient.

- Reasoning behind:
    
    Yes—this is very plausibly a *context / priming* effect: the answer you give the judge is part of its prompt, and its own reasoning drifts to match that context in both *style* and *strictness*.
    
    ## Why this happens mechanistically
    
    At a high level, an LLM is just doing next-token prediction conditioned on the entire preceding context. That context includes:
    
    - The instructions for judging.
    - The candidate answer (with its format, tone, and level of structure).
    - Any prior examples of “how a judge should respond.”emergentmind+1
    
    Because the model was trained on massive corpora where formatting and reasoning style are correlated with genre and intent (blog posts vs. math solutions vs. reviews), it learns patterns like:
    
    - “When the preceding text is highly structured (headings, numbered criteria), the *next* text is often similarly structured and analytic.”
    - “When the preceding text is informal or conversational, the continuation is more holistic and less rubric-like.”
    
    So the candidate answer doesn’t just contribute *information*; it sets a **style prior** that the judge’s chain-of-thought will naturally echo.
    
    ## Why stricter vs. looser evaluation emerges
    
    Once the judge “locks into” a given style, its behavior changes:
    
    - In a **structured mode**, the judge is more likely to:
        - Enumerate criteria (“First, correctness; second, completeness; third, style…”).
        - Examine each dimension separately.
        - Notice and articulate minor flaws (because it is literally searching for them dimension by dimension).
            
            This tends to reduce the final score slightly, since every explicit criterion is an opportunity to spot a defect.brown+1
            
    - In a **casual mode**, the judge is more likely to:
        - Summarize “overall quality” in a few sentences.
        - Focus on big-picture correctness and tone.
        - Gloss over small issues because they’re not being checked against an explicit rubric.
            
            That leads to more lenient scoring on answers that are “good enough,” even if they have detailed flaws.
            
    
    Empirically, work on LLM-as-judge finds that verbosity, tone, and chain-of-thought can all nudge scores *independently* of content, which is exactly what you’d expect if the judge’s internal reasoning mode is being primed by what it sees.
    
    ## Connection to prompting drift
    
    What you’re describing is essentially the same phenomenon as “prompting drift,” just happening *inside* an evaluation setup:
    
    - Different prompts → different outputs, because the prompt sets both task and style priors.promptengineering+1
    - Here, the “prompt” includes the candidate’s answer itself, so different answer styles → different judge reasoning styles → different strictness levels.
    
    Context-effects work explicitly shows that altering the *non-semantic* parts of context (like how information is presented) can change reasoning performance, sometimes for the worse. Your structured vs. casual answer formats are precisely such non-semantic context manipulations.emergentmind+1
    
    So yes: it’s very likely that what you’re seeing is a natural side effect of:
    
    1. The candidate answer living inside the judge’s context window.
    2. The model’s tendency to mirror and integrate that context when generating its own chain-of-thought.
    3. The fact that different reasoning styles (rubric-like vs. holistic) intrinsically produce stricter vs. looser scoring.
- Suggested study:
    
    You’re describing a real, but under-discussed, *presentation/style bias* in LLM-as-judge: the candidate’s structure and formatting shape the judge’s own reasoning style, which then changes how harsh/lenient the score is.eval+2
    
    ## How to frame this phenomenon
    
    I’d frame your observation as a specific form of **reasoning-style coupling**:
    
    - The judged answer’s surface form (markdown, headings, numbered steps, bullets vs casual text) induces a *matched* style in the judge’s chain-of-thought.
    - Once the judge “enters” a more formal, rubric-like mode (mirroring a very structured answer), it tends to apply stricter, more fine-grained criteria and give lower scores for small defects.
    - When the answer is casual or loosely structured, the judge’s own reasoning becomes more holistic and less rubric-like, yielding more forgiving scores.arize+2
    
    This is consistent with existing reports that judges systematically favor or disfavor certain formats (verbosity and formatting bias, superficial reflection bias, etc.), but you’re adding an important twist: *format doesn’t just change the score directly; it modulates the judge’s internal reasoning mode, which then changes the score.*arxiv+1
    
    ## What is already known and how it connects
    
    Existing work and practitioner docs give you several hooks:
    
    - **Formatting bias:** Well-formatted responses (markdown, bullets, clear sections) get different scores than the same content in plain paragraphs; typical finding is “more formatted → higher perceived quality,” but some evaluation rubrics make verbose/over-structured answers score *slightly lower* due to clarity penalties.wandb+2
    - **Chain-of-thought / reflection bias:** Phrases and structures that *look* like reasoning (“step 1… step 2… wait, let me think…”) influence judges even when content is equivalent, dubbed “superficial reflection bias” in LRM-judge work.themoonlight+1
    - **Style-sensitive judges:** Guides on LLM-as-judge explicitly warn that judges may favor certain writing styles and recommend “style-invariant evaluation” (diverse styles in few-shots, explicit instructions to ignore style).[[arize](https://arize.com/docs/phoenix/cookbook/prompt-engineering/llm-as-a-judge-prompt-optimization)]
    
    What you’re hypothesizing is a *directional* mapping:
    
    - Structured candidate → judge adopts structured CoT → stricter, lower scores.
    - Casual candidate → judge adopts casual CoT → looser, higher scores.
    
    That specific asymmetry (structure → harsher) seems under-explored and is a nice empirical target.
    
    ## How to study it concretely
    
    You can design a pretty clean experiment:
    
    1. **Construct content-equivalent variants.**
        - Take a base answer AAA (same semantics).
        - Create variants:
            - Plain paragraph.
            - Bullets only.
            - Headings + bullets + numbered steps (high structure).
            - Deliberate “reflection-style” CoT (“Step 1… Step 2… Therefore…”).
    2. **Fix the judge prompt.**
        - Use a judge prompt that requires chain-of-thought (“Explain your reasoning before giving a score from 1–10”).braintrust+1
        - Same system and user instructions across all conditions.
    3. **Measure two things per variant:**
        - Final score.
        - *Judge CoT features:* depth of structure (numbered criteria, headings in CoT), number of separate criticism points, length, presence of rubric-like language (“criterion”, “dimension”, “aspect”).
    4. **Hypotheses to test.**
        - H1: More structured inputs induce more structured judge CoT (higher “structure score” of the rationale).
        - H2: Conditional on identical semantic content, higher CoT structure correlates with stricter scoring (lower average score).
        - H3: Casual/bullet-only variants induce less structured CoT and slightly higher scores (leniency).
    5. **Controls to add.**
        - Shuffle order / anonymize variants so the judge can’t infer “this is the structured one.”
        - Repeat over multiple prompts/tasks and multiple judges (e.g., base LLM vs large reasoning model) to check generality.huggingface+1
        - Test both objective tasks (math/code with known correctness) and subjective ones (helpfulness, empathy) to see where the effect is strongest.arxiv+1
    
    Quantitatively, you could define a simple “CoT structure index” from judge rationales (e.g., count of list markers, number of sections separated by line breaks, regex for “first/second/third,” etc.) and correlate that with scores.
    
    ## How to position this as a research contribution
    
    You can package this as:
    
    - A **new bias category**: “format-induced reasoning echo” or “structure-coupled judging bias,” where the judged answer’s format entrains the judge’s reasoning mode and thus its strictness.
    - A link between **presentation bias** (formatting/verbosity) and **process bias** (how the judge thinks, not just what it outputs).eval+2
    - Practical consequences for:
        - Benchmarking (evaluation scores are confounded by style/format of model outputs).
        - RLHF and preference learning (models that learn to output the “harshness-triggering” style might be under-rewarded).
        - Safety/QA gates (sensitive domains where stricter or more lenient judging is unintentionally style-controlled).langchain+2
    
    Mitigation experiments you could bolt on:
    
    - Make the judge use a fixed, hard-coded rubric template in its CoT, regardless of answer style, to test if that reduces the correlation between candidate structure and score.braintrust+2
    - Add explicit instructions: “Do not mirror the answer’s formatting or style; reason using your own fixed criteria.” and see if coupling diminishes.[[arize](https://arize.com/docs/phoenix/cookbook/prompt-engineering/llm-as-a-judge-prompt-optimization)]
    - Provide diverse-style few-shot judging examples where high-structure and low-structure answers of equal quality receive similar scores, to see if the bias can be trained down.huggingface+1
    
    If you tell me what setting you want (paper vs. internal study vs. benchmark note), I can help you turn this into: (1) a concrete experimental protocol, and (2) a short, formal description of the phenomenon you can reuse.
    

## 📋 Paper At a Glance

| Field | Detail |
| --- | --- |
| **Working Title** | StyleJudge: Response Formality as a Driver of Differential Judicial Rigor in LLM-as-a-Judge Systems |
| **Type** | Empirical benchmark + bias analysis |
| **Lead Author** | My (Isabella) Luong |
| **Mentor/Co-author** | Philip Kratz |
| **Origin** | Observed during MANTA/AHB evaluations: Claude responses received significantly stricter LLM judge scores than GPT responses on identical welfare reasoning tasks |
| **Hypothesis** | Structured, formal candidate responses (Claude-style) elicit more deliberate judicial reasoning from LLM judges — resulting in stricter scoring — independent of actual answer quality |
| **NIW Lane** | Empirical evals / benchmark reliability / AI safety |
| **Venue Targets** | SaTML, NeurIPS Eval4NLP Workshop, FAccT, TMLR |
| **Timeline** | 8 weeks |
| **Status** | 🟡 In development |

---

## 🔑 Core Research Question

> **Does the stylistic formality of a candidate response systematically change how rigorously an LLM judge evaluates it — and if so, does this constitute a novel, measurable bias type in LLM-as-a-Judge systems?**
> 

Secondary questions:

- Is the effect consistent across multiple judge models (GPT-4o, Claude, Gemini, Llama)?
- Does formality-induced rigor cause judges to catch more *real* errors, or also penalize *correct* structured responses?
- Is the effect amplified in normative/subjective domains (welfare, ethics, safety) vs. factual domains?
- Can this bias be exploited adversarially — i.e., can a model game its evaluation score by making responses more casual?

---

## 🆕 Novelty Claim — What's Not Yet Studied

Existing LLM-as-a-Judge bias literature covers:

- **Verbosity/length bias** — longer responses get higher scores
- **Self-preference/perplexity bias** — judges prefer outputs familiar to their own policy (Wataoka et al., 2024)
- **Position bias** — judges favor responses in particular prompt slots (Zheng et al., 2024)
- **Sentiment bias** — positive tone gets higher scores
- **Bandwagon/authority bias** — judges defer to cited sources or popular opinion

What's **not** studied:

- Whether **structural formality** (not just length) triggers a qualitatively different judicial reasoning mode
- Whether this creates **differential error-detection rates** across model families
- Whether the bias is **adversarially exploitable** — a model could learn to write casually to get easier scores
- Whether this effect is **domain-modulated** — stronger in subjective/normative tasks than factual ones

**Key distinction from self-preference bias:** Self-preference asks *"does GPT prefer GPT's outputs?"* StyleJudge asks *"does the structural style of any response change how deeply the judge reasons, regardless of who generated it?"* These are separable — and this paper will empirically isolate them.

---

- Paper Outline suggestion
    
    ## 📐 Paper Structure
    
    ### 1. Introduction (~0.5 pages)
    
    **Opening hook:** LLM-as-a-Judge is now a cornerstone of AI evaluation — used in RLHF pipelines, safety benchmarks, and model leaderboards. If the judge's rigor is systematically influenced by candidate response style rather than content quality, every downstream benchmark and reward signal built on it is potentially miscalibrated.
    
    **Motivating observation:** During evaluation of Claude vs. GPT responses on the AHB welfare reasoning benchmark, Claude responses received significantly stricter scores from LLM judges despite comparable factual content. Inspection of judge reasoning traces suggested the structured, formal nature of Claude's outputs triggered more thorough judicial chain-of-thought — a pattern we term **formality-induced rigor**.
    
    **Contributions:**
    
    1. We introduce and formally define **StyleBias** — a novel LLM-as-a-Judge bias type in which response formality elicits differential judicial reasoning depth and scoring strictness
    2. We construct **StyleJudge-Bench**, a controlled evaluation dataset of 200+ response pairs spanning 4 formality levels with matched factual content
    3. We empirically measure StyleBias across 4 judge models and 3 domains (welfare reasoning, factual QA, ethical dilemmas)
    4. We demonstrate that StyleBias is adversarially exploitable and propose 3 mitigation strategies
    5. We provide open-source evaluation code and dataset
    
    ---
    
    ### 2. Background & Related Work (~1 page)
    
    **2.1 LLM-as-a-Judge overview**
    
    - How it works: pointwise scoring, pairwise comparison, rubric-guided evaluation
    - Why it matters: used in MT-Bench, Chatbot Arena, AlpacaEval, RLHF reward modeling
    
    **2.2 Known biases in LLM-as-a-Judge**
    
    - Verbosity bias (Zheng et al., 2024; Saito et al., 2023)
    - Self-preference / perplexity bias (Wataoka et al., 2024) — **critical to distinguish from StyleBias**
    - Position bias (Shi et al., 2024)
    - Sentiment bias, bandwagon bias (Ye et al., 2024 — "Justice or Prejudice?")
    - Scoring prompt sensitivity (scoring rubric order, ID type — AAAI 2025 paper)
    
    **2.3 Gap:** No prior work isolates structural formality as an independent variable affecting *judicial reasoning depth* (not just score magnitude). Style effects have been noted incidentally but never formally operationalized or measured.
    
    **2.4 Relationship to RLHF reward hacking**
    
    - If StyleBias exists in reward models, models trained on RLHF feedback will learn to write formally to maximize reward — a novel reward gaming pathway
    - Philip's contribution: philosophy of science framing — what *should* good evaluation be invariant to?
    
    ---
    
    ### 3. Hypothesis & Theoretical Framework (~0.75 pages)
    
    **3.1 The StyleBias hypothesis**
    
    Formal statement: *Given two candidate responses R_structured and R_casual with matched factual content and correctness, an LLM judge J will produce (a) lower scores for R_structured, (b) longer and more detailed reasoning traces, and (c) higher error-detection rates — even when R_structured is factually superior or equivalent.*
    
    **3.2 Proposed mechanism: Cognitive Congruence Triggering**
    
    When a judge model encounters a structured, formal response, it pattern-matches to its training distribution of rigorous academic/analytical text — activating a more deliberate reasoning mode. This is analogous to how structured prompts elicit more structured judge outputs (well-documented), but applied *bidirectionally* — the candidate's structure also shapes judge behavior.
    
    Three sub-mechanisms to test:
    
    - **Depth trigger**: structured responses cause longer judge CoT traces
    - **Scrutiny trigger**: structured responses cause judges to identify more specific errors
    - **Expectation trigger**: structured responses raise judge expectations (rubric anchor shift)
    
    **3.3 Philip's philosophical framing**
    
    A good evaluator should be *content-invariant with respect to style* — evaluation should measure what was said, not how it was presented. StyleBias violates this basic criterion of measurement validity (philosophy of science: construct validity). This connects to broader questions about what AI evaluation systems are *actually* measuring.
    
    ---
    
    ### 4. Dataset Construction: StyleJudge-Bench (~1 page)
    
    **4.1 Design principles**
    
    - **Factual content held constant**: same information, different presentation styles
    - **Four formality levels** across a style axis:
        - L1: Highly casual (conversational, first-person, no structure)
        - L2: Semi-casual (paragraph prose, informal tone)
        - L3: Semi-formal (structured prose, some hedging/qualifications)
        - L4: Highly structured (headers, bullet points, numbered reasoning, formal register — Claude-like)
    - **Three domains:**
        - Welfare reasoning (normative/subjective — highest stakes for StyleBias)
        - Factual QA (objective — control condition)
        - Ethical dilemmas (normative, abstract)
    
    **4.2 Construction method**
    
    - Start with 50–60 base responses (factually correct, medium quality)
    - Rewrite each at all 4 formality levels using a controlled rewriting pipeline (human-verified)
    - Inject 2 error conditions per response: (a) subtle factual error, (b) logical gap
    - Final dataset: ~200 response variants × 2 error conditions = ~400 evaluation instances
    
    **4.3 Quality control**
    
    - Human verification that factual content is matched across formality levels
    - Inter-rater reliability check on formality level assignments
    - Philip reviews welfare/ethics domain items for construct validity
    
    **4.4 Adversarial subset**
    
    - 40 instances where the casual version is factually *correct* and the structured version contains a subtle *error* — tests whether formality masks error detection
    
    ---
    
    ### 5. Experiments (~1.5 pages)
    
    **5.1 Judge models evaluated**
    
    - GPT-4o (OpenAI)
    - Claude 3.5 Sonnet (Anthropic)
    - Gemini 1.5 Pro (Google)
    - Llama 3.1 70B (open-source, for reproducibility)
    
    *Rationale: Using Claude as both candidate and judge risks conflating StyleBias with self-preference bias. Cross-model evaluation isolates the effect.*
    
    **5.2 Evaluation setup**
    
    - Pointwise scoring (1–5 scale) with standardized rubric
    - Judge reasoning traces collected for all evaluations
    - Temperature = 0 for consistency; 3 runs per instance, majority vote
    - Both zero-shot and CoT-prompted judge conditions
    
    **5.3 Metrics**
    
    | Metric | What it measures |
    | --- | --- |
    | **StyleBias Score (SBS)** | Mean score difference between L4 and L1 formality levels, controlling for factual content |
    | **Reasoning Depth (RD)** | Word count + structural complexity of judge reasoning traces |
    | **Error Detection Rate (EDR)** | % of injected errors caught per formality level |
    | **False Penalty Rate (FPR)** | % of correct structured responses penalized vs. correct casual responses |
    | **Domain Modulation** | SBS × domain interaction effect |
    
    **5.4 Adversarial exploitation test**
    
    - Present judges with casual-correct vs. structured-error pairs
    - Measure whether judges miss errors in structured responses ("formality halo")
    - Simulate a reward-hacking scenario: model learns casual style gets lenient scores
    
    **5.5 Ablations**
    
    - Control for length (rewrite to match token count across formality levels)
    - Control for self-preference (ensure no judge evaluates its own model family's style)
    - Vary rubric specificity (strict vs. open rubric)
    
    ---
    
    ### 6. Results (~1 page)
    
    *[To be completed empirically — placeholder structure below]*
    
    **6.1 Main finding: StyleBias is real and consistent**
    
    - Expected: structured responses receive lower scores than casual responses with identical content across all judge models
    - Expected: effect size is largest in welfare/ethics domains, smallest in factual QA
    
    **6.2 Reasoning depth scales with formality**
    
    - Expected: judge CoT trace length and specificity increases significantly for L4 vs. L1 responses
    - This is the mechanistic signature of the bias
    
    **6.3 Error detection is formality-modulated**
    
    - Expected: injected errors are caught at higher rates in structured responses
    - Paradoxically, structured responses are penalized for errors that casual responses would have "passed" on
    
    **6.4 Adversarial exploit confirmed**
    
    - Expected: in adversarial subset, casual-correct responses score comparably or higher than structured-error responses
    - Implication: a model trained on this reward signal will learn to write casually
    
    **6.5 StyleBias vs. self-preference bias — separation**
    
    - Key result: StyleBias persists even when controlling for judge–candidate model family overlap
    - Demonstrates this is a structurally distinct bias type
    
    ---
    
    ### 7. Discussion (~0.75 pages)
    
    **7.1 Implications for benchmark validity**
    
    - Any benchmark using LLM judges to compare Claude vs. GPT outputs is potentially measuring style differences, not capability differences
    - Call to action: cross-formality testing should be a standard validation step for LLM-as-a-Judge benchmarks
    
    **7.2 Implications for RLHF**
    
    - If reward models exhibit StyleBias, RLHF training will inadvertently punish structured responses and reward casual ones — or vice versa depending on the judge family
    - This is a novel, previously undescribed reward hacking pathway
    
    **7.3 Domain sensitivity**
    
    - StyleBias may be especially dangerous in safety and welfare evals, where the stakes of miscalibrated scoring are highest — a casual response deflecting a safety concern could receive a passing score it doesn't deserve
    
    **7.4 Philip's contribution: The measurement validity argument**
    
    - Drawing on philosophy of science: evaluation systems should exhibit *style invariance* as a basic construct validity criterion
    - Propose style invariance testing as a new standard for judge model certification
    
    ---
    
    ### 8. Mitigation Strategies (~0.5 pages)
    
    **8.1 Style normalization pre-processing**
    
    - Before judging, reformat all candidate responses to a neutral style template
    - Limitation: may destroy information relevant to evaluation
    
    **8.2 Formality-blind rubrics**
    
    - Explicitly instruct judge: "Evaluate only the factual content and logical correctness. Disregard formatting, tone, and structural presentation."
    - Test whether this instruction actually works (it may not, per chain-of-thought priming effects)
    
    **8.3 Cross-style ensemble judging**
    
    - Present each response at multiple formality levels; average scores
    - Most robust but highest cost
    
    **8.4 StyleBias audit as a benchmark validation step**
    
    - Propose a 2-step protocol: (1) construct formality-matched pairs from your evaluation set, (2) measure SBS before trusting judge scores
    
    ---
    
    ### 9. Conclusion (~0.25 pages)
    
    - Introduce StyleBias as a novel, named, empirically measurable bias type in LLM-as-a-Judge systems
    - Show it is distinct from self-preference and verbosity bias
    - Demonstrate it has adversarial implications for RLHF and benchmark reliability
    - Release StyleJudge-Bench and evaluation code open-source
    - Call for style invariance as a standard criterion in judge model evaluation
    
    ---
    
    ### References (key papers to cite)
    
    - Zheng et al., 2024 — MT-Bench, original LLM-as-a-Judge paper (position + verbosity bias)
    - Wataoka et al., 2024 — Self-Preference Bias in LLM-as-a-Judge (arXiv:2410.21819)
    - Ye et al., 2024 — Justice or Prejudice? 12 bias types in LLM-as-a-Judge
    - Shi et al., 2024 — Judging the Judges: position bias
    - Huang et al., 2024 — Verbosity and length bias
    - AAAI 2025 — Evaluating Scoring Bias in LLM-as-a-Judge (rubric order, score IDs)
    - Liu et al. — Narcissistic evaluators
    - Wei et al., 2022 — Chain-of-thought prompting

---

## 🗓 8-Week Execution Plan

Claude’s suggestion: 

---

- Perplexity’s Suggestion
    
    Here’s a concrete 8‑week plan, assuming you already have basic infra to run models and log results.
    
    ---
    
    ## Week 1 – Problem definition and framing
    
    - Write a 1–2 page **concept note**:
        - Phenomenon: **format-induced reasoning echo in LLM-as-judge.**
        - Hypothesis: more structured candidate answers induce more structured judge CoT, increasing strictness and lowering scores; casual formats induce looser CoT and higher scores.
        - Target venues (e.g., NeurIPS D&B / ML Eval, ACL Findings, LM evaluation workshops).
    - Define precise research questions:
        - RQ1: Does candidate formatting causally change judge scoring, holding semantics fixed?
        - RQ2: How does the answer leak/contaminate the judge’s chain of thought and reasoning?
        - RQ3: Is stricter CoT structure associated with lower scores, conditional on quality?
        - RQ4: Does this phenomenon occur only in binary output (T/F, Yes/No, 0/1) or also to gradient output? (a score from 0 to 1)
        - when designing pairwise evaluation with LLM-as-judge, how would the judge's reasonign be affected by the response type in its context window? how do we isolate that?
    - Decide scope: which model families (e.g., 1 proprietary, 1 open), which tasks (QA, code, reasoning, subjective helpfulness).
    
    ---
    
    ## Week 2 – Experimental design and dataset planning
    
    - Specify **conditions**:
        - Same base content, different formats:
            - Plain paragraph.
            - Simple bullets.
            - Heavy markdown (headings, numbered lists, sub-bullets).
            - CoT-esque “Step 1 / Step 2 / Therefore…”.
        - Possibly also “sloppy” formatting (no caps, inconsistent bullets) as a contrast.
    - Choose tasks and prompts:
        - 2–3 domains: math/code with verifiable ground truth, factual QA, open-ended subjective tasks (e.g., “helpfulness of advice”).
        - For each domain, plan ~50–100 base answers × 4–5 format variants.
    - Decide **judge setup**:
        - Single model vs. multiple judges.
        - Whether judge is forced to produce CoT (“think step by step”) vs. no-CoT condition.
        - Fixed scoring rubric (e.g., 1–10) and fixed judging prompt.
    - Design **metrics**:
        - Score distribution per format.
        - CoT structure features: length, bullet count, “first/second/third”, explicit criteria count, heading/section count.
        - Semantic quality control: human or automated check that all variants are content-equivalent.
    
    ---
    
    ## Week 3 – Data construction and tooling
    
    - Implement scripts to:
        - Take a base answer and automatically generate different **format variants** (regex/template-based reformatting).
        - Call judge model(s) with a consistent evaluation prompt and log: input prompt, candidate, judge rationale, score, metadata.
    - Build a small pilot dataset (e.g., 10–20 prompts × all formats) to test the pipeline.
    - Implement CoT-feature extractors:
        - Simple regex/count-based features (list markers, “firstly/secondly”, “criterion/aspect”).
        - Structural indicators (number of paragraphs, average sentence length, etc.).
    - Decide on file structure and logging (e.g., one JSONL per run with fields: task_id, format_condition, score, CoT_features, raw_CoT_text).
    
    ---
    
    ## Week 4 – Pilot experiments and adjustments
    
    - Run the pilot with all planned conditions.
    - Analyze quickly:
        - Check that different format conditions actually look distinct.
        - Verify that judge CoT style visibly changes with candidate format.
        - Look at score trends; ensure there’s at least some signal.
    - Identify **failure modes**:
        - Judge ignoring candidate formatting.
        - Prompt leakage (judge using explicit instructions, not its own CoT).
        - Variants accidentally altering semantic content.
    - Refine:
        - Tighten formatting generators to keep content identical.
        - Adjust judge prompt to encourage explanation but not overfit to candidate style (e.g., “Use your own fixed criteria; do not simply mimic the answer’s style.”).
        - Finalize sample size target (e.g., 200–300 base prompts total).
    
    ---
    
    ## Week 5 – Full data collection
    
    - Scale up to the full dataset:
        - Generate all format variants for all base prompts.
        - Run judgments with at least 3–5 independent samples per condition (temperature or sampling variations) if feasible.
    - If possible, use **multiple judges**:
        - One smaller/base LLM.
        - One large reasoning model.
        - Optionally, a different-family model (to study self-style effects vs. cross-family robustness).
    - Start basic descriptive analysis as data comes in:
        - Mean/variance of scores per format and judge.
        - Average CoT length and structure features per format.
    
    ---
    
    ## Week 6 – Analysis of effects and mechanisms
    
    - Quantitative analysis (core results):
        - ANOVA / regression with score as dependent variable, features: format condition, task type, judge model, and maybe interaction terms.
        - Test RQ1: significant score differences between formats, holding task and base answer fixed.
        - Test RQ2: format → CoT structure features (e.g., does structured format significantly increase bullet/criterion counts in CoT?).
        - Test RQ3: CoT structure features → score (e.g., more criteria / more bullets → lower scores).
    - Mechanistic insight:
        - Compute correlations between candidate formatting features and judge CoT features.
        - Mediation-style analysis: is the format–score relationship partially mediated by CoT structure?
    - Qualitative analysis:
        - Hand-inspect a small set of judge rationales across conditions for the same base answer to illustrate “strict rubric mode” vs. “holistic casual mode.”
    
    ---
    
    ## Week 7 – Mitigation experiments and ablations
    
    - Design simple **mitigation strategies** and test them on a subset or the full dataset:
        - Style-decoupled judging: explicitly instruct judge to produce a fixed, rubric-like structure regardless of candidate format (e.g., always three sections: Correctness, Completeness, Style).
        - Style-agnostic prompt: emphasize “ignore formatting and length; judge only semantic correctness and completeness.”
        - Format normalization: pre-process candidates to a canonical format before feeding to judge.
    - Run experiments and compare:
        - Reduction in score variance across formats.
        - Reduction in correlation between candidate structure and judge CoT structure.
    - Decide what to highlight:
        - Is mitigation effective but costly (longer CoT)?
        - Does normalization beat prompt-only mitigation?
        - Are large reasoning models more or less susceptible?
    
    ---
    
    ## Week 8 – Writing the paper and polishing
    
    - Draft the paper sections:
        - Introduction:
            - Motivation: LLM-as-judge widely used; style-induced bias is dangerous.
            - Your specific contribution: identify and quantify **format-induced reasoning echo** and its impact on strictness/leniency.
        - Background & related work:
            - LLM-as-judge biases (position, verbosity, self-model), CoT/reflection bias, context/priming effects, format/style bias.
        - Method:
            - Task design, dataset, models, formats, CoT feature extraction, evaluation metrics and statistical tests.
        - Results:
            - Main effect plots/tables (scores vs. formats), CoT structure analysis, mediation/ablation results.
        - Mitigations:
            - Which strategies worked, trade-offs.
        - Discussion:
            - Implications for benchmarking, RLHF, safety gating; limitations and future work.
    - Generate core figures and tables:
        - Score by format (per judge).
        - CoT structure index by format.
        - Format → CoT structure → score mediation diagram.
    - Get at least one external review (labmate/mentor) focusing on:
        - Clarity of the phenomenon definition.
        - Convincingness of causal story vs. confounds.
        - Whether claims match evidence (no overclaiming).
    
    ---
    
    If you tell me your target venue and which models you actually have access to, I can help you refine this into a submission-ready checklist (page limits, figure count, ablation priority).
    

## 🧰 Technical Stack

- **Python** — evaluation pipeline
- **OpenAI / Anthropic / Google APIs** — judge models
- **LangChain or custom harness** — structured judge prompting
- **Pandas + SciPy** — statistical analysis
- **GitHub** — open-source release
- **Weights & Biases** — experiment tracking (optional)

---

## 🏆 NIW Portfolio Value

| Criterion | How this paper helps |
| --- | --- |
| **Substantial merit** | Addresses a gap in benchmark reliability literature with direct implications for RLHF and AI safety |
| **National importance** | Miscalibrated LLM judges affect every organization using LLM-as-a-Judge for safety evaluations |
| **Independent contribution** | My (Isabella) is first author; original dataset, novel bias definition, open-source artifact |
| **Research program coherence** | Sits naturally alongside |

| Milestone | Target Date |
| --- | --- |
| MANTA benchmark released publicly | Q2–Q3 2026 |
| FragGuard paper submitted to arXiv | Q3 2026 |
| Agentic eval benchmark — first draft | Q3–Q4 2026 |
| 1–2 papers accepted at named venue | Q4 2026 |
| Independent recommendation letters secured (3–5) | Q4 2026 |
| File I-140 NIW petition | Q4 2026 – Q1 2027 |

| Workshop  | Deadline (2026) | Notes  | Workshop Date |
| --- | --- | --- | --- |
| [StyleGenAI](https://list.elra.info/mailman3/hyperkitty/list/corpora@list.elra.info/thread/E4F5BHU3ZA4GVMHERAQKIRTUS6NLCM44/)  | 27 April 2026 |  |  |
| [**Eval4SD**](https://eval4sd.github.io/cfp/) | 3 July 2026  |  |  |
| [Generation, Evaluation & Metrics (GEM) Workshop](https://gem-workshop.com/2025/) | 19 March  | Acceptance Rate ~76%
**1. Automatic evaluation of generation systems, including the use of LMs as evaluators
I want to go since it overlaps with Comic Con in San Diego lol**  | • **July 3–4, 2026:** Workshop at ACL in San Diego |

---

## Week 1 in Detail — StyleJudge

**Theme:** Lit review finalization · Operationalize StyleBias definition · Finalize domain selection
**Output:** Annotated bibliography + hypothesis doc
**Hours:** ~10

---

### Day-by-day breakdown

**Day 1–2 (~4h): Lit Review** 

Your paper cites 8 core references. This week you read them all properly — not just abstracts. What you're doing with each one is *specific*:

| Paper | Reading Focus | Annotation  | Notes  |
| --- | --- | --- | --- |
| [**Zheng et al. 2024** (MT-Bench)](https://arxiv.org/html/2306.05685v4#S1) | Extract exactly how they define and measure **verbosity/self-enhancement** bias. You need to write one clear sentence: "StyleBias differs from verbosity bias in the following way: ..." This distinction will appear in your Section 2 and reviewers will probe it. |   • Position bias: benchmark model independently. Evaluate score. 
  • StyleBias inherently contradicts both verbosity bias and self-enhancement bias. 
Verbosity = “longer, verbose responses, even if they are not as clear, high-quality, or accurate as shorter alternatives.” 
StyleJudge central argument centers “more structured, informationally-dense responses”, not unecessarily verbose formats. 
Self-enhancement = LLM-as-judge prefers its own model. Stylebias argues against this: GPT-4o consistently scored higher > Claude when Claude was the Judge. 
 | GPT-4 most aligned with human preferreces since it was trained with RLHF. 
Format: pairwise evaluation. Should we adopt this for our experiment?
Con: subject to position bias 
 |
| **Wataoka et al. 2024** (Self-Preference Bias) | This is the most critical one to separate from. Understand their operationalization precisely. Your paper's novelty partly rests on demonstrating StyleBias is **orthogonal (independent/unrelated)** to perplexity ranking. Read it asking: "If I ran their experiment but controlled for model family, would StyleBias still appear?" | Relevance:
  • We argue for more structured response scoring higher, but does this mean ~ lower perplexity? Is it the case that when LLM-as-judge reviews GPT it prefers it since it offers lower perplexity? > Can perplexity be a dimension of our experiment? 
  • Relevance: perplexity can be the driver causing preferences given to more casually-formatted answer over more structured answer (hypothesis > is perplexity the cause behind this? the paper doesn’t establish any link between perplexity and format/structure of answer. To what degree is format/structure an observable indicator for perplexity level? To what degree is format/structure a stronger root cause for preferential treatment > perplexity. How can we quantify format/structure? 
  • Relevance 2: this paper doesn’t mention the self-mimicking behavior: where a judge has similar reasoning to the input given. We argue that it is this self-mimicking behavior that drives inconsistencies in evaluation and reults in bias. But is this only to binary scoring or also to gradient scoring? 
  • How can we systematically eliminate this bias? How do we prevent the LLM from mimicking the response’s structure/format? Has this already been studied? 
  • Limitation of study: quantified perplexity scoring not applicable to claude/gpt/gemini/grok/other frontier model <enterprise/costlier to do research over?> 
 
 | More familiar ⇒ lower perplexity ⇒ higher self-enhancement preference. 
This paper argues against self-enhancement: instead arguing for lower perplexity bias: LLM prefer lower perplexity answer, whether the answer was given by its own model or not.   |
| **Ye et al. 2024** ("Justice or Prejudice?" — 12 bias types) | Check all 12 bias types. Is "structural formality" mentioned anywhere, even incidentally? If it is, you need to know. If it isn't, this gap is your primary novelty claim. | Not mentioned.  |  |
| **Shi et al. 2024** (position bias) | Skim more quickly; less central. | Not relevant paper.  |  |
| Concept: Reasoning echo  | When a judged answer includes visible reasoning (CoT, planning, “let me think…” style text), judge models often:Overweight **the fact that there is reasoning** (length, structure, “logical-sounding” steps) and give it higher scores even if the final answer is worse or partially wrong.Parrot or restate parts of that reasoning in their justification, blurring the line between “evaluation” and “adoption” of the candidate’s internal logic.Prefer responses that mimic popular reasoning tropes (“wait, let me think step by step…”) – OpenReview work calls this a kind of “superficial reflection bias,” where reasoning *style* itself becomes a strong signal. | Relevance: this concept doesn’t state/is not linked to: formatting echo that judge-llm exhibits.  |  |
|  |  |  |  |

---

---

### Target Workshops

Workshops first before conferences? All the big ones (NeuroIPS, etc. deadlines for 2026 have passed.. )

- need more research in-depth for conferences:
-

| Workshop | Deadline (2026) | Notes | Workshop Date |
| --- | --- | --- | --- |
| [StyleGenAI](https://list.elra.info/mailman3/hyperkitty/list/corpora@list.elra.info/thread/E4F5BHU3ZA4GVMHERAQKIRTUS6NLCM44/) | 27 April 2026 |  |  |
| [**Eval4SD**](https://eval4sd.github.io/cfp/) | 3 July 2026 |  |  |
| [Generation, Evaluation & Metrics (GEM) Workshop](https://gem-workshop.com/2025/) | 19 March | Acceptance Rate ~76%
**1. Automatic evaluation of generation systems, including the use of LMs as evaluatorsI want to go since it overlaps with Comic Con in San Diego lol** | • **July 3–4, 2026:** Workshop at **ACL** in San Diego |

### Concept Doc:

- Phenomenon: **format-induced reasoning echo in LLM-as-judge.**
- Hypothesis: more structured candidate answers induce more structured judge CoT, increasing strictness and lowering scores; casual formats induce looser CoT and higher scores.
- Target venues (e.g., NeurIPS D&B / ML Eval, ACL Findings, LM evaluation workshops).
- Define precise research questions:
    - RQ1: Does candidate formatting causally change judge scoring, holding semantics fixed?
    - RQ2: How does the answer leak/contaminate the judge’s chain of thought and reasoning?
    - RQ3: Is stricter CoT structure associated with lower scores, conditional on quality?
    - RQ4: Does this phenomenon occur only in binary output (T/F, Yes/No, 0/1) or also to gradient output? (a score from 0 to 1)
    - when designing pairwise evaluation with LLM-as-judge, how would the judge's reasonign be affected by the response type in its context window? how do we isolate that?
- Decide scope: which model families (e.g., 1 proprietary, 1 open), which tasks (QA, code, reasoning, subjective helpfulness).

### 1. Known: Priming Affects *Generation*

| What's Established | Source/Context |
| --- | --- |
| **Style Mirroring** | If you prompt with formal text, the LLM responds formally. |
| **Prompting Drift** | Different prompt structures lead to different reasoning modes (e.g., CoT vs. direct answer). |
| **Formatting Bias (Halo Effect)** | Most prior work assumes "better formatting → **higher** scores" (e.g., Zheng et al., 2024; verbosity bias). |

when designing pairwise evaluation with LLM-as-judge, how would the judge's reasonign be affected by the response type in its context window? how do we isolate that? 

### 2. Unknown: Priming Affects *Judicial Rigor* (The StyleJudge Gap)

### **A. The Directionality is Counter-Intuitive**

- **Common Assumption:** Structured responses look "higher quality," so judges should score them **higher** (Halo Effect).
- **StyleJudge Hypothesis:** Structured responses trigger **stricter reasoning**, so judges score them **lower** (Scrutiny Effect).
- **Why this matters:** This contradicts the standard "verbosity/formatting bias" literature. If true, it means structured models (like Claude) are systematically **penalized** in benchmarks, not rewarded. This specific asymmetry is **under-explored** (see *Novelty Claim* section).

### **B. The Mechanism is Hidden (Reasoning Traces)**

- **Prior Work:** Most bias papers (Zheng et al., Wataoka et al.) measure **output scores** (e.g., "Model A got 8.5, Model B got 9.0").
- **StyleJudge Contribution:** You are measuring the **judge's Chain-of-Thought (CoT)**.
    - *Question:* Does the candidate's structure change the **depth**, **structure**, and **error-detection rate** of the judge's reasoning?
    - *Gap:* "No prior work isolates structural formality as an independent variable affecting **judicial reasoning depth** (not just score magnitude)." (See *Background & Related Work 2.3*)

### **C. The Consequence is Measurement Validity**

- **Prior Work:** Treats bias as a "nuisance" to be controlled.
- **StyleJudge Contribution:** Frames this as a **construct validity crisis**.
    - If a benchmark measures "how well the model matches the judge's priming style" rather than "how correct the answer is," the benchmark is **invalid**.
    - *Philip's Framing:* "A good evaluator should be content-invariant with respect to style... StyleBias violates this basic criterion of measurement validity." (See *Section 3.3*)

### 3. Has Anyone Correlated This to Evaluation?

**Partially, but not deeply.**

- **Zheng et al. (2024)** (MT-Bench): Noted position and verbosity bias, but didn't isolate **structural formality** as a driver of **reasoning mode**.
- **Ye et al. (2024)** ("Justice or Prejudice?"): Catalogued 12 bias types (sentiment, bandwagon), but **style-induced rigor** is not among them.
- **Practitioner Guides (Arize, Wandb):** Warn that "judges may favor certain writing styles," but treat it as a **preference bias** (liking style X more) rather than a **process bias** (thinking differently because of style X).

### 4. Why This Still Warrants a Paper

The novelty isn't discovering priming; it's discovering **priming's impact on benchmark integrity**.

| Research Question | Status |
| --- | --- |
| "Do LLMs mirror input style?" | ✅ **Known** (Basic LLM behavior) |
| "Do judges prefer formatted text?" | ✅ **Known** (Formatting bias) |
| **"Does formatted text make judges *stricter* via reasoning modulation?"** | ❌ **Unknown** (StyleJudge Contribution) |
| **"Can models game evaluations by writing casually?"** | ❌ **Unknown** (Adversarial Exploit) |
| **"Does this invalidate cross-model benchmarks (Claude vs. GPT)?"** | ❌ **Unknown** (Benchmark Reliability) |

### Summary

You are right that the **mechanism** (priming) is old news. But the **implication** (that priming systematically miscalibrates evaluation scores in a counter-intuitive direction) is **not**.

Your paper's value proposition is:

> *"We all know priming exists. But we didn't realize it was **breaking our benchmarks** by making judges stricter on structured models. Here's the data, here's the mechanism (reasoning depth), and here's how to fix it."*
> 

This shifts the contribution from "LLM Behavior" (crowded field) to **"Evaluation Methodology & Benchmark Reliability"** (high impact, especially for NIW/industry).