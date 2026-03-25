# Future Plan Notes

## Natural Mode (V-natural) — Correction

**Decision:** V-natural variants must use raw model output only — no system prompt imposing any structure or style constraints.

- **V-natural-simple**: DeepSeek-V3 (`deepseek-chat`) — question passed as user message, **no system prompt**
- **V-natural-abstract**: DeepSeek-R1 (`deepseek-reasoner`) — question passed as user message, **no system prompt**; `<think>` block stripped, final answer only stored

**Why:** The Natural mode is designed to capture each model's *native* output style. Adding a system prompt that instructs structure or plain prose defeats the purpose — it becomes another artificial rewrite, not a natural response. The format difference between V3 and R1 should emerge organically from the models' default behaviour.

**What to fix before the full run:**
- Remove `generate_natural_response_simple.txt` system prompt from `natural_generator.py` (pass `system_prompt=None` or empty string for V-natural-simple calls)
- Remove `generate_natural_response_abstract.txt` system prompt from V-natural-abstract calls
- Re-generate all natural variants (delete `data/dataset/natural_variants.json` and reset `natural_generation` state)
