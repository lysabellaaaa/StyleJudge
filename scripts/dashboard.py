"""
StyleJudge Evaluation Dashboard

Run with:
    py -3 -m streamlit run scripts/dashboard.py
"""
import json
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).parent.parent

# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_json(path):
    p = ROOT / path
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data
def load_all_data():
    variants_art  = {v["variant_id"]: v for v in load_json("data/dataset/style_variants.json")}
    variants_nat  = {v["variant_id"]: v for v in load_json("data/dataset/natural_variants.json")}
    base_responses = {r["prompt_id"]: r for r in load_json("data/dataset/base_responses.json")}

    judges = ["claude", "gpt4o", "llama70b"]
    modes  = ["artificial", "natural", "adversarial"]
    scores = {}
    for judge in judges:
        for mode in modes:
            path = f"results/evaluations/{judge}/{mode}_scores.json"
            data = load_json(path)
            if data:
                scores[(judge, mode)] = {s["instance_id"]: s for s in data}

    pairwise = {}
    for judge in ["claude", "gpt4o"]:
        for mode in ["artificial", "natural"]:
            path = f"results/pairwise/{judge}_{mode}.json"
            data = load_json(path)
            if data:
                pairwise[(judge, mode)] = {p["question_id"]: p for p in data}

    mitigation = {}
    for cond in ["format_agnostic", "style_norm", "fixed_rubric"]:
        path = f"results/mitigation/{cond}_scores.json"
        data = load_json(path)
        if data:
            mitigation[cond] = {s["instance_id"]: s for s in data}

    prompt_templates = {}
    for f in (ROOT / "config/prompts").glob("*.txt"):
        prompt_templates[f.name] = f.read_text(encoding="utf-8")

    return variants_art, variants_nat, base_responses, scores, pairwise, mitigation, prompt_templates


def get_base_ids(scores, judge, mode):
    key = (judge, mode)
    if key not in scores:
        return []
    ids = set()
    for iid in scores[key]:
        bid = (iid.replace("eval_", "")
                  .replace("_V-simple", "").replace("_V-abstract", "")
                  .replace("_V-natural-simple", "").replace("_V-natural-abstract", ""))
        ids.add(bid)
    return sorted(ids)


def score_label(s):
    if s is None:
        return "—"
    return f"{s:.1f}"


def delta_color(simple, abstract):
    if simple is None or abstract is None:
        return "gray"
    d = abstract - simple
    if d > 0.1:
        return "green"
    elif d < -0.1:
        return "red"
    return "gray"


def get_agg(score_record):
    if score_record is None:
        return None
    return score_record.get("aggregate_score") or score_record.get("score")


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="StyleJudge Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("StyleJudge — Evaluation Inspector")

variants_art, variants_nat, base_responses, scores, pairwise, mitigation, prompt_templates = load_all_data()

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters")

    judge = st.selectbox("Judge", ["claude", "gpt4o", "llama70b"])
    mode  = st.selectbox("Mode",  ["artificial", "natural", "adversarial"])
    stream_filter = st.selectbox("Stream", ["all", "factual", "non_factual"])

    st.divider()

    base_ids = get_base_ids(scores, judge, mode)

    # build display list with scores
    display_items = []
    for bid in base_ids:
        suffix_s = "V-simple"   if mode == "artificial" else "V-natural-simple"
        suffix_a = "V-abstract" if mode == "artificial" else "V-natural-abstract"
        vars_dict = variants_art if mode == "artificial" else variants_nat
        vs = vars_dict.get(f"{bid}_{suffix_s}")
        va = vars_dict.get(f"{bid}_{suffix_a}")
        this_stream = (vs or va or {}).get("stream", "?")
        if stream_filter != "all" and this_stream != stream_filter:
            continue
        ss = scores.get((judge, mode), {}).get(f"eval_{bid}_{suffix_s}")
        sa = scores.get((judge, mode), {}).get(f"eval_{bid}_{suffix_a}")
        agg_s = get_agg(ss)
        agg_a = get_agg(sa)
        delta = round(agg_a - agg_s, 2) if agg_s is not None and agg_a is not None else None
        delta_str = f"{delta:+.1f}" if delta is not None else "—"
        label = f"{bid}  [{score_label(agg_s)} vs {score_label(agg_a)}  Δ{delta_str}]"
        display_items.append((bid, label, this_stream))

    if not display_items:
        st.warning("No scored items found for this combination.")
        selected_id = None
    else:
        labels     = [x[1] for x in display_items]
        base_id_list = [x[0] for x in display_items]
        idx = st.selectbox("Question", range(len(labels)), format_func=lambda i: labels[i])
        selected_id = base_id_list[idx]

    st.divider()
    show_base     = st.checkbox("Show base response", value=True)
    show_prompt   = st.checkbox("Show prompt template", value=False)
    show_cot      = st.checkbox("Show full CoT trace", value=False)

# ── main panel ────────────────────────────────────────────────────────────────

if not selected_id:
    st.info("Select a judge, mode and question from the sidebar.")
    st.stop()

bid = selected_id
suffix_s = "V-simple"   if mode == "artificial" else "V-natural-simple"
suffix_a = "V-abstract" if mode == "artificial" else "V-natural-abstract"
vars_dict = variants_art if mode == "artificial" else variants_nat

vs = vars_dict.get(f"{bid}_{suffix_s}")
va = vars_dict.get(f"{bid}_{suffix_a}")
base = base_responses.get(bid)
ss = scores.get((judge, mode), {}).get(f"eval_{bid}_{suffix_s}")
sa = scores.get((judge, mode), {}).get(f"eval_{bid}_{suffix_a}")
agg_s = get_agg(ss)
agg_a = get_agg(sa)
this_stream = (vs or va or {}).get("stream", "?")

# Header
st.subheader(f"{bid}  |  stream: {this_stream}  |  judge: {judge}  |  mode: {mode}")

q_text = (vs or va or {}).get("question_text", "")
if not q_text and base:
    q_text = base.get("prompt_text", "") or base.get("question_text", "")
if not q_text:
    bp = load_json("data/raw/base_prompts.json")
    bp_map = {r["question_id"]: r for r in bp}
    q_text = bp_map.get(bid, {}).get("question_text", "")

st.markdown("**Question:**")
st.info(q_text if q_text else "(question text not found)")

if show_base and base:
    with st.expander("Base response (DeepSeek-V3, before rewriting)", expanded=False):
        st.markdown(base.get("response_text", ""))

st.divider()

# ── Side-by-side variants ─────────────────────────────────────────────────────
col_s, col_a = st.columns(2)

with col_s:
    dc = delta_color(agg_s, agg_a)
    st.markdown(f"### {suffix_s}  —  score: **{score_label(agg_s)}**")
    if vs:
        st.markdown(vs["response_text"])
    else:
        st.warning("Variant not found.")

    if ss:
        st.markdown("**Judge reasoning:**")
        for criterion, detail in ss.get("criteria_scores", {}).items():
            with st.expander(f"{criterion}  —  {detail['score']}"):
                st.write(detail.get("reasoning", ""))
        if show_cot and ss.get("cot_trace"):
            with st.expander("Full CoT trace"):
                st.code(ss["cot_trace"], language="json")

with col_a:
    st.markdown(f"### {suffix_a}  —  score: **{score_label(agg_a)}**")
    if va:
        st.markdown(va["response_text"])
    else:
        st.warning("Variant not found.")

    if sa:
        st.markdown("**Judge reasoning:**")
        for criterion, detail in sa.get("criteria_scores", {}).items():
            with st.expander(f"{criterion}  —  {detail['score']}"):
                st.write(detail.get("reasoning", ""))
        if show_cot and sa.get("cot_trace"):
            with st.expander("Full CoT trace"):
                st.code(sa["cot_trace"], language="json")

# ── Score delta banner ────────────────────────────────────────────────────────
if agg_s is not None and agg_a is not None:
    delta = agg_a - agg_s
    if delta > 0.1:
        st.success(f"V-abstract scored **{delta:+.2f}** higher (Halo Effect direction)")
    elif delta < -0.1:
        st.error(f"V-abstract scored **{delta:+.2f}** lower (StyleBias confirmed direction)")
    else:
        st.info(f"Scores tied (delta = {delta:+.2f})")

# ── Pairwise result ───────────────────────────────────────────────────────────
if mode in ["artificial", "natural"]:
    pw_key = (judge, mode)
    if pw_key in pairwise and bid in pairwise[pw_key]:
        pw = pairwise[pw_key][bid]
        st.divider()
        st.markdown("**Pairwise result (blind A/B comparison):**")
        preferred_label   = pw.get("preferred_label", "?")
        preferred_variant = pw.get("preferred_variant", "?")
        strength          = pw.get("preference_strength", "?")
        st.markdown(f"- Preferred: `{preferred_label}` → `{preferred_variant}`  (strength: {strength})")
        if pw.get("primary_reason"):
            with st.expander("Pairwise reasoning"):
                st.write(pw["primary_reason"])
                if pw.get("a_strengths"):
                    st.markdown(f"**A strengths:** {pw['a_strengths']}")
                if pw.get("b_strengths"):
                    st.markdown(f"**B strengths:** {pw['b_strengths']}")

# ── Mitigation ────────────────────────────────────────────────────────────────
if mode == "artificial" and judge == "claude":
    mit_data = {}
    for cond, cond_scores in mitigation.items():
        s_rec = cond_scores.get(f"eval_{bid}_{suffix_s}")
        a_rec = cond_scores.get(f"eval_{bid}_{suffix_a}")
        mit_data[cond] = (get_agg(s_rec), get_agg(a_rec))

    if any(v[0] is not None or v[1] is not None for v in mit_data.values()):
        st.divider()
        st.markdown("**Mitigation conditions (Claude only):**")
        cols = st.columns(3)
        for i, (cond, (ms, ma)) in enumerate(mit_data.items()):
            with cols[i]:
                delta_m = round(ma - ms, 2) if ms is not None and ma is not None else None
                st.metric(
                    label=cond,
                    value=f"Δ {delta_m:+.2f}" if delta_m is not None else "—",
                    delta=f"simple={score_label(ms)}  abstract={score_label(ma)}",
                )

# ── Prompt template ───────────────────────────────────────────────────────────
if show_prompt:
    st.divider()
    tmpl_name = "judge_rubric_factual.txt" if this_stream == "factual" else "judge_rubric_nonfactual.txt"
    tmpl = prompt_templates.get(tmpl_name, "(not found)")
    with st.expander(f"Prompt template: {tmpl_name}", expanded=True):
        st.code(tmpl, language="text")

# ── All questions overview table ──────────────────────────────────────────────
st.divider()
with st.expander("All questions — score overview", expanded=False):
    rows = []
    for b, label, s in display_items:
        sfx_s = "V-simple"   if mode == "artificial" else "V-natural-simple"
        sfx_a = "V-abstract" if mode == "artificial" else "V-natural-abstract"
        sc_s = get_agg(scores.get((judge, mode), {}).get(f"eval_{b}_{sfx_s}"))
        sc_a = get_agg(scores.get((judge, mode), {}).get(f"eval_{b}_{sfx_a}"))
        d = round(sc_a - sc_s, 2) if sc_s is not None and sc_a is not None else None
        rows.append({
            "id": b,
            "stream": s,
            "V-simple": sc_s,
            "V-abstract": sc_a,
            "delta (A-S)": d,
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=400)
