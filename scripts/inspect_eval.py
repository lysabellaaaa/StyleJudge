"""
Inspect evaluation dialogues for any question.

Usage:
    py -3 scripts/inspect_eval.py --id fq_002
    py -3 scripts/inspect_eval.py --id fq_002 --judge gpt4o
    py -3 scripts/inspect_eval.py --id fq_002 --judge claude --mode natural
    py -3 scripts/inspect_eval.py --list               # list all scored question IDs
"""
import argparse
import json
import os
import textwrap
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv("config/api_keys.env")
ROOT = Path(__file__).parent.parent


def _load(path):
    p = ROOT / path
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _wrap(text, width=90, indent="    "):
    lines = text.splitlines()
    out = []
    for line in lines:
        if len(line) <= width:
            out.append(indent + line)
        else:
            for chunk in textwrap.wrap(line, width=width):
                out.append(indent + chunk)
    return "\n".join(out)


def _divider(char="-", width=90):
    return char * width


def _load_prompt_template(name):
    p = ROOT / "config" / "prompts" / name
    return p.read_text(encoding="utf-8") if p.exists() else "(template not found)"


def show_pair(base_id, judge_id="claude", mode="artificial"):
    # Load data
    cfg = yaml.safe_load((ROOT / "config/experiment.yaml").read_text(encoding="utf-8"))

    # Variants
    if mode == "natural":
        all_variants = _load("data/dataset/natural_variants.json")
        suffix_a, suffix_b = "V-natural-simple", "V-natural-abstract"
    else:
        all_variants = _load("data/dataset/style_variants.json")
        suffix_a, suffix_b = "V-simple", "V-abstract"

    variants = {v["variant_id"]: v for v in all_variants}
    va = variants.get(f"{base_id}_{suffix_a}")
    vb = variants.get(f"{base_id}_{suffix_b}")

    if not va and not vb:
        print(f"No variants found for {base_id} in mode={mode}")
        return

    # Base responses
    base_responses = {r["prompt_id"]: r for r in _load("data/dataset/base_responses.json")}
    base = base_responses.get(base_id)

    # Scores
    score_file = ROOT / f"results/evaluations/{judge_id}/{mode}_scores.json"
    if not score_file.exists():
        print(f"No score file: {score_file}")
        return
    all_scores = json.loads(score_file.read_text(encoding="utf-8"))
    scores = {s["instance_id"]: s for s in all_scores}
    sa = scores.get(f"eval_{base_id}_{suffix_a}")
    sb = scores.get(f"eval_{base_id}_{suffix_b}")

    # Prompt template used
    prompt_tmpl_name = "judge_rubric_factual.txt" if (va or vb or {}).get("stream") == "factual" else "judge_rubric_nonfactual.txt"
    stream = (va or vb or {}).get("stream", "?")
    if stream == "factual":
        prompt_tmpl_name = "judge_rubric_factual.txt"
    else:
        prompt_tmpl_name = "judge_rubric_nonfactual.txt"
    prompt_template = _load_prompt_template(prompt_tmpl_name)

    # ── Print ──────────────────────────────────────────────────────────────────
    print()
    print(_divider("="))
    print(f"  INSPECTION: {base_id}  |  judge={judge_id}  |  mode={mode}  |  stream={stream}")
    print(_divider("="))

    q_text = (va or vb or {}).get("question_text", base.get("question_text", "?") if base else "?")
    print(f"\nQUESTION:\n{_wrap(q_text)}\n")

    if base:
        print(_divider())
        print(f"  BASE RESPONSE (DeepSeek-V3, before rewriting):")
        print(_divider())
        print(_wrap(base.get("response_text", ""), width=88))

    # ── Variant A ──────────────────────────────────────────────────────────────
    print()
    print(_divider())
    score_a = sa.get("aggregate_score", "?") if sa else "NOT SCORED"
    print(f"  {suffix_a.upper()}  ->  aggregate score: {score_a}")
    print(_divider())
    if va:
        print(_wrap(va["response_text"], width=88))
    else:
        print("    (variant missing)")

    if sa:
        print(f"\n  Judge reasoning ({judge_id}):")
        for criterion, detail in sa.get("criteria_scores", {}).items():
            print(f"\n    [{criterion}]  score={detail['score']}")
            print(_wrap(detail.get("reasoning", ""), width=84, indent="      "))

    # ── Variant B ──────────────────────────────────────────────────────────────
    print()
    print(_divider())
    score_b = sb.get("aggregate_score", "?") if sb else "NOT SCORED"
    print(f"  {suffix_b.upper()}  ->  aggregate score: {score_b}")
    print(_divider())
    if vb:
        print(_wrap(vb["response_text"], width=88))
    else:
        print("    (variant missing)")

    if sb:
        print(f"\n  Judge reasoning ({judge_id}):")
        for criterion, detail in sb.get("criteria_scores", {}).items():
            print(f"\n    [{criterion}]  score={detail['score']}")
            print(_wrap(detail.get("reasoning", ""), width=84, indent="      "))

    # ── Prompt template ────────────────────────────────────────────────────────
    print()
    print(_divider())
    print(f"  PROMPT TEMPLATE SENT TO JUDGE  ({prompt_tmpl_name})")
    print(_divider())
    print(_wrap(prompt_template[:1200], width=88))
    if len(prompt_template) > 1200:
        print(f"    ... (truncated, full template at config/prompts/{prompt_tmpl_name})")

    print()
    print(_divider("="))


def list_ids(judge_id="claude", mode="artificial"):
    score_file = ROOT / f"results/evaluations/{judge_id}/{mode}_scores.json"
    if not score_file.exists():
        print(f"No score file found: {score_file}")
        return
    scores = json.loads(score_file.read_text(encoding="utf-8"))
    ids = sorted(set(
        s["instance_id"].replace("eval_", "").replace("_V-simple", "").replace("_V-abstract", "")
                        .replace("_V-natural-simple", "").replace("_V-natural-abstract", "")
        for s in scores
    ))
    print(f"\nScored question IDs for judge={judge_id}, mode={mode} ({len(ids)} questions):\n")
    for i, qid in enumerate(ids):
        sa = next((s for s in scores if f"_{qid}_V" in s["instance_id"] and "simple" in s["instance_id"]), None)
        sb = next((s for s in scores if f"_{qid}_V" in s["instance_id"] and "abstract" in s["instance_id"]), None)
        score_a = f"{sa['aggregate_score']:.1f}" if sa and 'aggregate_score' in sa else "?"
        score_b = f"{sb['aggregate_score']:.1f}" if sb and 'aggregate_score' in sb else "?"
        print(f"  {qid:<20}  simple={score_a}  abstract={score_b}")


def main():
    parser = argparse.ArgumentParser(description="Inspect StyleJudge evaluation dialogues")
    parser.add_argument("--id", help="Base question ID, e.g. fq_002")
    parser.add_argument("--judge", default="claude", choices=["claude", "gpt4o", "llama70b"])
    parser.add_argument("--mode", default="artificial", choices=["artificial", "natural", "adversarial"])
    parser.add_argument("--list", action="store_true", help="List all scored question IDs")
    args = parser.parse_args()

    if args.list:
        list_ids(args.judge, args.mode)
    elif args.id:
        show_pair(args.id, args.judge, args.mode)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
