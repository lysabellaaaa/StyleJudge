"""
Resume the StyleJudge experiment from the last checkpoint.
The experiment_state.json handles which items are complete; this script
simply re-invokes the orchestrator.

Usage:
  python scripts/resume_experiment.py
  python scripts/resume_experiment.py --show-progress
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.orchestrator import load_cfg, run
from src.utils.state import ExperimentState


def show_progress(state: ExperimentState) -> None:
    data = state.data
    print("\n=== Experiment Progress ===")
    print(f"Current phase: {data['current_phase']}")
    print(f"OSF URL: {data.get('osf_preregistration_url') or 'NOT SET'}")
    for phase, items in data["completed"].items():
        count = len(items)
        if count > 0:
            print(f"  {phase}: {count} items complete")
    if data["errors"]:
        print(f"\nErrors logged: {len(data['errors'])}")
        for e in data["errors"][-3:]:
            print(f"  [{e['phase']}] {e['item_id']}: {e['error'][:80]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Resume StyleJudge experiment")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    state = ExperimentState.from_file(cfg["paths"]["state"])

    if args.show_progress:
        show_progress(state)
        return

    show_progress(state)
    run(cfg, state)


if __name__ == "__main__":
    main()
