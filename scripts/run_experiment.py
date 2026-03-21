"""
Entry point for the StyleJudge experiment.
Usage:
  python scripts/run_experiment.py                  # full experiment
  python scripts/run_experiment.py --limit 2        # dry-run on 2 base prompts
  python scripts/run_experiment.py --phase dataset  # run only dataset phase
  python scripts/run_experiment.py --set-osf URL    # set OSF pre-registration URL
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.agents.orchestrator import load_cfg, run
from src.utils.state import ExperimentState
from src.utils import logger as log


def parse_args():
    parser = argparse.ArgumentParser(description="StyleJudge Experiment Runner")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N base prompts (for dry-run)")
    parser.add_argument("--set-osf", metavar="URL",
                        help="Set OSF pre-registration URL in state and exit")
    parser.add_argument("--skip-smoke-test", action="store_true",
                        help="Skip API smoke test (use only if already validated)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    state = ExperimentState.from_file(cfg["paths"]["state"])

    if args.set_osf:
        state.set_osf_url(args.set_osf)
        print(f"OSF URL set: {args.set_osf}")
        return

    if not args.skip_smoke_test:
        log.info("Runner", "Running API smoke test...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/smoke_test.py"],
            capture_output=False,
        )
        if result.returncode != 0:
            print("Smoke test failed. Fix API issues before running experiment.")
            sys.exit(1)

    run(cfg, state, limit=args.limit)


if __name__ == "__main__":
    main()
