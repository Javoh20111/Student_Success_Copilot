"""
main.py
=======
Student Success Copilot — non-interactive entry point.

Trains ML models once, then runs Demo A and Demo B end-to-end.
No user input required — use this for submission output and testing.

Usage
-----
    python main.py               # run both demos
    python main.py --scenario A  # run only Demo A
    python main.py --scenario B  # run only Demo B
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback
from pathlib import Path


def _ensure_project_python() -> None:
    """
    Re-exec with the local .venv interpreter when the current Python
    environment is missing required ML dependencies.
    """
    if importlib.util.find_spec("sklearn") is not None:
        return

    project_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
    if not project_python.exists():
        return

    current_python = Path(sys.executable).resolve()
    if current_python == project_python.resolve():
        return

    os.execv(str(project_python), [str(project_python), *sys.argv])


_ensure_project_python()

from ui.display  import print_header
from ui.pipeline import train_once, run_demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Student Success Copilot — demo runner"
    )
    parser.add_argument(
        "--scenario", choices=["A", "B", "C"],
        help="Run only one scenario (default: all three)",
    )
    args = parser.parse_args()

    print_header("Student Success Copilot", char="█")
    print("  Explainable Hybrid AI — Plans, Predicts, Explains\n")

    models = train_once()

    scenarios = [args.scenario] if args.scenario else ["A", "B", "C"]

    for s in scenarios:
        try:
            run_demo(s, models=models)
        except Exception as exc:
            print(f"\n  [ERROR] Scenario {s} failed: {exc}")
            traceback.print_exc()
            sys.exit(1)

    print_header("All scenarios complete", char="█")


if __name__ == "__main__":
    main()
