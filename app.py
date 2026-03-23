"""
app.py
======
Student Success Copilot — interactive menu-driven CLI.

Usage
-----
    python app.py
"""

from __future__ import annotations

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

from ui.display       import print_header, print_subheader, separator
from ui.pipeline      import train_once, run_demo, run_pipeline
from ui.input_layer   import collect_student_inputs
from ui.nlp_interface import nlp_collect_inputs

MENU = """
  ┌─────────────────────────────────────────────┐
  │       Student Success Copilot  v1.0          │
  ├─────────────────────────────────────────────┤
  │  Demos                                       │
  │  1 — Demo A  (Alex, low risk)                │
  │  2 — Demo B  (Jordan, high risk)             │
  │  3 — Demo C  (Sam, mixed signals)            │
  │  4 — Run all three demos                     │
  ├─────────────────────────────────────────────┤
  │  Input modes                                 │
  │  5 — Enter profile  (form / example)         │
  │  6 — Enter profile  (NLP chat interface)     │
  ├─────────────────────────────────────────────┤
  │  AI explainer  (requires ANTHROPIC_API_KEY)  │
  │  7 — Explain last result                     │
  │  8 — Ask a free-text question                │
  │  q — Quit                                    │
  └─────────────────────────────────────────────┘
"""


def _run_genai_explainer(result: dict) -> None:
    """Run all three GenAI explanations for the last pipeline result."""
    from ui.genai_explainer import (
        explain_study_tips, explain_search_comparison, explain_risk,
    )
    profile      = result["profile"]
    report       = result["report"]
    planner      = result["planner"]
    signal_check = result.get("signal_check")

    print_header("GenAI Explainer")

    print_subheader("1. Personalised study tips")
    print(explain_study_tips(profile, report, planner))

    print_subheader("2. Why A* vs BFS?")
    print(explain_search_comparison(profile, planner))

    print_subheader("3. Risk explanation")
    print(explain_risk(profile, report, signal_check))


def main() -> None:
    print_header("Student Success Copilot", char="█")
    print("  Explainable Hybrid AI — Plans, Predicts, Explains\n")

    models      = train_once()
    last_result = None   # holds the most recent pipeline result for GenAI

    while True:
        print(MENU)
        choice = input("  Choose an option: ").strip().lower()

        if choice == "q":
            print("\n  Goodbye!\n")
            sys.exit(0)

        elif choice == "1":
            last_result = run_demo("A", models=models)

        elif choice == "2":
            last_result = run_demo("B", models=models)

        elif choice == "3":
            last_result = run_demo("C", models=models)

        elif choice == "4":
            for s in ("A", "B", "C"):
                last_result = run_demo(s, models=models)

        elif choice == "5":
            print_header("Enter your own profile  (form mode)")
            try:
                profile     = collect_student_inputs(mode="interactive")
                last_result = run_pipeline(profile, models=models)
            except KeyboardInterrupt:
                print("\n\n  Input cancelled.")
            except Exception as exc:
                print(f"\n  [ERROR] {exc}")
                traceback.print_exc()

        elif choice == "6":
            print_header("Enter your own profile  (NLP chat)")
            try:
                profile     = nlp_collect_inputs()
                last_result = run_pipeline(profile, models=models)
            except KeyboardInterrupt:
                print("\n\n  Chat cancelled.")
            except Exception as exc:
                print(f"\n  [ERROR] {exc}")
                traceback.print_exc()

        elif choice == "7":
            if last_result is None:
                print("\n  No result yet — run a demo first.\n")
            else:
                _run_genai_explainer(last_result)

        elif choice == "8":
            if last_result is None:
                print("\n  No result yet — run a demo first.\n")
            else:
                from ui.genai_explainer import run_interactive_explainer
                run_interactive_explainer(
                    last_result["profile"],
                    last_result["report"],
                    last_result["planner"],
                    last_result.get("signal_check"),
                )

        else:
            print("  Invalid choice — please enter 1–8 or q.")

        separator()
        input("  Press ENTER to return to menu...")


if __name__ == "__main__":
    main()
