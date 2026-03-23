"""
ui/pipeline.py
==============
Integration pipeline — wires all components together.

Flow
----
  1. Student profile  (ui/input_layer.py)
  2. Search planner   (planner/planner.py)   BFS + A* + comparison
  3. Rule engine      (rules/engine.py)       forward chain → risk report
  4. ML prediction    (ml/model.py)           Decision Tree + Naive Bayes
  5. Final output     (ui/display.py)         combined risk report

Both main.py and app.py call run_pipeline() — no logic is duplicated.

ML models are trained once per session and passed in to avoid retraining
on every scenario. Call train_once() at startup.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ui.display import (
    print_header, print_subheader, print_check,
    display_profile, display_schedule, display_risk_report,
)
from ui.input_layer import load_demo_profile, collect_student_inputs
from planner.planner import compare_planners, display_schedule as planner_display_schedule
from rules.engine import forward_chain
from ml.dataset import load_dataset, generate_dataset



# =============================================================================
# 1.  ML model cache  (train once per session)
# =============================================================================

_MODEL_CACHE: Optional[Dict[str, Any]] = None


def train_once(force: bool = False) -> Dict[str, Any]:
    global _MODEL_CACHE
    if _MODEL_CACHE is None or force:
        from ml.model import train_models   # ← moved inside
        ...
    """
    Train ML models once and cache them for the session.
    Always generates a fresh 1000-row dataset to avoid stale CSVs
    (e.g. from test runs that write smaller datasets).
    Subsequent calls return the cached models unless force=True.
    """

def train_once(force: bool = False) -> Dict[str, Any]:
    global _MODEL_CACHE
    if _MODEL_CACHE is None or force:
        from ml.model import train_models   # ← moved inside
        ...
        print_subheader("Training ML models  (one-time setup)")
        df = generate_dataset(n=1000)   # always 1000 rows — never read stale CSV
        _MODEL_CACHE = train_models(df)
        print(f"  Models trained on {len(df)} rows.")
        print(f"  Ready: {list(_MODEL_CACHE.keys())}\n")
    return _MODEL_CACHE


# =============================================================================
# 2.  Core pipeline
# =============================================================================
def run_pipeline(profile, models=None, verbose=True):
    if models is None:
        models = train_once()
    from ml.model import predict_risk       # ← moved inside
    ...

    """
    Run all components on a single student profile.

    Parameters
    ----------
    profile : validated StudentProfile dict from input_layer
    models  : pre-trained ML model dict (from train_once()).
              If None, models are trained on the fly.
    verbose : print formatted output as each step completes

    Returns
    -------
    result dict with keys:
      profile, planner, report, ml_predictions
    """
    if models is None:
        models = train_once()

    result: Dict[str, Any] = {
        "profile":         profile,
        "planner":         None,
        "report":          None,
        "ml_predictions":  None,
    }

    # ── Step 1: Profile summary ───────────────────────────────────────────────
    if verbose:
        print_subheader("Step 1 — Student profile")
        display_profile(profile)

    # ── Step 2: Search planner ────────────────────────────────────────────────
    if verbose:
        print_subheader("Step 2 — Search planner  (BFS vs A*)")

    planner_result = compare_planners(profile)
    result["planner"] = planner_result

    if verbose:
        print_subheader("A* weekly schedule")
        planner_display_schedule(planner_result["astar"], strategy="A*")

    # ── Step 3: Rule-based expert system ─────────────────────────────────────
    if verbose:
        print_subheader("Step 3 — Rule-based expert system")

    report = forward_chain(profile, planner_result)
    result["report"] = report

    if verbose:
        display_risk_report(report)

    # ── Step 4: ML prediction ─────────────────────────────────────────────────
    if verbose:
        print_subheader("Step 4 — ML risk prediction")

    ml_preds = predict_risk(profile, models)
    result["ml_predictions"] = ml_preds

    if verbose:
        for model_name, pred in ml_preds.items():
            print(f"  {model_name:<38} "
                  f"→  {pred['prediction']:<6}  "
                  f"(confidence {pred['confidence']:.0%})")
        print()

    # ── Mixed signals check ───────────────────────────────────────────────────
    signal_check = _detect_mixed_signals(report, planner_result, ml_preds)
    result["signal_check"] = signal_check

    # Adjust rules confidence if conflict detected
    if signal_check["has_conflict"]:
        report["confidence"] = signal_check["adjusted_confidence"]

    # ── Step 5: Combined final output ─────────────────────────────────────────
    if verbose:
        _print_final_summary(profile, planner_result, report, ml_preds,
                             signal_check)

    return result


# =============================================================================
# 3.  Mixed signals detector
# =============================================================================

def _detect_mixed_signals(
    report:        Dict[str, Any],
    planner:       Dict[str, Any],
    ml_preds:      Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare ML verdict, rules verdict, and planner outcome.
    When they conflict, return a warning dict with explanation.

    Conflict cases
    --------------
    CASE 1  ML=Pass  +  Rules=High risk  → most dangerous mismatch
            Student looks good on paper (historical patterns) but current
            behaviour signals are alarming.  Rules win.

    CASE 2  ML=Pass  +  Planner failed   → schedule impossible despite
            good academic metrics.  Workload reality overrides statistics.

    CASE 3  ML=Fail  +  Rules=Low risk   → model may be picking up noise.
            Use Rules verdict but flag uncertainty.

    Returns a dict:
      { "has_conflict": bool, "severity": "high"|"medium"|"none",
        "ml_verdict": str, "rules_verdict": str, "planner_ok": bool,
        "message": str, "adjusted_confidence": float }
    """
    rules_risk  = report.get("risk_level", "Low")
    orig_conf   = report.get("confidence", 0.5)
    astar       = planner.get("astar", {})
    planner_ok  = astar.get("found_goal", True)

    pass_votes  = sum(
        1 for p in ml_preds.values() if p["prediction"] == "Pass"
    ) if ml_preds else 0
    total_votes = len(ml_preds) if ml_preds else 1
    ml_verdict  = "Pass" if pass_votes > total_votes / 2 else "Fail"

    # ── No conflict ───────────────────────────────────────────────────────────
    if ml_verdict == "Fail" and rules_risk == "High":
        return {"has_conflict": False, "severity": "none",
                "ml_verdict": ml_verdict, "rules_verdict": rules_risk,
                "planner_ok": planner_ok, "message": "",
                "adjusted_confidence": orig_conf}

    if ml_verdict == "Pass" and rules_risk == "Low" and planner_ok:
        return {"has_conflict": False, "severity": "none",
                "ml_verdict": ml_verdict, "rules_verdict": rules_risk,
                "planner_ok": planner_ok, "message": "",
                "adjusted_confidence": orig_conf}

    # ── High severity: ML says Pass but rules or planner say otherwise ────────
    if ml_verdict == "Pass" and (rules_risk == "High" or not planner_ok):
        reasons = []
        if rules_risk == "High":
            reasons.append(f"rules engine flags {rules_risk} risk "
                           f"({len(report.get('rules_fired', []))} rules fired)")
        if not planner_ok:
            ns = len(astar.get("not_scheduled", []))
            pt = len(astar.get("partial", []))
            reasons.append(f"planner cannot fit all tasks "
                           f"({pt} partial, {ns} not started)")
        adj = round(max(0.35, orig_conf - 0.25), 2)
        return {
            "has_conflict":        True,
            "severity":            "high",
            "ml_verdict":          ml_verdict,
            "rules_verdict":       rules_risk,
            "planner_ok":          planner_ok,
            "adjusted_confidence": adj,
            "message": (
                "⚠  Mixed signals detected\n"
                f"   ML predicts PASS  (based on historical patterns — "
                f"{pass_votes}/{total_votes} models agree)\n"
                + ("\n".join(f"   Rules flag: {r}" for r in reasons)) + "\n"
                "   Final verdict: AT RISK until schedule conflict is resolved.\n"
                "   The ML model was trained on students who had already\n"
                "   completed the term. It cannot see your current week's\n"
                "   overload — the rules engine can."
            ),
        }

    # ── Medium severity: ML says Fail but rules are Low ───────────────────────
    if ml_verdict == "Fail" and rules_risk == "Low":
        adj = round(max(0.40, orig_conf - 0.10), 2)
        return {
            "has_conflict":        True,
            "severity":            "medium",
            "ml_verdict":          ml_verdict,
            "rules_verdict":       rules_risk,
            "planner_ok":          planner_ok,
            "adjusted_confidence": adj,
            "message": (
                "⚠  Mild signal conflict\n"
                "   ML predicts FAIL but current behaviour signals are Low risk.\n"
                "   The ML model may be picking up noise from similar profiles.\n"
                "   Rules verdict (Low risk) is used — monitor closely."
            ),
        }

    # ── Default: no significant conflict ─────────────────────────────────────
    return {"has_conflict": False, "severity": "none",
            "ml_verdict": ml_verdict, "rules_verdict": rules_risk,
            "planner_ok": planner_ok, "message": "",
            "adjusted_confidence": orig_conf}

def _print_final_summary(
    profile:       Dict[str, Any],
    planner:       Dict[str, Any],
    report:        Dict[str, Any],
    ml_preds:      Dict[str, Any],
    signal_check:  Optional[Dict[str, Any]] = None,
) -> None:
    """
    Print a single-screen summary of all outputs for the student.
    """
    name      = profile.get("name", "Student")
    risk      = report.get("risk_level", "?")
    conf      = report.get("confidence", 0.0)
    astar     = planner.get("astar", {})
    goal      = astar.get("found_goal", False)
    completed = astar.get("completed", [])
    partial   = astar.get("partial", [])
    not_sched = astar.get("not_scheduled", [])

    ICONS = {"Low": "✅", "Medium": "⚠️ ", "High": "🚨"}

    print_header(f"Copilot Summary — {name}", char="▓")

    # ── Mixed signals warning (printed first so it's impossible to miss) ──────
    if signal_check and signal_check.get("has_conflict"):
        print()
        for line in signal_check["message"].split("\n"):
            print(f"  {line}")
        print()

    # Risk verdict (confidence may have been adjusted down)
    conf_note = " (adjusted — see conflict above)" \
        if signal_check and signal_check.get("has_conflict") else ""
    print(f"  {ICONS.get(risk, '')}  Risk level     : {risk}  "
          f"(confidence {conf:.0%}{conf_note})")

    # Schedule verdict
    if goal:
        print("  ✅  Schedule       : Complete — all tasks fit within your week")
    else:
        parts = []
        if completed:
            parts.append(f"{len(completed)} completed")
        if partial:
            parts.append(f"{len(partial)} partial")
        if not_sched:
            parts.append(f"{len(not_sched)} not started")
        print(f"  ⚠️   Schedule       : Incomplete — {', '.join(parts)}")

    # ML verdict
    if ml_preds:
        pass_votes  = sum(1 for p in ml_preds.values()
                         if p["prediction"] == "Pass")
        total_votes = len(ml_preds)
        ml_verdict  = "Pass" if pass_votes > total_votes / 2 else "Fail"
        conflict_flag = " ⚠  (conflicts with rules)" \
            if signal_check and signal_check.get("has_conflict") and \
               ml_verdict != ("Fail" if risk == "High" else "Pass") else ""
        print(f"  {'✅' if ml_verdict == 'Pass' else '🚨'}  ML prediction  : "
              f"{ml_verdict}  "
              f"({pass_votes}/{total_votes} models predict Pass)"
              f"{conflict_flag}")

    # Top signals
    signals = report.get("signals", [])
    if signals:
        print_subheader("Key risk signals")
        for s in signals[:5]:
            print(f"  {s}")
        if len(signals) > 5:
            print(f"  ... and {len(signals)-5} more signal(s)")

    # Recommendations
    recs = report.get("recommendations", [])
    if recs:
        print_subheader("Recommendations")
        for i, r in enumerate(recs[:4], 1):
            print(f"  {i}. {r}")
        if len(recs) > 4:
            print(f"  ... and {len(recs)-4} more recommendation(s)")
    else:
        print_subheader("Recommendations")
        print("  No urgent actions needed — keep up the current approach.")

    print()


# =============================================================================
# 4.  Demo scenario runners
# =============================================================================

def run_demo(scenario: str, models: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load a pre-built demo profile and run the full pipeline.
    scenario = "A" (low risk) or "B" (high risk).
    """
    labels = {
        "A": "Alex — on track  (low risk)",
        "B": "Jordan — at risk  (high risk / overloaded)",
        "C": "Sam — mixed signals  (ML vs rules conflict)",
    }
    print_header(f"Demo Scenario {scenario} — {labels.get(scenario, scenario)}")
    profile = load_demo_profile(scenario)
    return run_pipeline(profile, models=models)
