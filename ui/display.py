"""
ui/display.py
=============
Shared output helpers used by every section.

Replaces the inline stubs we injected per cell in Colab.
Import from here — never redefine these elsewhere.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

# ── Width constants ────────────────────────────────────────────────────────────
WIDE  = 62
INNER = 50


# =============================================================================
# Core formatting helpers
# =============================================================================

def print_header(title: str, char: str = "═", width: int = WIDE) -> None:
    bar = char * width
    print(f"\n{bar}\n  {title}\n{bar}")


def print_subheader(title: str, char: str = "─", width: int = INNER) -> None:
    bar = char * width
    print(f"\n{bar}\n  {title}\n{bar}")


def print_check(label: str, value: Any, width: int = 32) -> None:
    print(f"  ✓ {label:<{width}} {value}")


def print_rule_fired(rule_name: str, conclusion: str,
                     confidence: float) -> None:
    print(f"  → [{confidence:.2f}]  {rule_name:<34}  ⟹  {conclusion}")


def separator(char: str = "─", width: int = INNER) -> None:
    print("  " + char * width)


# =============================================================================
# Profile display
# =============================================================================

def display_profile(profile: Dict[str, Any]) -> None:
    """Print a clean formatted summary of a student profile."""
    AVL = {
        1: "Low  (≥5 h/day free)",
        2: "Medium  (2–4 h/day free)",
        3: "High  (<2 h/day free)",
    }
    g = profile.get("gender")

    print_header(f"Student Profile — {profile.get('name', '?')}")

    print_subheader("Identity")
    print(f"  Name              : {profile.get('name', '?')}")
    print(f"  Gender            : {'Male' if g == 0 else 'Female' if g == 1 else '?'}")

    print_subheader("Academic metrics")
    print(f"  Attendance        : {profile.get('attendance', '?')}%")
    print(f"  Confidence        : {profile.get('confidence_level', '?')} / 5")
    print(f"  Stress            : {profile.get('stress_level', '?')} / 5")
    print(f"  Quiz score avg    : {profile.get('quiz_score_avg', '?')}%")
    print(f"  Credits this term : {profile.get('workload_credits', '?')}")

    print_subheader("Tasks")
    tasks = profile.get("tasks", [])
    if tasks:
        print(f"  {'Deadline':>7}  {'Task':<32} {'Hours':>6}  Subject")
        print("  " + "─" * 60)
        for t in sorted(tasks, key=lambda x: x.get("deadline_days", 99)):
            if t.get("overdue"):
                flag = "  !! OVERDUE"
            elif t.get("deadline_days", 99) <= 3:
                flag = "  !! URGENT"
            else:
                flag = ""
            print(f"  {t.get('deadline_days', '?'):>6}d  "
                  f"{t.get('name', '?'):<32}"
                  f"{t.get('hours', 0):>5.1f}h  "
                  f"{t.get('subject', '?')}{flag}")
    else:
        print("  (none)")

    print_subheader("Exams")
    exams = profile.get("exams", [])
    if exams:
        for e in sorted(exams, key=lambda x: x.get("days_until", 99)):
            print(f"  {e.get('days_until', '?'):>6}d  {e.get('subject', '?')}")
    else:
        print("  (none)")

    print_subheader("Availability")
    print(f"  Available days    : "
          f"{', '.join(profile.get('available_days', []))}")
    print(f"  Free hrs / day    : {profile.get('daily_free_hours', '?')}")
    print(f"  Constraint level  : "
          f"{AVL.get(profile.get('availability_constraints'), '?')}")

    print_subheader("Derived  (auto-computed)")
    print(f"  workload_tasks    : {profile.get('workload_tasks', '?')}")
    print(f"  workload_hours    : {profile.get('workload_hours', '?')} h total")
    print(f"  deadlines ≤ 7 d   : {profile.get('deadlines', '?')}")
    print()


# =============================================================================
# Schedule display
# =============================================================================

def display_schedule(result: Dict[str, Any],
                     strategy: str = "Schedule") -> None:
    """Print a formatted weekly timetable from a planner result dict."""
    schedule = result.get("schedule", [])
    day_order = ["Monday", "Tuesday", "Wednesday",
                 "Thursday", "Friday", "Saturday", "Sunday"]

    day_map: Dict[str, List] = {d: [] for d in day_order}
    for task_name, day, hour in schedule:
        if day in day_map:
            day_map[day].append((hour, task_name))

    print_subheader(f"{strategy} — weekly view")
    any_day = False
    for day in day_order:
        entries = sorted(day_map[day])
        if not entries:
            continue
        any_day = True
        print(f"\n  {day}")
        for hour, task_name in entries:
            print(f"    {hour:02d}:00 – {hour + 1:02d}:00   {task_name}")

    if not any_day:
        print("  (no slots assigned)")

    unsch = result.get("unscheduled", [])
    print(f"\n  Slots used      : {len(schedule)}")
    print(f"  Goal reached    : {result.get('found_goal', '?')}")
    print(f"  Conflicts       : {result.get('conflicts', '?')}")
    print(f"  Score           : {result.get('score', '?')}")
    if unsch:
        print(f"  Unscheduled     : {', '.join(unsch)}")
    print()


# =============================================================================
# Risk report display
# =============================================================================

def display_risk_report(report: Dict[str, Any]) -> None:
    """
    Print the full output of the pipeline for one student:
      - risk level + confidence
      - explanation (rules fired)
      - recommendations
    """
    RISK_COLOUR = {"Low": "✅", "Medium": "⚠️ ", "High": "🚨"}

    risk    = report.get("risk_level", "Unknown")
    conf    = report.get("confidence", 0.0)
    name    = report.get("name", "Student")
    icon    = RISK_COLOUR.get(risk, "")

    print_header(f"Risk Report — {name}")
    print(f"\n  {icon}  Risk level   : {risk}  (confidence {conf:.0%})")

    print_subheader("Why this risk level")
    signals = report.get("signals", [])
    if signals:
        for s in signals:
            print(f"  • {s}")
    else:
        print("  No risk signals detected.")

    print_subheader("Rules fired  (forward chaining)")
    rules_fired = report.get("rules_fired", [])
    if rules_fired:
        for rule_name, conclusion, confidence in rules_fired:
            print_rule_fired(rule_name, conclusion, confidence)
    else:
        print("  (none)")

    print_subheader("Recommendations")
    recs = report.get("recommendations", [])
    if recs:
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r}")
    else:
        print("  No recommendations — keep up the good work!")

    print_subheader("Explanation")
    explanation = report.get("explanation", "")
    if explanation:
        # Word-wrap at 58 chars
        words = explanation.split()
        line, lines = [], []
        for w in words:
            if sum(len(x) + 1 for x in line) + len(w) > 58:
                lines.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append(" ".join(line))
        for ln in lines:
            print(f"  {ln}")
    print()
