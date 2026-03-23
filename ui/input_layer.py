"""
ui/input_layer.py
=================
Collects, validates, and cleans student profile data.

Public API
----------
collect_student_inputs(mode, demo_data)  — main entry point
validate_inputs(profile)                 — returns list of Issue objects
ask_followup(profile, issues)            — targeted re-prompt per issue
compute_derived_fields(profile)          — fills workload_tasks etc.
load_demo_profile(scenario)              — "A" or "B" pre-built profile
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ui.display import print_header, print_subheader

DAYS_OF_WEEK = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday",
]


# =============================================================================
# Issue dataclass
# =============================================================================

@dataclass
class Issue:
    kind:     str   # "MISSING" | "OUT_OF_RANGE" | "CONTRADICTION"
    field:    str
    message:  str
    question: str
    severity: str   # "error" | "warning"


# =============================================================================
# 1.  Validation
# =============================================================================

def validate_inputs(profile: Dict[str, Any]) -> List[Issue]:
    """
    Check for missing fields, out-of-range values, and contradictions.
    Returns a (possibly empty) list of Issue objects.
    Errors must be resolved before the pipeline continues.
    Warnings are surfaced but not blocking.
    """
    issues: List[Issue] = []

    def missing(fld: str, q: str) -> None:
        issues.append(Issue("MISSING", fld,
                            f"Required field '{fld}' is missing.",
                            q, "error"))

    def oor(fld: str, val: Any, lo: Any, hi: Any, q: str) -> None:
        issues.append(Issue("OUT_OF_RANGE", fld,
                            f"'{fld}' = {val} is outside valid range [{lo}, {hi}].",
                            q, "error"))

    def contra(fld: str, msg: str, q: str, sev: str = "warning") -> None:
        issues.append(Issue("CONTRADICTION", fld, msg, q, sev))

    # ── Scalar required fields ────────────────────────────────────────────────
    if not str(profile.get("name", "")).strip():
        missing("name", "What is your first name?")

    g = profile.get("gender")
    if g is None:
        missing("gender", "Your gender — enter 0 = male, 1 = female:")
    elif g not in (0, 1):
        oor("gender", g, 0, 1, "Please enter 0 (male) or 1 (female).")

    att = profile.get("attendance")
    if att is None:
        missing("attendance",
                "Class attendance % this term (0–100):")
    elif not (0 <= att <= 100):
        oor("attendance", att, 0, 100,
            "Attendance must be 0–100. What is your actual %?")

    conf = profile.get("confidence_level")
    if conf is None:
        missing("confidence_level",
                "Confidence in keeping up with studies (1 = very low … 5 = very high):")
    elif conf not in range(1, 6):
        oor("confidence_level", conf, 1, 5,
            "Please rate confidence 1–5.")

    stress = profile.get("stress_level")
    if stress is None:
        missing("stress_level",
                "Current stress level (1 = minimal … 5 = extreme):")
    elif stress not in range(1, 6):
        oor("stress_level", stress, 1, 5,
            "Please rate stress 1–5.")

    quiz = profile.get("quiz_score_avg")
    if quiz is None:
        missing("quiz_score_avg",
                "Average quiz / test score this term (0–100):")
    elif not (0 <= quiz <= 100):
        oor("quiz_score_avg", quiz, 0, 100,
            "Quiz score must be 0–100.")

    credits = profile.get("workload_credits")
    if credits is None:
        missing("workload_credits",
                "Total credit hours this term (0, 20, 40, 60, or 80):")
    elif credits not in (0, 20, 40, 60, 80):
        oor("workload_credits", credits, 0, 80,
            "Credits must be 0, 20, 40, 60, or 80.")

    # ── List fields ───────────────────────────────────────────────────────────
    tasks = profile.get("tasks", [])
    if not tasks:
        missing("tasks",
                "List your pending tasks "
                "(name, subject, estimated hours, deadline in days).")

    for i, t in enumerate(tasks):
        if t.get("hours", 0) <= 0:
            oor(f"tasks[{i}].hours", t.get("hours"), 0.5, 40,
                f"Task '{t.get('name', '?')}' has invalid hours. "
                "How many hours do you need for it?")
        dl = t.get("deadline_days")
        if dl is not None and dl < 0:
            # Negative deadline = past due. We clamp to 0 (today) and mark
            # OVERDUE so the planner still schedules it immediately.
            t["deadline_days"] = 0
            t["overdue"]       = True
            contra(
                f"tasks[{i}].deadline_days",
                f"Task '{t.get('name', '?')}' has a deadline in the past "
                f"({dl} days). It has been marked OVERDUE.",
                f"Has '{t.get('name', '?')}' already been submitted? "
                "Enter 'yes' to remove it, a new deadline in days (e.g. 3) "
                "to correct it, or press ENTER to keep it as OVERDUE:",
                sev="warning",
            )

    av_days = profile.get("available_days", [])
    if not av_days:
        missing("available_days",
                "Which days are you free to study? "
                "(e.g. Monday, Wednesday, Friday)")

    dfh = profile.get("daily_free_hours")
    if dfh is None:
        missing("daily_free_hours",
                "Average free hours per available study day:")
    elif not (0 < dfh <= 16):
        oor("daily_free_hours", dfh, 0.5, 16,
            "Free hours/day should be 0.5–16. "
            "What is a realistic number?")

    # ── Contradiction checks ──────────────────────────────────────────────────
    if tasks and dfh is not None and av_days:
        total_free = dfh * len(av_days)
        total_work = sum(t.get("hours", 0) for t in tasks)
        if total_work > total_free * 1.5:
            contra(
                "tasks / available_days / daily_free_hours",
                f"You have {total_work:.1f} h of tasks but only "
                f"{total_free:.1f} h free "
                f"({dfh} h/day × {len(av_days)} days). "
                "This schedule is impossible as-is.",
                "Would you like to: (a) reduce hour estimates, "
                "(b) add more available days, or (c) push some deadlines? "
                "Type a, b, or c:",
                sev="error",
            )

    if tasks and stress is not None:
        urgent = [t for t in tasks if t.get("deadline_days", 99) <= 2]
        if urgent and stress <= 2:
            contra(
                "tasks / stress_level",
                f"{len(urgent)} task(s) due within 2 days "
                f"but stress = {stress}/5 (very low).",
                "You have urgent deadlines but very low stress — "
                "is that accurate? Update stress (1–5) or press ENTER to keep:",
            )

    if conf is not None and att is not None:
        if conf >= 4 and att < 50:
            contra(
                "confidence_level / attendance",
                f"High confidence ({conf}/5) with low attendance ({att}%) "
                "is an unusual combination.",
                f"You rated confidence {conf}/5 but attendance is {att}%. "
                "Is your confidence score still accurate? (1–5 or ENTER):",
            )

    if quiz is not None and att is not None:
        # Only run contradiction if quiz is within valid range.
        # If quiz is already OOR the contradiction is meaningless.
        if 0 <= quiz < 40 and att >= 80:
            contra(
                "quiz_score_avg / attendance",
                f"Good attendance ({att}%) but low quiz scores ({quiz}%) "
                "suggests an understanding gap.",
                "Is that quiz average accurate, or has it improved recently? "
                "Enter updated average (0–100) or ENTER to keep:",
            )

    return issues


# =============================================================================
# 2.  Derived fields
# =============================================================================

def compute_derived_fields(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach computed fields that downstream modules need.
    Called automatically at the end of collect_student_inputs.
    """
    tasks = profile.get("tasks", [])
    dfh   = profile.get("daily_free_hours", 0)

    profile["workload_tasks"] = len(tasks)
    profile["workload_hours"] = round(
        sum(t.get("hours", 0) for t in tasks), 1
    )
    profile["deadlines"] = sum(
        1 for t in tasks if 0 <= t.get("deadline_days", 99) <= 7
    )

    # availability_constraints:
    #   1 = low restriction (≥ 5 h/day free)
    #   2 = medium (2–4 h/day)
    #   3 = high (<2 h/day)
    if dfh >= 5:
        profile["availability_constraints"] = 1
    elif dfh >= 2:
        profile["availability_constraints"] = 2
    else:
        profile["availability_constraints"] = 3

    return profile


# =============================================================================
# 3.  Interactive helpers
# =============================================================================

def _safe_float(s: str, default: Any) -> Any:
    try:    return float(s)
    except: return default   # noqa: E722


def _safe_int(s: str, default: Any) -> Any:
    try:    return int(s)
    except: return default   # noqa: E722


def _prompt(question: str) -> str:
    print(f"\n  {question}")
    return input("  > ").strip()


def _collect_interactive() -> Dict[str, Any]:
    """Full interactive prompt session."""
    print_header("Student Success Copilot — Input Collection")
    print("  Answer each question. Leave blank to skip optional items.\n")
    p: Dict[str, Any] = {}

    p["name"]             = _prompt("Your first name:") or "Student"
    p["gender"]           = _safe_int(
        _prompt("Gender — 0 = male, 1 = female:"), None)

    print_subheader("Academic metrics")
    p["attendance"]       = _safe_float(
        _prompt("Class attendance % (0–100):"), None)
    p["confidence_level"] = _safe_int(
        _prompt("Confidence in studies (1–5):"), None)
    p["stress_level"]     = _safe_int(
        _prompt("Current stress level (1–5):"), None)
    p["quiz_score_avg"]   = _safe_float(
        _prompt("Average quiz / test score % (0–100):"), None)
    p["workload_credits"] = _safe_int(
        _prompt("Total credits this term (0 / 20 / 40 / 60 / 80):"), None)

    print_subheader("Tasks")
    tasks: List[Dict] = []
    idx = 1
    print("  Enter tasks one by one. Press ENTER with no name to finish.\n")
    while True:
        name = input(f"  Task {idx} name (ENTER to finish): ").strip()
        if not name:
            break
        subj = input("  Subject / course: ").strip() or "General"
        hrs  = _safe_float(input("  Estimated hours: ").strip(), 2.0)
        dl   = _safe_int(input("  Deadline in days from today: ").strip(), 7)
        tasks.append({"name": name, "subject": subj,
                      "hours": hrs, "deadline_days": dl})
        idx += 1
    p["tasks"] = tasks

    print_subheader("Exams")
    exams: List[Dict] = []
    print("  Enter upcoming exams. Press ENTER with no subject to finish.\n")
    while True:
        subj = input("  Exam subject (ENTER to finish): ").strip()
        if not subj:
            break
        days = _safe_int(
            input(f"  Days until '{subj}' exam: ").strip(), 14)
        exams.append({"subject": subj, "days_until": days})
    p["exams"] = exams

    print_subheader("Fixed activities  (lectures, work shifts, etc.)")
    activities: List[Dict] = []
    print("  Enter fixed activities. Press ENTER with no label to finish.\n")
    while True:
        label = input("  Activity label (ENTER to finish): ").strip()
        if not label:
            break
        day   = input("  Day (e.g. Monday): ").strip().capitalize()
        start = _safe_int(input("  Start hour (0–23): ").strip(), 9)
        end   = _safe_int(input("  End hour   (0–23): ").strip(), start + 1)
        activities.append({"label": label, "day": day,
                           "start_hour": start, "end_hour": end})
    p["fixed_activities"] = activities

    print_subheader("Availability this week")
    raw = _prompt(
        "Available study days (comma-separated, e.g. Mon, Wed, Sat):")
    p["available_days"] = [
        d.strip().capitalize()
        for d in raw.split(",")
        if any(d.strip().lower().startswith(w[:3].lower())
               for w in DAYS_OF_WEEK)
    ]
    p["daily_free_hours"] = _safe_float(
        _prompt("Free hours per available study day (average):"), None)

    return p


# =============================================================================
# 4.  Follow-up loop
# =============================================================================

def ask_followup(
    profile: Dict[str, Any],
    issues:  List[Issue],
) -> Dict[str, Any]:
    """
    For each issue print a clear explanation then ask a targeted question.
    Updates the profile in place and returns it.
    Only called in interactive mode — demo mode never reaches this.
    """
    errors   = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    if errors:
        print(f"\n  [!] {len(errors)} error(s) that must be resolved:")
    if warnings:
        print(f"  [?] {len(warnings)} item(s) worth clarifying:\n")

    for issue in issues:
        tag = "[!]" if issue.severity == "error" else "[?]"
        print(f"\n  {tag}  {issue.kind}  —  {issue.field}")
        print(f"       {issue.message}")
        answer = _prompt(issue.question)
        if not answer:
            continue

        fld = issue.field

        if fld == "name":
            profile["name"] = answer

        elif fld == "gender":
            v = _safe_int(answer, None)
            if v in (0, 1):
                profile["gender"] = v

        elif fld == "attendance":
            v = _safe_float(answer, None)
            if v is not None:
                profile["attendance"] = v

        elif fld == "confidence_level":
            v = _safe_int(answer, None)
            if v and 1 <= v <= 5:
                profile["confidence_level"] = v

        elif fld == "stress_level":
            v = _safe_int(answer, None)
            if v and 1 <= v <= 5:
                profile["stress_level"] = v

        elif fld == "quiz_score_avg":
            v = _safe_float(answer, None)
            if v is not None and 0 <= v <= 100:
                profile["quiz_score_avg"] = v
            elif v is not None:
                print(f"  [!] {v} is out of range — quiz score must be 0–100. "
                      "Value not updated, will be asked again.")

        elif fld == "quiz_score_avg / attendance":
            # Contradiction: low quiz + high attendance.
            # User may enter a corrected quiz score here.
            v = _safe_float(answer, None)
            if v is not None and 0 <= v <= 100:
                profile["quiz_score_avg"] = v
            elif v is not None:
                print(f"  [!] {v} is out of range — quiz score must be 0–100. "
                      "Value not updated.")

        elif fld.endswith(".deadline_days"):
            # Overdue task follow-up.
            # Three valid responses:
            #   "yes" / "y"  → task was submitted, remove it from the list
            #   a number ≥ 0 → user corrects the deadline to that many days
            #   ENTER / anything else → keep task as OVERDUE (deadline_days = 0)
            task_idx = None
            try:
                task_idx = int(fld.split("[")[1].split("]")[0])
            except (IndexError, ValueError):
                pass

            stripped = answer.strip().lower()

            if stripped in ("yes", "y"):
                # Remove the task entirely
                if task_idx is not None:
                    tasks_list = profile.get("tasks", [])
                    if 0 <= task_idx < len(tasks_list):
                        removed = tasks_list.pop(task_idx)
                        print(f"  Removed '{removed.get('name','?')}' "
                              "from task list.")

            else:
                # Try to parse as a corrected deadline number
                v = _safe_int(answer, None)
                if v is None:
                    v_f = _safe_float(answer, None)
                    v = int(v_f) if v_f is not None else None

                if v is not None and v >= 0:
                    # Valid corrected deadline — update the task
                    if task_idx is not None:
                        tasks_list = profile.get("tasks", [])
                        if 0 <= task_idx < len(tasks_list):
                            tasks_list[task_idx]["deadline_days"] = v
                            tasks_list[task_idx].pop("overdue", None)
                            name = tasks_list[task_idx].get("name", "?")
                            print(f"  Updated '{name}' deadline to {v} day(s).")
                else:
                    print("  Task kept as OVERDUE — will be scheduled immediately.")

        elif fld == "workload_credits":
            v = _safe_int(answer, None)
            if v in (0, 20, 40, 60, 80):
                profile["workload_credits"] = v

        elif fld == "tasks":
            print("  Re-enter tasks (previous entries kept if you leave blank):")
            new: List[Dict] = []
            while True:
                n = input("  Task name (ENTER to finish): ").strip()
                if not n:
                    break
                s = input("  Subject: ").strip() or "General"
                h = _safe_float(input("  Hours: ").strip(), 2.0)
                d = _safe_int(input("  Deadline days: ").strip(), 7)
                new.append({"name": n, "subject": s,
                            "hours": h, "deadline_days": d})
            if new:
                profile["tasks"] = new

        elif fld == "available_days":
            profile["available_days"] = [
                d.strip().capitalize()
                for d in answer.split(",")
                if any(d.strip().lower().startswith(w[:3].lower())
                       for w in DAYS_OF_WEEK)
            ]

        elif fld == "daily_free_hours":
            v = _safe_float(answer, None)
            if v is not None:
                profile["daily_free_hours"] = v

        elif "daily_free_hours" in fld or "available_days" in fld:
            if answer.lower().startswith("a"):
                for t in profile.get("tasks", []):
                    h = _safe_float(
                        input(f"  Revised hours for '{t['name']}'"
                              f" (was {t['hours']}): ").strip(),
                        t["hours"])
                    t["hours"] = h
            elif answer.lower().startswith("b"):
                extra = [d.strip().capitalize()
                         for d in input(
                             "  Additional available days: ").split(",")]
                profile["available_days"] = list(
                    set(profile.get("available_days", [])) | set(extra))
            elif answer.lower().startswith("c"):
                for t in profile.get("tasks", []):
                    d = _safe_int(
                        input(f"  New deadline for '{t['name']}'"
                              f" (was {t['deadline_days']}d): ").strip(),
                        t["deadline_days"])
                    t["deadline_days"] = d

    return profile


# =============================================================================
# 5.  Example profile generator
# =============================================================================

# Pool data the generator draws from
_SUBJECTS = ["Mathematics", "Biology", "History", "English Lit",
             "Chemistry", "Statistics", "Law", "Economics", "Physics"]

_TASK_TEMPLATES = [
    ("Essay",           3.0, 5.0),   # (name fragment, min_hours, max_hours)
    ("Problem set",     1.5, 4.0),
    ("Lab report",      2.0, 5.0),
    ("Case study",      3.0, 7.0),
    ("Reading",         1.0, 3.0),
    ("Revision notes",  2.0, 4.0),
    ("Past paper",      2.0, 5.0),
    ("Group project",   3.0, 8.0),
    ("Presentation",    2.0, 6.0),
    ("Research notes",  1.5, 4.0),
]

_NAMES = [
    ("Sam",    0), ("Jordan", 1), ("Alex",  0), ("Morgan", 1),
    ("Jamie",  0), ("Taylor", 1), ("Riley", 0), ("Casey",  1),
    ("Drew",   0), ("Quinn",  1), ("Avery", 0), ("Skyler", 1),
]

_ACTIVITY_TEMPLATES = [
    ("lecture",  2), ("tutorial", 1), ("seminar", 1),
    ("lab",      2), ("workshop", 2),
]


def _generate_example_profile() -> Dict[str, Any]:
    """
    Build a random but logically consistent student profile.

    Consistency rules:
    - A high-risk student (stress ≥ 4, confidence ≤ 2) gets more tasks,
      tighter deadlines, lower attendance, and fewer free hours.
    - A low-risk student gets the opposite.
    - workload_hours stays within ±50% of total free time so the
      schedule is tight but not always impossible.
    """
    rng = random.Random()   # unseeded → different every call

    # ── Archetype: drives all correlations ───────────────────────────────────
    archetype = rng.choice(["low_risk", "medium_risk", "high_risk"])

    if archetype == "low_risk":
        stress       = rng.randint(1, 2)
        confidence   = rng.randint(3, 5)
        attendance   = round(rng.uniform(75, 95), 1)
        quiz_avg     = round(rng.uniform(65, 90), 1)
        n_tasks      = rng.randint(2, 4)
        deadline_min, deadline_max = 6, 14
        free_hours   = round(rng.uniform(4.0, 6.0), 1)
        n_days       = rng.randint(4, 5)
        credits      = rng.choice([40, 60])

    elif archetype == "medium_risk":
        stress       = rng.randint(3, 3)
        confidence   = rng.randint(2, 3)
        attendance   = round(rng.uniform(60, 80), 1)
        quiz_avg     = round(rng.uniform(50, 70), 1)
        n_tasks      = rng.randint(4, 6)
        deadline_min, deadline_max = 3, 10
        free_hours   = round(rng.uniform(2.5, 4.0), 1)
        n_days       = rng.randint(3, 5)
        credits      = rng.choice([40, 60])

    else:   # high_risk
        stress       = rng.randint(4, 5)
        confidence   = rng.randint(1, 2)
        attendance   = round(rng.uniform(40, 65), 1)
        quiz_avg     = round(rng.uniform(30, 55), 1)
        n_tasks      = rng.randint(5, 7)
        deadline_min, deadline_max = 1, 7
        free_hours   = round(rng.uniform(1.5, 3.0), 1)
        n_days       = rng.randint(2, 4)
        credits      = rng.choice([60, 80])

    # ── Identity ──────────────────────────────────────────────────────────────
    name, gender = rng.choice(_NAMES)

    # ── Tasks — pick without replacement from templates ───────────────────────
    task_pool     = rng.sample(_TASK_TEMPLATES, k=min(n_tasks, len(_TASK_TEMPLATES)))
    subjects_used = rng.sample(_SUBJECTS, k=min(n_tasks, len(_SUBJECTS)))
    tasks = []
    for i, (tmpl_name, min_h, max_h) in enumerate(task_pool):
        subj    = subjects_used[i % len(subjects_used)]
        hours   = round(rng.uniform(min_h, max_h), 1)
        dl_days = rng.randint(deadline_min, deadline_max)
        tasks.append({
            "name":          f"{tmpl_name} — {subj}",
            "subject":       subj,
            "hours":         hours,
            "deadline_days": dl_days,
        })

    # ── Exams ─────────────────────────────────────────────────────────────────
    exam_subjects = rng.sample(subjects_used, k=min(rng.randint(0, 2), len(subjects_used)))
    exams = [
        {"subject": s, "days_until": rng.randint(deadline_min + 1, deadline_max + 5)}
        for s in exam_subjects
    ]

    # ── Available days ────────────────────────────────────────────────────────
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    available_days = sorted(
        rng.sample(weekdays, k=n_days),
        key=lambda d: weekdays.index(d),
    )

    # ── Fixed activities (1–3) ────────────────────────────────────────────────
    fixed_activities = []
    n_fixed = rng.randint(1, 3)
    for _ in range(n_fixed):
        act_name, duration = rng.choice(_ACTIVITY_TEMPLATES)
        day   = rng.choice(available_days)
        start = rng.randint(9, 15)
        subj  = rng.choice(subjects_used)
        fixed_activities.append({
            "label":      f"{subj} {act_name}",
            "day":        day,
            "start_hour": start,
            "end_hour":   start + duration,
        })

    profile: Dict[str, Any] = {
        "name":             name,
        "gender":           gender,
        "attendance":       attendance,
        "confidence_level": confidence,
        "stress_level":     stress,
        "quiz_score_avg":   quiz_avg,
        "workload_credits": credits,
        "tasks":            tasks,
        "exams":            exams,
        "fixed_activities": fixed_activities,
        "available_days":   available_days,
        "daily_free_hours": free_hours,
    }

    print(f"\n  Generated a '{archetype.replace('_', ' ')}' example profile "
          f"for {name}  "
          f"({len(tasks)} tasks · {free_hours} h/day free · "
          f"stress {stress}/5 · confidence {confidence}/5)\n")

    return profile


# =============================================================================
# 6.  Main entry point
# =============================================================================

def collect_student_inputs(
    mode:        str = "interactive",
    demo_data:   Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    mode = "interactive"  →  asks "manual" or "example" first, then
                              either prompts the user or generates a
                              random profile automatically.
    mode = "demo"         →  accepts demo_data dict directly (no prompts).

    Returns a fully validated StudentProfile dict with derived fields.
    """
    if mode == "demo":
        if demo_data is None:
            raise ValueError("demo_data required when mode='demo'")
        profile = dict(demo_data)
        issues  = validate_inputs(profile)
        hard = [
            i for i in issues
            if i.severity == "error" and i.kind in ("MISSING", "OUT_OF_RANGE")
        ]
        if hard:
            raise ValueError(
                f"Demo profile has {len(hard)} data error(s):\n" +
                "\n".join(f"  - {i.message}" for i in hard)
            )
        contradictions = [i for i in issues if i.kind == "CONTRADICTION"]
        if contradictions:
            print(f"  [demo] {len(contradictions)} contradiction(s) "
                  "detected (expected for this scenario):")
            for c in contradictions:
                print(f"    ↳ {c.message}")
        return compute_derived_fields(profile)

    # ── Interactive path — ask manual vs example first ────────────────────────
    print_header("Student Success Copilot — Input Collection")
    print("  How would you like to provide your profile?\n")
    print("    manual   — fill in your own details step by step")
    print("    example  — let the system generate a random student profile\n")

    while True:
        choice = input("  Enter 'manual' or 'example': ").strip().lower()
        if choice in ("manual", "example", "m", "e"):
            break
        print("  Please type 'manual' or 'example'.")

    if choice in ("example", "e"):
        profile = _generate_example_profile()
        # Example profiles are already consistent — run validation once
        # in read-only mode to surface any issues as information only.
        issues = validate_inputs(profile)
        if issues:
            print(f"  [example] {len(issues)} note(s) about this profile:")
            for i in issues:
                print(f"    ↳ [{i.kind}] {i.message}")
        return compute_derived_fields(profile)

    # manual path — existing interactive collection + follow-up loop
    profile  = _collect_interactive()
    attempts = 0

    while attempts < max_retries:
        issues = validate_inputs(profile)
        errors = [i for i in issues if i.severity == "error"]
        if not issues:
            print("\n  All inputs validated successfully.")
            break
        print(f"\n  Validation pass {attempts + 1} / {max_retries}")
        profile  = ask_followup(profile, issues)
        attempts += 1
        if attempts == max_retries and errors:
            print(f"\n  [!] Proceeding with {len(errors)} "
                  "unresolved error(s).")

    return compute_derived_fields(profile)


# =============================================================================
# 6.  Demo profiles
# =============================================================================

def load_demo_profile(scenario: str) -> Dict[str, Any]:
    """
    scenario = "A"  →  Alex — on track, low risk
    scenario = "B"  →  Jordan — overloaded, high risk
    """
    scenario = scenario.upper()

    if scenario == "A":
        raw = {
            "name": "Alex", "gender": 0,
            "attendance": 88.0, "confidence_level": 4,
            "stress_level": 2,  "quiz_score_avg": 78.0,
            "workload_credits": 60,
            "tasks": [
                {"name": "Essay draft",    "subject": "English Lit",
                 "hours": 4.0, "deadline_days": 6},
                {"name": "Lab report",     "subject": "Biology",
                 "hours": 3.0, "deadline_days": 9},
                {"name": "Problem set 3",  "subject": "Mathematics",
                 "hours": 2.5, "deadline_days": 11},
                {"name": "Reading ch 4-6", "subject": "History",
                 "hours": 2.0, "deadline_days": 14},
            ],
            "exams": [
                {"subject": "Mathematics", "days_until": 18},
            ],
            "fixed_activities": [
                {"label": "Maths lecture",   "day": "Monday",
                 "start_hour": 9,  "end_hour": 11},
                {"label": "Biology lab",     "day": "Wednesday",
                 "start_hour": 14, "end_hour": 16},
                {"label": "English seminar", "day": "Friday",
                 "start_hour": 10, "end_hour": 11},
            ],
            "available_days":   ["Monday", "Tuesday", "Wednesday",
                                  "Thursday", "Friday"],
            "daily_free_hours": 5.0,
        }

    elif scenario == "B":
        raw = {
            "name": "Jordan", "gender": 1,
            "attendance": 54.0, "confidence_level": 2,
            "stress_level": 5,  "quiz_score_avg": 41.0,
            "workload_credits": 80,
            "tasks": [
                {"name": "Research paper",       "subject": "Sociology",
                 "hours": 12.0, "deadline_days": 2},
                {"name": "Statistics exam prep", "subject": "Statistics",
                 "hours": 10.0, "deadline_days": 3},
                {"name": "Group project slides", "subject": "Business",
                 "hours": 6.0,  "deadline_days": 5},
                {"name": "Case study",           "subject": "Law",
                 "hours": 8.0,  "deadline_days": 6},
                {"name": "Weekly readings",      "subject": "Sociology",
                 "hours": 4.0,  "deadline_days": 7},
                {"name": "Problem set 5",        "subject": "Statistics",
                 "hours": 5.0,  "deadline_days": 8},
            ],
            "exams": [
                {"subject": "Statistics", "days_until": 4},
                {"subject": "Law",        "days_until": 10},
            ],
            "fixed_activities": [
                {"label": "Sociology lecture",   "day": "Monday",
                 "start_hour": 9,  "end_hour": 11},
                {"label": "Statistics tutorial", "day": "Monday",
                 "start_hour": 14, "end_hour": 15},
                {"label": "Business seminar",    "day": "Tuesday",
                 "start_hour": 11, "end_hour": 13},
                {"label": "Part-time work",      "day": "Wednesday",
                 "start_hour": 16, "end_hour": 20},
                {"label": "Law lecture",         "day": "Thursday",
                 "start_hour": 10, "end_hour": 12},
                {"label": "Part-time work",      "day": "Saturday",
                 "start_hour": 10, "end_hour": 18},
            ],
            "available_days":   ["Monday", "Tuesday", "Thursday", "Friday"],
            "daily_free_hours": 2.5,
        }

    elif scenario == "C":
        # ── Demo C: Mixed-signals trap ────────────────────────────────────────
        # Surface-level metrics look healthy (good attendance, confidence,
        # low stress, decent quiz scores) so the ML model predicts Pass with
        # high confidence.  But the student has 8 h of work, only ONE day
        # available (Monday, 3 free hours), and a deadline in 1 day.
        # The planner cannot fit everything, the rules engine fires overload
        # and deadline-cluster signals, and the mixed-signals detector kicks
        # in to downgrade confidence and print the AT RISK warning.
        # Purpose: demonstrate that the system catches manipulation /
        # accidental optimism bias in self-reported inputs.
        raw = {
            "name": "Sam", "gender": 0,
            "attendance": 85.0, "confidence_level": 4,
            "stress_level": 2,  "quiz_score_avg": 74.0,
            "workload_credits": 40,
            "tasks": [
                {"name": "Assignment draft",  "subject": "Economics",
                 "hours": 5.0, "deadline_days": 1},   # due TOMORROW
                {"name": "Tutorial worksheet","subject": "Statistics",
                 "hours": 3.0, "deadline_days": 2},
            ],
            "exams": [],
            "fixed_activities": [],
            # Only Monday is available — 3 free hours total
            "available_days":   ["Monday"],
            "daily_free_hours": 3.0,
        }

    else:
        raise ValueError(f"Unknown scenario '{scenario}'. Use 'A', 'B', or 'C'.")

    return collect_student_inputs(mode="demo", demo_data=raw)
