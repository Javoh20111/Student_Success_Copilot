"""
ui/nlp_interface.py
===================
Option 2 — NLP Mini-Interface

A chatbot-style text interface that:
  • collects student constraints from free-form natural language
  • detects missing information and asks targeted follow-up questions
  • detects contradictions and flags them explicitly
  • produces a validated StudentProfile dict identical to the one
    produced by the form-based input_layer.py

Implementation approach
-----------------------
We use rule-based NLP (regex + keyword matching) rather than a language
model for this layer, for two reasons:
  1. Transparency — every extraction decision is auditable
  2. No API dependency — runs fully offline

The extractor looks for patterns like:
  "I attend about 80% of my classes"     → attendance = 80
  "I have a report due in 3 days"        → task with deadline_days = 3
  "pretty stressed, like a 4 out of 5"  → stress_level = 4
  "free on Monday and Wednesday"         → available_days = [Monday, Wednesday]

After extraction, the same validate_inputs() from input_layer.py is called
so ALL existing validation logic (OOR, contradictions, overdue) is reused.
The NLP layer only handles the input collection differently.

Limitations (documented per coursework requirement)
----------------------------------------------------
• Handles one value per sentence — "I'm 70% confident and 80% attended"
  will only extract one.
• Date arithmetic is not implemented — "due Friday" is not parsed, only
  "due in N days".
• Negation is not handled — "I am NOT stressed" is treated as stressed.
• Only English is supported.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ui.display     import print_header, print_subheader, separator
from ui.input_layer import (
    validate_inputs, compute_derived_fields,
    ask_followup, DAYS_OF_WEEK,
)


# =============================================================================
# 1.  Extraction patterns
# =============================================================================

# Each pattern: (compiled_regex, field_name, extractor_fn)
# extractor_fn takes the regex match and returns the parsed value

_PATTERNS: List[Tuple[re.Pattern, str, Any]] = [

    # ── attendance ─────────────────────────────────────────────────────────────
    (re.compile(
        r'\b(?:attend(?:ed|ance)?[^.]*?|go(?:ing)?\s+to\s+class[^.]*?)'
        r'(\d{1,3})\s*%',
        re.I),
     "attendance", lambda m: float(m.group(1))),

    (re.compile(
        r'(\d{1,3})\s*%[^.]*?(?:attend|class|lecture)',
        re.I),
     "attendance", lambda m: float(m.group(1))),

    # ── confidence_level ──────────────────────────────────────────────────────
    (re.compile(
        r'confiden(?:t|ce)[^.]*?(\d)\s*(?:out\s+of\s+5|/\s*5)',
        re.I),
     "confidence_level", lambda m: int(m.group(1))),

    (re.compile(
        r'(\d)\s*/\s*5[^.]*?confiden',
        re.I),
     "confidence_level", lambda m: int(m.group(1))),

    # ── stress_level ──────────────────────────────────────────────────────────
    (re.compile(
        r'stress(?:ed)?[^.]*?(\d)\s*(?:out\s+of\s+5|/\s*5)',
        re.I),
     "stress_level", lambda m: int(m.group(1))),

    (re.compile(
        r'(\d)\s*/\s*5[^.]*?stress',
        re.I),
     "stress_level", lambda m: int(m.group(1))),

    # ── quiz_score_avg ────────────────────────────────────────────────────────
    (re.compile(
        r'(?:quiz|test|exam\s+score|average)[^.]*?(\d{1,3})\s*%',
        re.I),
     "quiz_score_avg", lambda m: float(m.group(1))),

    (re.compile(
        r'(?:scoring|scored|averaging)[^.]*?(\d{1,3})\s*%',
        re.I),
     "quiz_score_avg", lambda m: float(m.group(1))),

    # ── workload_credits ──────────────────────────────────────────────────────
    (re.compile(
        r'(\d+)\s*credits?',
        re.I),
     "workload_credits", lambda m: int(m.group(1))),

    # ── daily_free_hours ──────────────────────────────────────────────────────
    (re.compile(
        r'(\d+(?:\.\d+)?)\s*(?:free\s+)?hours?\s+(?:per|a|each)\s+day',
        re.I),
     "daily_free_hours", lambda m: float(m.group(1))),

    (re.compile(
        r'free\s+(\d+(?:\.\d+)?)\s*hours?\s+(?:per|a|each)?\s*day',
        re.I),
     "daily_free_hours", lambda m: float(m.group(1))),
]

# Task pattern: "I have a <name> due in <N> days, about <H> hours"
_TASK_PATTERN = re.compile(
    r'(?:have\s+(?:a\s+)?|need\s+to\s+(?:do\s+)?|working\s+on\s+)?'
    r'(?P<name>[a-zA-Z][\w\s\-]{2,40}?)'
    r'\s+(?:due|deadline)\s+(?:in\s+)?(?P<days>\d+)\s+days?'
    r'(?:[^.]*?(?P<hours>\d+(?:\.\d+)?)\s*hours?)?',
    re.I,
)

# Available days pattern
_DAYS_PATTERN = re.compile(
    r'(?:free|available|study|open)\s+(?:on\s+)?'
    r'((?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)'
    r'(?:[,\s]+(?:and\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))*)',
    re.I,
)

# Exam pattern: "<subject> exam in <N> days"
_EXAM_PATTERN = re.compile(
    r'(?P<subject>[\w\s]+?)\s+exam\s+(?:in\s+)?(?P<days>\d+)\s+days?',
    re.I,
)


# =============================================================================
# 2.  Single-utterance extractor
# =============================================================================

def extract_from_text(text: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run all patterns against a single line of text.
    Updates profile in place with any extracted values.
    Returns the updated profile.
    """
    # ── Name (special case — plain word or "my name is X") ───────────────────
    if profile.get("name") is None:
        stripped = text.strip()
        # "My name is Sam" / "I'm Sam" / "call me Sam"
        m_intro = re.search(
            r"(?:my name is|i(?:'m| am)|call me)\s+([A-Z][a-z]+)",
            stripped, re.I,
        )
        if m_intro:
            profile["name"] = m_intro.group(1).strip().title()
        # Single capitalised word on its own — treat as a name
        elif re.fullmatch(r"[A-Z][a-zA-Z'\-]{1,30}", stripped):
            profile["name"] = stripped.title()
        # Single lowercase word that is clearly a name (no digits/symbols)
        elif re.fullmatch(r"[a-zA-Z]{2,20}", stripped):
            profile["name"] = stripped.title()

    # ── Gender (special case — "I am male/female" or standalone word) ─────────
    if profile.get("gender") is None:
        lower = text.lower().strip()
        if re.search(r'\b(male|man|guy|he|he/him)\b', lower):
            profile["gender"] = 0
        elif re.search(r'\b(female|woman|girl|she|she/her)\b', lower):
            profile["gender"] = 1
        elif lower in ("0", "1"):
            profile["gender"] = int(lower)

    # ── Scalar fields ─────────────────────────────────────────────────────────
    for pattern, field, extractor in _PATTERNS:
        if profile.get(field) is not None:
            continue                    # already have this value — skip
        m = pattern.search(text)
        if m:
            try:
                profile[field] = extractor(m)
            except (ValueError, TypeError):
                pass

    # ── Tasks ─────────────────────────────────────────────────────────────────
    for m in _TASK_PATTERN.finditer(text):
        name  = m.group("name").strip().title()
        days  = int(m.group("days"))
        hours = float(m.group("hours")) if m.group("hours") else 2.0
        task  = {"name": name, "subject": "General",
                 "hours": hours, "deadline_days": days}
        existing_names = {t["name"].lower() for t in profile.get("tasks", [])}
        if name.lower() not in existing_names:
            profile.setdefault("tasks", []).append(task)

    # ── Available days ────────────────────────────────────────────────────────
    m = _DAYS_PATTERN.search(text)
    if m:
        raw_days = re.findall(
            r'monday|tuesday|wednesday|thursday|friday|saturday|sunday',
            m.group(1), re.I,
        )
        extracted = [d.capitalize() for d in raw_days]
        existing  = set(profile.get("available_days", []))
        combined  = list(existing | set(extracted))
        profile["available_days"] = [
            d for d in DAYS_OF_WEEK if d in combined
        ]

    # ── Exams ─────────────────────────────────────────────────────────────────
    for m in _EXAM_PATTERN.finditer(text):
        subject = m.group("subject").strip().title()
        days    = int(m.group("days"))
        existing = {e["subject"].lower() for e in profile.get("exams", [])}
        if subject.lower() not in existing:
            profile.setdefault("exams", []).append(
                {"subject": subject, "days_until": days}
            )

    return profile


# =============================================================================
# 3.  Missing field detector
# =============================================================================

_REQUIRED_FIELDS = [
    ("name",             "What is your name?"),
    ("gender",           "Are you male or female? (or enter 0/1)"),
    ("attendance",       "What percentage of classes have you attended? (0–100)"),
    ("confidence_level", "How confident do you feel about your studies? (1–5)"),
    ("stress_level",     "How stressed are you right now? (1–5)"),
    ("quiz_score_avg",   "What is your average quiz / test score? (0–100)"),
    ("workload_credits", "How many credit hours are you taking? (0, 20, 40, 60, or 80)"),
    ("tasks",            "Tell me about a task you need to complete. "
                         "Try: 'I have a <name> due in <N> days, about <H> hours'"),
    ("available_days",   "Which days are you free to study? "
                         "(e.g. 'free on Monday and Wednesday')"),
    ("daily_free_hours", "How many free hours do you have per study day on average?"),
]


def get_next_question(profile: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    Return (field_name, question) for the next missing required field.
    Returns None when all required fields are present.
    """
    for field, question in _REQUIRED_FIELDS:
        val = profile.get(field)
        if val is None or val == [] or (isinstance(val, str) and not val.strip()):
            return field, question
    return None


# =============================================================================
# 3b.  Context-aware bare-number resolver
# =============================================================================

# Filler words that precede a bare number ("maybe 4", "around 78")
_FILLER = re.compile(
    r'^(?:maybe|around|about|approximately|roughly|like|i\'d say|probably|'
    r'i think|i\'m at|i\'m a|i am a|i am)[\s,]*',
    re.I,
)

def _try_bare_answer(text: str, field: str, profile: Dict[str, Any]) -> bool:
    """
    When the system just asked about `field`, try to parse the user's
    reply as a direct answer — even if it's just a number like "4" or
    a hedged phrase like "maybe 78" or "approximately 4".

    Returns True if the field was successfully updated.
    """
    # Strip filler words to get at the number
    cleaned = _FILLER.sub("", text.strip())
    # Extract the first number (int or float)
    m = re.search(r'\d+(?:\.\d+)?', cleaned)
    if not m:
        return False

    val_f = float(m.group())
    val_i = int(val_f)

    updated = False

    if field == "attendance" and 0 <= val_f <= 100:
        profile["attendance"] = val_f
        updated = True

    elif field == "confidence_level" and 1 <= val_i <= 5:
        profile["confidence_level"] = val_i
        updated = True

    elif field == "stress_level" and 1 <= val_i <= 5:
        profile["stress_level"] = val_i
        updated = True

    elif field == "quiz_score_avg" and 0 <= val_f <= 100:
        profile["quiz_score_avg"] = val_f
        updated = True

    elif field == "workload_credits" and val_i in (0, 20, 40, 60, 80):
        profile["workload_credits"] = val_i
        updated = True

    elif field == "daily_free_hours" and 0 < val_f <= 16:
        profile["daily_free_hours"] = val_f
        updated = True

    elif field == "gender" and val_i in (0, 1):
        profile["gender"] = val_i
        updated = True

    return updated

def nlp_collect_inputs(max_turns: int = 30) -> Dict[str, Any]:
    """
    Chatbot-style collection loop.

    The student types freely. Each message is passed to extract_from_text().
    After extraction the system asks about the next missing field.
    Once all required fields are present, validate_inputs() is called.
    Any issues are surfaced with ask_followup().

    Returns a validated StudentProfile dict (same format as input_layer.py).
    """
    print_header("Student Success Copilot — Chat Interface")
    print(
        "  Talk to me naturally. Tell me about your courses, tasks,\n"
        "  how you're feeling, and when you're free to study.\n"
        "  Type 'done' when you're finished entering information.\n"
        "  Type 'status' to see what I've collected so far.\n"
    )

    profile: Dict[str, Any] = {
        "tasks":            [],
        "exams":            [],
        "fixed_activities": [],
        "available_days":   [],
    }

    # Opening question
    next_q = get_next_question(profile)
    pending_field = next_q[0] if next_q else None
    if next_q:
        print(f"  Copilot: Let's start — {next_q[1]}\n")

    turns = 0
    while turns < max_turns:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  [Chat ended by user]")
            break

        if not user_input:
            continue

        if user_input.lower() == "done":
            print("  Copilot: Got it, let me check what I have...\n")
            break

        if user_input.lower() == "status":
            _print_status(profile)
            turns += 1
            continue

        # Extract from this message
        before = _snapshot(profile)
        extract_from_text(user_input, profile)

        # If patterns found nothing AND we know what field was asked,
        # try parsing the reply as a direct bare-number answer
        after      = _snapshot(profile)
        new_fields = {k for k in after if after[k] != before.get(k)}
        if not new_fields and pending_field:
            if _try_bare_answer(user_input, pending_field, profile):
                new_fields = {pending_field}

        # Acknowledge
        if new_fields:
            acks = _acknowledge(new_fields, profile)
            if acks:
                print(f"  Copilot: Got it — {acks}")
        else:
            print("  Copilot: I didn't catch that. "
                  "Try a number or a short phrase, e.g. '78%' or '4 out of 5'.")

        # Ask about the next missing field
        next_q = get_next_question(profile)
        if next_q:
            pending_field = next_q[0]
            print(f"  Copilot: {next_q[1]}\n")
        else:
            pending_field = None
            print("  Copilot: I think I have everything. "
                  "Type 'done' to continue or keep adding details.\n")

        turns += 1

    # ── Post-chat validation ──────────────────────────────────────────────────
    print_subheader("Validating your profile")

    # Fill in defaults for optional fields
    profile.setdefault("fixed_activities", [])
    if not profile.get("name"):
        profile["name"] = "Student"

    # Run the same validation as the form-based interface
    issues = validate_inputs(profile)
    if issues:
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            print(f"  I still need {len(errors)} piece(s) of information:\n")
            profile = ask_followup(profile, errors)

    return compute_derived_fields(profile)


# =============================================================================
# 5.  Helpers
# =============================================================================

def _snapshot(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow snapshot for change detection."""
    return {
        k: (list(v) if isinstance(v, list) else v)
        for k, v in profile.items()
    }


def _acknowledge(new_fields: set, profile: Dict[str, Any]) -> str:
    """Build a short acknowledgement string for newly extracted fields."""
    parts = []
    for f in sorted(new_fields):
        val = profile.get(f)
        if f == "attendance":
            parts.append(f"attendance {val}%")
        elif f == "confidence_level":
            parts.append(f"confidence {val}/5")
        elif f == "stress_level":
            parts.append(f"stress {val}/5")
        elif f == "quiz_score_avg":
            parts.append(f"quiz avg {val}%")
        elif f == "tasks":
            latest = profile["tasks"][-1] if profile["tasks"] else None
            if latest:
                parts.append(
                    f"task '{latest['name']}' "
                    f"(due in {latest['deadline_days']}d, {latest['hours']}h)"
                )
        elif f == "available_days":
            parts.append(f"available days: {', '.join(profile['available_days'])}")
        elif f == "daily_free_hours":
            parts.append(f"{val}h free per day")
        elif f == "gender":
            parts.append("Male" if val == 0 else "Female")
        elif f == "name":
            parts.append(f"name: {val}")
        elif f == "workload_credits":
            parts.append(f"{val} credits")
    return ", ".join(parts)


def _print_status(profile: Dict[str, Any]) -> None:
    """Print a compact summary of everything collected so far."""
    print_subheader("What I have so far")
    fields = [
        ("Name",           profile.get("name")),
        ("Gender",         "Male" if profile.get("gender") == 0
                           else "Female" if profile.get("gender") == 1
                           else None),
        ("Attendance",     f"{profile.get('attendance')}%" if profile.get("attendance") is not None else None),
        ("Confidence",     f"{profile.get('confidence_level')}/5" if profile.get("confidence_level") else None),
        ("Stress",         f"{profile.get('stress_level')}/5" if profile.get("stress_level") else None),
        ("Quiz avg",       f"{profile.get('quiz_score_avg')}%" if profile.get("quiz_score_avg") is not None else None),
        ("Credits",        profile.get("workload_credits")),
        ("Available days", ", ".join(profile.get("available_days", [])) or None),
        ("Free hrs/day",   profile.get("daily_free_hours")),
    ]
    for label, val in fields:
        tick = "✓" if val is not None else "○"
        print(f"  {tick}  {label:<18} {val or '(not yet provided)'}")
    tasks = profile.get("tasks", [])
    if tasks:
        print(f"  ✓  Tasks ({len(tasks)}):")
        for t in tasks:
            print(f"       • {t['name']}  — {t['hours']}h, due in {t['deadline_days']}d")
    else:
        print("  ○  Tasks           (not yet provided)")
    print()
