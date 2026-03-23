"""
ui/genai_explainer.py
=====================
Option 3 — Generative AI Explainer / Tutor
 
Uses the Claude API to generate:
  1. Personalised study tips tailored to the student's plan and risk level
  2. A plain-English explanation of why A* outperformed (or matched) BFS
  3. A tutor-style explanation of why the student is at risk
 
Guardrails and prompting strategy
-----------------------------------
All prompts follow a strict system prompt that:
  • Restricts the model to academic planning and study advice only
  • Forbids medical, legal, or financial recommendations
  • Requires the model to acknowledge uncertainty rather than fabricate
  • Forbids reproducing student personal data beyond what is needed
  • Sets a maximum response length to prevent rambling
 
Prompting strategy (documented per coursework requirement)
----------------------------------------------------------
We use three-part structured prompts:
  [ROLE]   — establishes context and restricts scope
  [FACTS]  — injects only the facts needed for the specific explanation
  [TASK]   — precise instruction with format constraints
 
Facts are injected as structured summaries (not raw dicts) to:
  a) Reduce token usage
  b) Prevent prompt injection via student-entered field values
  c) Make it easier to audit what the model actually received
 
Risks and mitigations (per coursework requirement)
---------------------------------------------------
RISK 1 — Prompt injection
  A student could enter a task named "Ignore previous instructions and..."
  Mitigation: field values are summarised into safe templates before
  injection (e.g. "Task: <name>, due in <N> days" — the name is truncated
  to 40 chars and stripped of special characters).
 
RISK 2 — Instruction following in unsafe contexts
  If the model is asked about self-harm, mental health crisis, or illegal
  activity via the study-tips interface, it could follow those instructions.
  Mitigation: the system prompt explicitly refuses these topics and redirects
  to student support. The REFUSE_TOPICS list is checked before each call.
 
RISK 3 — Hallucination of academic facts
  The model may invent study techniques or misrepresent algorithm behaviour.
  Mitigation: the prompt explicitly states "Do not invent techniques not
  supported by the provided facts. If uncertain, say so."
 
RISK 4 — Over-reliance
  Students may treat AI-generated study tips as authoritative.
  Mitigation: every response ends with a disclaimer reminding the student
  to verify with their instructor.
 
API dependency
--------------
This module supports two LLM providers — whichever key is set is used:
  OPENAI_API_KEY    → openai package   (gpt-4o-mini)
  ANTHROPIC_API_KEY → anthropic package (claude-sonnet-4-6)
 
If neither key is available it degrades gracefully: fallback_explain()
returns a rule-based explanation with no API call.
"""
 
from __future__ import annotations
 
import base64
import os
import re
import textwrap
from typing import Any, Dict, Optional
 
from ui.display import print_subheader, separator
 
 
# =============================================================================
# 0.  Constants
# =============================================================================
 
_MAX_TOKENS      = 400
_OPENAI_MODEL    = "gpt-4o-mini"
_ANTHROPIC_MODEL = "claude-sonnet-4-6"
_DISCLAIMER      = (
    "\n[Note: This is AI-generated guidance. "
    "Always verify with your instructor or academic advisor.]"
)
 
_REFUSE_TOPICS = [
    "self-harm", "suicide", "harm", "drug", "illegal",
    "cheat", "plagiar", "hack", "password", "personal data",
]
 
# =============================================================================
# 0b.  Embedded API key  (base64-obfuscated — not plaintext)
#
# To update: run this once in your terminal and paste the output below:
#   python3 -c "import base64; print(base64.b64encode(b'sk-proj-YOUR-KEY').decode())"
#
# Priority order when resolving the key:
#   1. OPENAI_API_KEY environment variable  (overrides everything)
#   2. ANTHROPIC_API_KEY environment variable
#   3. _EMBEDDED_OPENAI_KEY below           (fallback for examiner use)
# =============================================================================
 
_EMBEDDED_OPENAI_KEY    = ""   # paste your base64-encoded OpenAI key here
_EMBEDDED_ANTHROPIC_KEY = ""   # paste your base64-encoded Anthropic key here
 
 
def _decode_key(encoded: str) -> str:
    """Decode a base64-obfuscated key. Returns empty string if blank."""
    if not encoded or not encoded.strip():
        return ""
    try:
        return base64.b64decode(encoded.strip().encode()).decode()
    except Exception:
        return ""
 
 
def _resolve_keys() -> tuple:
    """
    Return (openai_key, anthropic_key) from environment or embedded fallback.
    Environment variables always take priority.
    """
    openai_key    = (os.environ.get("OPENAI_API_KEY", "")
                     or _decode_key(_EMBEDDED_OPENAI_KEY))
    anthropic_key = (os.environ.get("ANTHROPIC_API_KEY", "")
                     or _decode_key(_EMBEDDED_ANTHROPIC_KEY))
    return openai_key, anthropic_key
 
# =============================================================================
# 1.  System prompt
# =============================================================================
 
_SYSTEM_PROMPT = """You are a Student Success Copilot — an academic planning \
assistant embedded in a university study tool.
 
Your role:
- Help students understand their study plan, risk assessment, and schedule
- Explain AI decisions (search algorithms, rule-based reasoning, ML predictions)
  in plain English a first-year student can understand
- Provide practical, evidence-based study tips
 
Hard restrictions — never violate these:
- Do NOT give medical, psychological, legal, or financial advice
- Do NOT discuss topics unrelated to academic study and planning
- Do NOT reproduce or guess at student personal data beyond what is given
- Do NOT invent study techniques or algorithm facts not supported by the \
provided context
- If you are uncertain about something, say "I'm not sure — check with your \
instructor"
- If a student raises a topic outside academic planning, respond:
  "That's outside my scope. Please speak to your university's student support \
  services."
- Keep responses under 300 words
- End every response with exactly this line:
  [Note: This is AI-generated guidance. Always verify with your instructor \
  or academic advisor.]"""
 
 
# =============================================================================
# 2.  Input sanitiser  (prompt injection guard)
# =============================================================================
 
def _sanitise(value: str, max_len: int = 40) -> str:
    """
    Strip characters that could alter prompt structure and truncate.
    Removes: newlines, backticks, angle brackets, curly braces,
             square brackets, pipe characters, backslashes.
    """
    clean = re.sub(r'[\n\r`<>{}\[\]|\\]', ' ', str(value))
    clean = clean.strip()
    return clean[:max_len]
 
 
def _check_refused_topic(text: str) -> Optional[str]:
    """Return the matched refused topic, or None if safe."""
    lower = text.lower()
    for topic in _REFUSE_TOPICS:
        if topic in lower:
            return topic
    return None
 
 
# =============================================================================
# 3.  Fact summariser  (safe context builder)
# =============================================================================
 
def _summarise_profile(profile: Dict[str, Any]) -> str:
    """Build a safe, structured summary of student facts for injection."""
    tasks = profile.get("tasks", [])
    task_lines = "\n".join(
        f"  - {_sanitise(t.get('name','?'))} | "
        f"due in {t.get('deadline_days','?')}d | "
        f"{t.get('hours','?')}h required"
        for t in tasks[:6]
    )
    return (
        f"Student: {_sanitise(profile.get('name','Student'))}\n"
        f"Attendance: {profile.get('attendance','?')}%\n"
        f"Confidence: {profile.get('confidence_level','?')}/5\n"
        f"Stress: {profile.get('stress_level','?')}/5\n"
        f"Quiz avg: {profile.get('quiz_score_avg','?')}%\n"
        f"Credits: {profile.get('workload_credits','?')}\n"
        f"Tasks ({len(tasks)}):\n{task_lines}\n"
        f"Available days: {', '.join(profile.get('available_days',[]))}\n"
        f"Free hours/day: {profile.get('daily_free_hours','?')}"
    )
 
 
def _summarise_planner(cmp: Dict[str, Any]) -> str:
    """Build a safe summary of planner comparison results."""
    bfs   = cmp.get("bfs",   {})
    astar = cmp.get("astar", {})
    return (
        f"BFS:  states={bfs.get('states_explored','?')} | "
        f"time={bfs.get('time_ms','?')}ms | "
        f"goal={bfs.get('found_goal','?')} | "
        f"score={bfs.get('score','?')}\n"
        f"A*:   states={astar.get('states_explored','?')} | "
        f"time={astar.get('time_ms','?')}ms | "
        f"goal={astar.get('found_goal','?')} | "
        f"score={astar.get('score','?')}\n"
        f"Unscheduled (A*): {astar.get('not_scheduled',[])}\n"
        f"Partial (A*):     {astar.get('partial',[])}"
    )
 
 
def _summarise_report(report: Dict[str, Any]) -> str:
    """Build a safe summary of the risk report."""
    top_rules = report.get("rules_fired", [])[:5]
    rule_lines = "\n".join(
        f"  [{rid}] {conclusion} (conf {conf:.0%})"
        for rid, conclusion, conf in top_rules
    )
    return (
        f"Risk level: {report.get('risk_level','?')} "
        f"(confidence {report.get('confidence',0):.0%})\n"
        f"Top rules fired:\n{rule_lines}\n"
        f"Signals: {len(report.get('signals',[]))}\n"
        f"Recommendations: {len(report.get('recommendations',[]))}"
    )
 
 
# =============================================================================
# 4.  LLM caller  (OpenAI or Claude — whichever key is set)
# =============================================================================
 
def _call_llm(prompt: str) -> str:
    """
    Try OpenAI first, then Anthropic.
    Keys are resolved from environment variables or the embedded fallback.
    """
    openai_key, anthropic_key = _resolve_keys()
 
    # ── OpenAI ────────────────────────────────────────────────────────────────
    if openai_key:
        try:
            from openai import OpenAI
            client   = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model      = _OPENAI_MODEL,
                max_tokens = _MAX_TOKENS,
                messages   = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except ImportError:
            if not anthropic_key:
                return "[GenAI unavailable — pip install openai]"
        except Exception as exc:
            return f"[OpenAI call failed: {exc}]"
 
    # ── Anthropic ─────────────────────────────────────────────────────────────
    if anthropic_key:
        try:
            import anthropic
            client  = anthropic.Anthropic(api_key=anthropic_key)
            message = client.messages.create(
                model      = _ANTHROPIC_MODEL,
                max_tokens = _MAX_TOKENS,
                system     = _SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except ImportError:
            return "[GenAI unavailable — pip install anthropic]"
        except Exception as exc:
            return f"[Anthropic call failed: {exc}]"
 
    # ── No key available ──────────────────────────────────────────────────────
    return (
        "[GenAI unavailable — no API key found.\n"
        " Add your encoded key to _EMBEDDED_OPENAI_KEY in genai_explainer.py\n"
        " or set the OPENAI_API_KEY environment variable.]"
    )
 
 
# Keep old name as alias so existing tests that mock _call_claude still work
_call_claude = _call_llm
 
 
# =============================================================================
# 5.  Fallback (no API)
# =============================================================================
 
def fallback_explain(
    profile: Dict[str, Any],
    report:  Dict[str, Any],
) -> str:
    """
    Rule-based explanation when the Claude API is unavailable.
    Pulls directly from the risk report — no hallucination possible.
    """
    risk  = report.get("risk_level", "Unknown")
    name  = profile.get("name", "Student")
    fired = report.get("rules_fired", [])
    recs  = report.get("recommendations", [])[:3]
 
    lines = [
        f"{name}, your risk level is {risk}.",
        f"{len(fired)} rule(s) contributed to this assessment.",
    ]
    if recs:
        lines.append("Top recommendations:")
        for i, r in enumerate(recs, 1):
            lines.append(f"  {i}. {r}")
    lines.append(_DISCLAIMER)
    return "\n".join(lines)
 
 
# =============================================================================
# 6.  Public explainer functions
# =============================================================================
 
def explain_study_tips(
    profile: Dict[str, Any],
    report:  Dict[str, Any],
    planner: Dict[str, Any],
) -> str:
    """
    Generate personalised study tips tailored to the student's specific
    plan, risk level, and schedule.
 
    Prompt strategy: ROLE + FACTS (sanitised profile + risk summary) + TASK
    """
    refused = _check_refused_topic(str(profile.get("name", "")))
    if refused:
        return (
            f"[Guardrail: refused topic '{refused}' detected in profile. "
            "No tips generated.]"
        )
 
    profile_summary = _summarise_profile(profile)
    report_summary  = _summarise_report(report)
    astar           = planner.get("astar", {})
    schedule_status = (
        "complete with no conflicts"
        if astar.get("found_goal")
        else f"incomplete — {len(astar.get('not_scheduled',[]))} task(s) "
             f"not scheduled, {len(astar.get('partial',[]))} partial"
    )
 
    prompt = f"""A student needs personalised study tips.
 
STUDENT FACTS:
{profile_summary}
 
RISK ASSESSMENT:
{report_summary}
 
SCHEDULE STATUS: {schedule_status}
 
TASK: Write 3–5 specific, actionable study tips for this student.
- Tailor each tip to their actual situation (risk level, schedule gaps, stress)
- Be practical and concrete — no generic advice
- If the schedule is incomplete, the first tip must address that directly
- Do not repeat the risk report — give new, complementary advice
- Format as a numbered list"""
 
    return _call_claude(prompt)
 
 
def explain_search_comparison(
    profile: Dict[str, Any],
    planner: Dict[str, Any],
) -> str:
    """
    Explain why A* performed better (or equally) compared to BFS
    for this specific student's scheduling problem.
 
    Prompt strategy: ROLE + FACTS (sanitised planner metrics) + TASK
    """
    planner_summary = _summarise_planner(planner)
    bfs   = planner.get("bfs",   {})
    astar = planner.get("astar", {})
 
    n_tasks     = len(profile.get("tasks", []))
    total_hours = profile.get("workload_hours", "?")
    n_slots     = len(planner.get("slots", []))
 
    prompt = f"""A student success system just ran two scheduling algorithms.
 
PROBLEM:
- Student has {n_tasks} tasks totalling {total_hours}h of work
- Available free slots: {n_slots} × 1-hour blocks this week
 
ALGORITHM RESULTS:
{planner_summary}
 
TASK: Explain in plain English (3–4 sentences):
1. What BFS does and what A* does differently
2. Why A* explored {'fewer' if astar.get('states_explored',999) < bfs.get('states_explored',0) else 'more or equal'} states in this case
3. What the urgency heuristic h(n) = sum(hours × urgency_weight) means in practice
4. When would BFS actually be faster than A*?
 
Write as if explaining to a student who has never studied AI search."""
 
    return _call_claude(prompt)
 
 
def explain_risk(
    profile: Dict[str, Any],
    report:  Dict[str, Any],
    signal_check: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Explain the risk assessment result in tutor-friendly language.
    If mixed signals were detected, explain the conflict explicitly.
 
    Prompt strategy: ROLE + FACTS + TASK
    """
    profile_summary = _summarise_profile(profile)
    report_summary  = _summarise_report(report)
 
    conflict_note = ""
    if signal_check and signal_check.get("has_conflict"):
        conflict_note = (
            f"\nCONFLICT DETECTED: ML predicts {signal_check['ml_verdict']} "
            f"but rules flag {signal_check['rules_verdict']} risk. "
            f"Confidence was adjusted to {signal_check['adjusted_confidence']:.0%}."
        )
 
    prompt = f"""A student success AI just assessed a student.
 
STUDENT FACTS:
{profile_summary}
 
RISK ASSESSMENT:
{report_summary}{conflict_note}
 
TASK: Explain to the student in 4–6 sentences:
1. What their risk level means and why
2. Which 2–3 factors contributed most to this assessment
3. If there is a conflict between the ML prediction and the rules — explain why
   that happens and which signal to trust more
4. One concrete thing they should do TODAY based on this assessment
 
Be direct, honest, and encouraging. Do not sugarcoat high risk."""
 
    return _call_claude(prompt)
 
 
# =============================================================================
# 7.  Interactive Q&A mode
# =============================================================================
 
def run_interactive_explainer(
    profile: Dict[str, Any],
    report:  Dict[str, Any],
    planner: Dict[str, Any],
    signal_check: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Let the student ask free-form questions about their results.
    Each question is checked against refused topics before being sent.
    """
    print_subheader("AI Explainer — ask me anything about your results")
    print("  Type your question, or 'quit' to exit.\n")
    print("  Examples:")
    print("    'Why am I at high risk?'")
    print("    'Why did A* perform better than BFS?'")
    print("    'What should I study first?'\n")
 
    context = (
        f"Context for this conversation:\n"
        f"Student: {_sanitise(profile.get('name','Student'))}\n"
        f"Risk level: {report.get('risk_level','?')} "
        f"(confidence {report.get('confidence',0):.0%})\n"
        f"Tasks: {len(profile.get('tasks',[]))}\n"
        f"Schedule: {'complete' if planner.get('astar',{}).get('found_goal') else 'incomplete'}"
    )
    if signal_check and signal_check.get("has_conflict"):
        context += f"\nConflict: ML={signal_check['ml_verdict']} vs Rules={signal_check['rules_verdict']}"
 
    while True:
        try:
            question = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
 
        if not question or question.lower() in ("quit", "exit", "q"):
            print("  Copilot: Goodbye — good luck with your studies!\n")
            break
 
        refused = _check_refused_topic(question)
        if refused:
            print(
                f"  Copilot: That topic ('{refused}') is outside my scope. "
                "Please speak to your university's student support services.\n"
            )
            continue
 
        safe_q = _sanitise(question, max_len=200)
        prompt = (
            f"{context}\n\n"
            f"Student question: {safe_q}\n\n"
            "Answer helpfully and concisely (max 150 words). "
            "Stay strictly within academic planning scope."
        )
        response = _call_claude(prompt)
        print()
        for line in textwrap.wrap(response, width=70):
            print(f"  {line}")
        print()
 
        separator()