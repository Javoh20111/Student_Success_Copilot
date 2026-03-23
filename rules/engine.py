"""
rules/engine.py
===============
Generic rule engine for student risk assessment.

Architecture
------------
Rules live entirely in rules/rule_base.json — the engine contains
NO hardcoded decisions.  It only loads, evaluates, and chains rules.

Layers fire in order:
  O  (OBSERVE)   raw inputs    →  signals
  T  (THINK)     signals       →  hypotheses
  D  (DECIDE)    hypotheses    →  risk level conclusions
  C  (CONFIRM)   positive cues →  trust reinforcement / risk downgrade

Each layer is iterated to fixed-point before the next layer begins.

Condition operators supported
------------------------------
Numeric/bool facts:
  >  >=  <  <=  ==  !=  is_true  is_false

List facts (any item in a list matches):
  {"list_fact": "exams", "any_field": "days_until", "op": "<=", "value": 4}

Length checks on list facts:
  {"fact": "planner_not_scheduled", "op": "len_ge", "value": 1}
  {"fact": "available_days",        "op": "len_eq", "value": 3}

Risk level derivation
---------------------
Each fired rule contributes its confidence score to its risk_vote bucket
(Low / Medium / High).  Highest bucket total wins.
DOWNGRADE_RISK conclusions shift the winning bucket toward Low.

Backward chaining
-----------------
Given a goal such as "HIGH_RISK", the engine identifies all rules that
could help prove it, then checks which premise facts are absent from the
current fact base and returns the targeted question for each gap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Path to rule base — one level up from this file, then rules/rule_base.json
_RULE_BASE_PATH = Path(__file__).resolve().parent / "rule_base.json"

# Layer fire order
_LAYER_ORDER = ["O", "T", "D", "C"]


# =============================================================================
# 1.  Rule base loader
# =============================================================================

def load_rule_base(path: Path = _RULE_BASE_PATH) -> List[Dict[str, Any]]:
    """
    Load and return all rules from rule_base.json.
    Strips comment-only entries (those without an 'id' key).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [r for r in data["rules"] if "id" in r]


# =============================================================================
# 2.  Condition evaluator
# =============================================================================

def _compare(a: Any, op: str, b: Any) -> bool:
    """Apply a comparison operator between fact value a and threshold b."""
    if a is None:
        return False
    try:
        if op == ">":        return a > b
        if op == ">=":       return a >= b
        if op == "<":        return a < b
        if op == "<=":       return a <= b
        if op == "==":       return a == b
        if op == "!=":       return a != b
        if op == "is_true":  return bool(a)
        if op == "is_false": return not bool(a)
        if op == "len_ge":   return len(a) >= b
        if op == "len_eq":   return len(a) == b
    except TypeError:
        return False
    return False


def _eval_condition(cond: Dict[str, Any], facts: Dict[str, Any]) -> bool:
    """
    Evaluate a single condition dict against the current fact base.

    Supports:
      Normal:      {"fact": "attendance", "op": "<", "value": 70}
      List-any:    {"list_fact": "exams", "any_field": "days_until",
                    "op": "<=", "value": 4}
    """
    # List-item check: does ANY item in the list satisfy the condition?
    if "list_fact" in cond:
        items = facts.get(cond["list_fact"], [])
        if not isinstance(items, list):
            return False
        field = cond["any_field"]
        op    = cond["op"]
        val   = cond.get("value")   # may be absent for is_true / is_false
        return any(_compare(item.get(field), op, val) for item in items)

    # Normal fact check
    fact_val = facts.get(cond["fact"])
    return _compare(fact_val, cond["op"], cond.get("value"))


def _eval_rule(rule: Dict[str, Any], facts: Dict[str, Any]) -> bool:
    """
    Return True if ALL conditions in the rule are satisfied (implicit AND).
    Gracefully returns False if any condition raises an exception.
    """
    try:
        return all(_eval_condition(c, facts) for c in rule["conditions"])
    except Exception:
        return False


# =============================================================================
# 3.  Derived facts  (pre-computed before rule evaluation)
# =============================================================================

def _add_derived_facts(facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute derived numeric facts that rules reference.
    Called once before the O-layer fires.
    """
    daily_free = facts.get("daily_free_hours", 0)
    n_days     = len(facts.get("available_days", []))
    free_total = daily_free * n_days

    facts["free_hours_total"] = round(free_total, 1)
    facts["hour_surplus"]     = round(
        max(0.0, facts.get("workload_hours", 0) - free_total), 1
    )
    return facts


# =============================================================================
# 4.  Forward chaining
# =============================================================================

def _run_layer(
    layer:    str,
    rules:    List[Dict[str, Any]],
    facts:    Dict[str, Any],
    fired:    List[Tuple[str, str, float]],
    signals:  List[str],
    recs:     List[str],
    fired_set: set,
) -> bool:
    """
    Run one complete pass of a single layer to fixed-point.
    Returns True if at least one new rule fired.
    """
    layer_rules = [r for r in rules if r["layer"] == layer]
    any_new = False
    changed = True

    while changed:
        changed = False
        for rule in layer_rules:
            if rule["conclusion"] in fired_set:
                continue
            if _eval_rule(rule, facts):
                fired_set.add(rule["conclusion"])
                facts[rule["conclusion"]] = True
                fired.append((rule["id"], rule["conclusion"], rule["confidence"]))
                if rule.get("signal"):
                    signals.append(f"[{rule['id']}]  {rule['signal']}")
                if rule.get("recommendation"):
                    recs.append(rule["recommendation"])
                changed = True
                any_new = True

    return any_new


def _derive_risk_level(
    fired: List[Tuple[str, str, float]],
    rules: List[Dict[str, Any]],
    facts: Dict[str, Any],
) -> Tuple[str, float]:
    """
    Weighted vote: each fired rule contributes its confidence to its
    risk_vote bucket.  DOWNGRADE_RISK shifts the result toward Low.
    """
    rule_map = {r["id"]: r for r in rules}
    votes: Dict[str, float] = {"Low": 0.0, "Medium": 0.0, "High": 0.0}

    for rule_id, conclusion, conf in fired:
        rule = rule_map.get(rule_id)
        if rule:
            bucket = rule.get("risk_vote", "Medium")
            if bucket in votes:
                votes[bucket] += conf

    if not any(v > 0 for v in votes.values()):
        return "Low", 0.5

    winner = max(votes, key=lambda k: votes[k])
    total  = sum(votes.values())
    conf   = round(votes[winner] / total, 2) if total > 0 else 0.5

    # DOWNGRADE: if DOWNGRADE_RISK fired and winner is High/Medium,
    # step it down one level
    if facts.get("DOWNGRADE_RISK") and winner in ("High", "Medium"):
        winner = "Low" if winner == "Medium" else "Medium"
        conf   = round(conf * 0.85, 2)   # slight confidence reduction

    return winner, conf


# =============================================================================
# 5.  RuleEngine class
# =============================================================================

class RuleEngine:
    """
    Loads rules from rule_base.json and runs layered forward / backward
    inference.  No decisions are hardcoded here.

    Usage
    -----
    engine = RuleEngine()
    report = engine.forward_chain(facts_dict)
    gaps   = engine.backward_chain("HIGH_RISK", facts_dict)
    """

    def __init__(self, rule_base_path: Path = _RULE_BASE_PATH) -> None:
        self.rules = load_rule_base(rule_base_path)

    # ── Forward chaining ──────────────────────────────────────────────────────

    def forward_chain(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fire rules layer by layer (O → T → D → C).
        Each layer iterates to fixed-point before the next begins.
        Returns a risk_report dict.
        """
        facts      = _add_derived_facts(facts)
        fired:     List[Tuple[str, str, float]] = []
        signals:   List[str] = []
        recs:      List[str] = []
        fired_set: set = set()

        for layer in _LAYER_ORDER:
            _run_layer(layer, self.rules, facts,
                       fired, signals, recs, fired_set)

        risk_level, confidence = _derive_risk_level(fired, self.rules, facts)

        return {
            "name":            facts.get("name", "Student"),
            "risk_level":      risk_level,
            "confidence":      confidence,
            "signals":         signals,
            "rules_fired":     fired,
            "recommendations": recs,
            "explanation":     "",   # filled by generate_explanation
            "facts":           facts,
        }

    # ── Backward chaining ─────────────────────────────────────────────────────

    def backward_chain(
        self,
        goal:  str,
        facts: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Work backward from a goal to find which INPUT facts are missing.

        Recursively traces through intermediate conclusions so that a chain
        like:  HIGH_RISK ← crisis_risk (T8) ← extreme_stress (O5) ← stress_level
        correctly surfaces the primitive `stress_level` as missing.

        Returns list of:
          { missing_fact, question, needed_by (rule_id), to_prove }
        """
        FACT_QUESTIONS: Dict[str, str] = {
            "attendance":            "What is your class attendance % this term? (0-100)",
            "confidence_level":      "How confident do you feel about your studies? (1-5)",
            "stress_level":          "What is your current stress level? (1-5)",
            "quiz_score_avg":        "What is your average quiz / test score? (0-100)",
            "workload_hours":        "How many total study hours do your tasks require?",
            "deadlines":             "How many of your tasks are due within 7 days?",
            "exams":                 "Do you have any upcoming exams? How many days away?",
            "planner_found_goal":    "Has a complete study schedule been generated yet?",
            "planner_not_scheduled": "Are there tasks the planner could not schedule?",
            "workload_credits":      "How many credit hours are you taking this term?",
            "daily_free_hours":      "How many free hours per available study day?",
            "available_days":        "Which days of the week are you available to study?",
        }

        GOAL_MAP: Dict[str, List[str]] = {
            "HIGH_RISK": [
                "HIGH_RISK", "critical_overload", "academic_failure_risk",
                "exam_failure_risk", "crisis_risk", "schedule_overload",
                "engagement_risk", "deadline_cluster", "hour_shortfall",
                "exam_imminent",
            ],
            "MEDIUM_RISK": [
                "MEDIUM_RISK", "burnout_risk", "paralysis_risk",
                "deadline_crunch", "overcommitted_risk",
                "high_stress", "low_confidence",
            ],
            "LOW_RISK": [
                "LOW_RISK", "DOWNGRADE_RISK", "low_pressure",
                "positive_trajectory", "trusted_engagement",
                "schedule_feasible", "good_engagement",
            ],
        }

        # Build map: conclusion → rules that produce it
        conc_to_rules: Dict[str, list] = {}
        for rule in self.rules:
            conc_to_rules.setdefault(rule["conclusion"], []).append(rule)

        all_conclusions = {r["conclusion"] for r in self.rules}
        facts_local     = _add_derived_facts(dict(facts))
        missing:         List[Dict[str, Any]] = []
        seen_facts:      set = set()
        seen_conc:       set = set()   # cycle guard

        def _recurse(conclusion: str) -> None:
            if conclusion in seen_conc:
                return
            seen_conc.add(conclusion)

            for rule in conc_to_rules.get(conclusion, []):
                for cond in rule["conditions"]:
                    fk = cond.get("fact") or cond.get("list_fact")
                    if not fk:
                        continue
                    if fk in all_conclusions:
                        # Intermediate conclusion — trace back recursively
                        _recurse(fk)
                    elif fk not in seen_facts and facts_local.get(fk) is None:
                        seen_facts.add(fk)
                        missing.append({
                            "missing_fact": fk,
                            "question":     FACT_QUESTIONS.get(
                                fk, f"Please provide a value for '{fk}'."),
                            "needed_by":    rule["id"],
                            "to_prove":     conclusion,
                        })

        for target in GOAL_MAP.get(goal, [goal]):
            _recurse(target)

        return missing


# =============================================================================
# 6.  Module-level convenience functions
# =============================================================================

def _build_facts(
    profile:        Dict[str, Any],
    planner_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Merge profile + planner result into a flat fact dict for the engine.
    """
    facts = dict(profile)

    if planner_result:
        astar = planner_result.get("astar", planner_result)
        facts["planner_found_goal"]    = astar.get("found_goal", False)
        facts["planner_score"]         = astar.get("score", 0)
        facts["planner_conflicts"]     = astar.get("conflicts", 0)
        facts["planner_not_scheduled"] = astar.get("not_scheduled", [])
        facts["planner_partial"]       = astar.get("partial", [])
    else:
        facts.setdefault("planner_found_goal",    False)
        facts.setdefault("planner_not_scheduled", [])
        facts.setdefault("planner_partial",       [])

    return facts


def forward_chain(
    profile:        Dict[str, Any],
    planner_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the full layered forward chain on a student profile.
    Returns a risk_report dict ready for display_risk_report().
    """
    engine = RuleEngine()
    facts  = _build_facts(profile, planner_result)
    report = engine.forward_chain(facts)
    report["explanation"] = generate_explanation(report, profile, planner_result)
    return report


def backward_chain(
    goal:    str,
    profile: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Identify facts missing from the profile to prove the goal.
    Returns [{missing_fact, question, needed_by, to_prove}, …].
    """
    engine = RuleEngine()
    facts  = _build_facts(profile)
    return engine.backward_chain(goal, facts)


def generate_explanation(
    report:         Dict[str, Any],
    profile:        Dict[str, Any],
    planner_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a readable paragraph from the risk report.
    """
    risk    = report.get("risk_level", "Unknown")
    conf    = report.get("confidence", 0.0)
    name    = profile.get("name", "The student")
    fired   = report.get("rules_fired", [])
    n_rules = len(fired)

    rules     = load_rule_base()
    rule_map  = {r["id"]: r for r in rules}

    lines = [
        f"{name} has been assessed as {risk} risk "
        f"(confidence {conf:.0%}) based on {n_rules} rule(s) "
        f"across layers O→T→D→C."
    ]

    # Top 3 contributing rules by confidence
    top = sorted(fired, key=lambda x: x[2], reverse=True)[:3]
    if top:
        lines.append("Key contributing factors:")
        for rule_id, conclusion, c in top:
            rule = rule_map.get(rule_id)
            if rule:
                lines.append(
                    f"  • [{rule['layer']}] {rule['description']} "
                    f"(rule {rule_id}, confidence {c:.0%})"
                )

    # Planner context
    if planner_result:
        astar = planner_result.get("astar", planner_result)
        if astar.get("found_goal"):
            lines.append(
                "The search planner produced a complete, conflict-free schedule."
            )
        else:
            ns = astar.get("not_scheduled", [])
            pt = astar.get("partial", [])
            if ns or pt:
                lines.append(
                    f"The planner could not fully schedule all tasks: "
                    f"{len(pt)} partial, {len(ns)} not started."
                )

    # Closing advice by risk tier
    if risk == "High":
        lines.append(
            "Immediate action is recommended — review the suggestions above "
            "and contact your instructor or student support if needed."
        )
    elif risk == "Medium":
        lines.append(
            "This is manageable but requires attention. "
            "Follow the generated schedule and monitor progress daily."
        )
    else:
        lines.append(
            "Keep up the current approach — "
            "you are well-positioned for success."
        )

    return " ".join(lines)
