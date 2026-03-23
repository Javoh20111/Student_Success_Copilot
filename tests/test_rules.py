"""
tests/test_rules.py
===================
Tests for rules/engine.py + rules/rule_base.json.

Run from project root:
    python -m unittest tests.test_rules -v
"""
import sys
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.input_layer import load_demo_profile
from rules.engine import (
    RuleEngine, load_rule_base,
    forward_chain, backward_chain, generate_explanation,
    _build_facts, _add_derived_facts, _eval_condition, _eval_rule,
)


# =============================================================================
# Helpers
# =============================================================================

def _facts(**overrides):
    base = {
        "name": "Test", "gender": 0,
        "attendance": 80.0, "confidence_level": 3,
        "stress_level": 2, "quiz_score_avg": 70.0,
        "workload_credits": 40, "workload_hours": 8.0,
        "workload_tasks": 2, "deadlines": 0,
        "daily_free_hours": 5.0,
        "available_days": ["Monday","Tuesday","Wednesday","Friday"],
        "tasks": [{"name":"Essay","subject":"Hist","hours":5,"deadline_days":8}],
        "exams": [],
        "planner_found_goal": True,
        "planner_not_scheduled": [],
        "planner_partial": [],
    }
    base.update(overrides)
    return base


# =============================================================================
# Rule base integrity
# =============================================================================

class TestRuleBase(unittest.TestCase):

    def setUp(self):
        self.rules = load_rule_base()

    def test_loads_without_error(self):
        self.assertIsInstance(self.rules, list)

    def test_39_rules_total(self):
        self.assertEqual(len(self.rules), 40)

    def test_layers_present(self):
        layers = {r["layer"] for r in self.rules}
        self.assertEqual(layers, {"O", "T", "D", "C"})

    def test_layer_counts(self):
        counts = {}
        for r in self.rules:
            counts[r["layer"]] = counts.get(r["layer"], 0) + 1
        self.assertEqual(counts["O"], 15)
        self.assertEqual(counts["T"], 10)
        self.assertEqual(counts["D"],  8)
        self.assertEqual(counts["C"],  7)

    def test_all_ids_unique(self):
        ids = [r["id"] for r in self.rules]
        self.assertEqual(len(ids), len(set(ids)))

    def test_required_keys_present(self):
        required = {"id","layer","description","conditions",
                    "conclusion","confidence","risk_vote","signal"}
        for rule in self.rules:
            missing = required - set(rule.keys())
            self.assertEqual(missing, set(),
                             f"Rule {rule['id']} missing: {missing}")

    def test_confidence_in_range(self):
        for rule in self.rules:
            self.assertGreater(rule["confidence"], 0,
                               f"Rule {rule['id']} confidence <= 0")
            self.assertLessEqual(rule["confidence"], 1.0,
                                 f"Rule {rule['id']} confidence > 1")

    def test_risk_votes_valid(self):
        valid = {"Low", "Medium", "High"}
        for rule in self.rules:
            self.assertIn(rule["risk_vote"], valid,
                          f"Rule {rule['id']} has invalid risk_vote")

    def test_json_is_valid(self):
        path = Path(__file__).resolve().parents[1] / "rules" / "rule_base.json"
        with open(path) as f:
            data = json.load(f)
        self.assertIn("rules", data)
        self.assertIn("meta", data)

    def test_meta_total_matches_actual(self):
        path = Path(__file__).resolve().parents[1] / "rules" / "rule_base.json"
        with open(path) as f:
            data = json.load(f)
        claimed = data["meta"]["total_rules"]
        actual  = len([r for r in data["rules"] if "id" in r])
        self.assertEqual(claimed, actual)   # meta must stay in sync with actual count


# =============================================================================
# Condition evaluator
# =============================================================================

class TestConditionEvaluator(unittest.TestCase):

    def test_numeric_gt(self):
        self.assertTrue(_eval_condition(
            {"fact": "attendance", "op": ">", "value": 50},
            {"attendance": 80}
        ))

    def test_numeric_lt_false(self):
        self.assertFalse(_eval_condition(
            {"fact": "attendance", "op": "<", "value": 50},
            {"attendance": 80}
        ))

    def test_is_true(self):
        self.assertTrue(_eval_condition(
            {"fact": "low_attendance", "op": "is_true"},
            {"low_attendance": True}
        ))

    def test_is_true_false_when_absent(self):
        self.assertFalse(_eval_condition(
            {"fact": "low_attendance", "op": "is_true"},
            {}
        ))

    def test_list_any_field(self):
        cond = {"list_fact": "exams", "any_field": "days_until",
                "op": "<=", "value": 4}
        self.assertTrue(_eval_condition(
            cond, {"exams": [{"days_until": 2}]}
        ))
        self.assertFalse(_eval_condition(
            cond, {"exams": [{"days_until": 14}]}
        ))

    def test_len_ge(self):
        self.assertTrue(_eval_condition(
            {"fact": "planner_not_scheduled", "op": "len_ge", "value": 1},
            {"planner_not_scheduled": ["Task1"]}
        ))
        self.assertFalse(_eval_condition(
            {"fact": "planner_not_scheduled", "op": "len_ge", "value": 1},
            {"planner_not_scheduled": []}
        ))

    def test_missing_fact_returns_false(self):
        self.assertFalse(_eval_condition(
            {"fact": "attendance", "op": ">", "value": 50},
            {}
        ))

    def test_eq_false_value(self):
        self.assertTrue(_eval_condition(
            {"fact": "planner_found_goal", "op": "==", "value": False},
            {"planner_found_goal": False}
        ))


# =============================================================================
# Derived facts
# =============================================================================

class TestDerivedFacts(unittest.TestCase):

    def test_free_hours_total_computed(self):
        f = _add_derived_facts({
            "daily_free_hours": 4.0,
            "available_days": ["Mon","Tue","Wed","Thu"],
        })
        self.assertAlmostEqual(f["free_hours_total"], 16.0)

    def test_hour_surplus_positive(self):
        f = _add_derived_facts({
            "workload_hours": 30.0,
            "daily_free_hours": 4.0,
            "available_days": ["Mon","Tue","Wed"],  # 12h free
        })
        self.assertGreater(f["hour_surplus"], 0)

    def test_hour_surplus_zero_when_enough_time(self):
        f = _add_derived_facts({
            "workload_hours": 5.0,
            "daily_free_hours": 5.0,
            "available_days": ["Mon","Tue","Wed","Thu","Fri"],  # 25h free
        })
        self.assertEqual(f["hour_surplus"], 0)


# =============================================================================
# OBSERVE layer — individual rule conditions
# =============================================================================

class TestObserveLayer(unittest.TestCase):

    def _run(self, **kwargs):
        return forward_chain(_facts(**kwargs))

    def _conclusions(self, **kwargs):
        return {c for _, c, _ in self._run(**kwargs)["rules_fired"]}

    def test_O1_low_attendance(self):
        self.assertIn("low_attendance", self._conclusions(attendance=65.0))

    def test_O2_very_low_attendance(self):
        self.assertIn("very_low_attendance", self._conclusions(attendance=45.0))

    def test_O3_low_confidence(self):
        self.assertIn("low_confidence", self._conclusions(confidence_level=2))

    def test_O4_high_stress(self):
        self.assertIn("high_stress", self._conclusions(stress_level=4))

    def test_O5_extreme_stress(self):
        self.assertIn("extreme_stress", self._conclusions(stress_level=5))

    def test_O6_low_quiz(self):
        self.assertIn("low_quiz", self._conclusions(quiz_score_avg=45.0))

    def test_O7_very_low_quiz(self):
        self.assertIn("very_low_quiz", self._conclusions(quiz_score_avg=30.0))

    def test_O8_urgent_workload(self):
        c = self._conclusions(deadlines=2, workload_hours=10.0)
        self.assertIn("urgent_workload", c)

    def test_O9_deadline_cluster(self):
        self.assertIn("deadline_cluster", self._conclusions(deadlines=4))

    def test_O10_exam_imminent(self):
        c = self._conclusions(exams=[{"subject":"Maths","days_until":3}])
        self.assertIn("exam_imminent", c)

    def test_O10_not_fired_distant_exam(self):
        c = self._conclusions(exams=[{"subject":"Maths","days_until":14}])
        self.assertNotIn("exam_imminent", c)

    def test_O11_hour_shortfall(self):
        # 40h needed, 3 days × 4h = 12h free → surplus = 28
        c = self._conclusions(
            workload_hours=40.0, daily_free_hours=4.0,
            available_days=["Monday","Tuesday","Wednesday"]
        )
        self.assertIn("hour_shortfall", c)

    def test_O12_schedule_overload(self):
        c = self._conclusions(
            planner_found_goal=False,
            planner_not_scheduled=["Essay","Lab"]
        )
        self.assertIn("schedule_overload", c)

    def test_O13_max_credit_load(self):
        self.assertIn("max_credit_load", self._conclusions(workload_credits=80))

    def test_O14_good_engagement(self):
        c = self._conclusions(attendance=90.0, confidence_level=5)
        self.assertIn("good_engagement", c)


# =============================================================================
# THINK layer — signal combination
# =============================================================================

class TestThinkLayer(unittest.TestCase):

    def _conclusions(self, **kwargs):
        return {c for _, c, _ in forward_chain(_facts(**kwargs))["rules_fired"]}

    def test_T1_engagement_risk(self):
        c = self._conclusions(attendance=60.0, confidence_level=2)
        self.assertIn("engagement_risk", c)

    def test_T2_academic_failure_risk(self):
        c = self._conclusions(attendance=45.0, quiz_score_avg=40.0)
        self.assertIn("academic_failure_risk", c)

    def test_T3_critical_overload(self):
        c = self._conclusions(
            deadlines=4,
            planner_found_goal=False,
            planner_not_scheduled=["T1","T2"]
        )
        self.assertIn("critical_overload", c)

    def test_T5_exam_failure_risk(self):
        c = self._conclusions(
            exams=[{"subject":"Maths","days_until":2}],
            quiz_score_avg=40.0
        )
        self.assertIn("exam_failure_risk", c)

    def test_T7_burnout_risk(self):
        c = self._conclusions(stress_level=4, quiz_score_avg=40.0)
        self.assertIn("burnout_risk", c)

    def test_T8_crisis_risk(self):
        c = self._conclusions(stress_level=5, deadlines=4)
        self.assertIn("crisis_risk", c)


# =============================================================================
# DECIDE layer — risk level conclusions
# =============================================================================

class TestDecideLayer(unittest.TestCase):

    def _run(self, **kwargs):
        return forward_chain(_facts(**kwargs))

    def test_D5_high_risk_from_schedule_overload(self):
        r = self._run(
            planner_found_goal=False,
            planner_not_scheduled=["Task1"]
        )
        self.assertEqual(r["risk_level"], "High")

    def test_D2_high_risk_from_academic_failure(self):
        r = self._run(attendance=45.0, quiz_score_avg=40.0)
        self.assertEqual(r["risk_level"], "High")

    def test_medium_risk_from_burnout(self):
        r = self._run(stress_level=4, quiz_score_avg=40.0)
        self.assertIn(r["risk_level"], ("Medium", "High"))

    def test_low_risk_perfect_student(self):
        r = self._run(
            attendance=95.0, confidence_level=5, stress_level=1,
            quiz_score_avg=90.0, workload_hours=5.0, deadlines=0,
            planner_found_goal=True,
        )
        self.assertIn(r["risk_level"], ("Low", "Medium"))


# =============================================================================
# CONFIRM layer — positive signals and downgrade
# =============================================================================

class TestConfirmLayer(unittest.TestCase):

    def _conclusions(self, **kwargs):
        return {c for _, c, _ in forward_chain(_facts(**kwargs))["rules_fired"]}

    def test_C1_strong_attendance(self):
        c = self._conclusions(attendance=90.0)
        self.assertIn("strong_attendance", c)

    def test_C2_strong_academic(self):
        c = self._conclusions(quiz_score_avg=80.0)
        self.assertIn("strong_academic", c)

    def test_C3_schedule_feasible(self):
        c = self._conclusions(planner_found_goal=True)
        self.assertIn("schedule_feasible", c)

    def test_C4_trusted_engagement(self):
        c = self._conclusions(attendance=90.0, confidence_level=5)
        self.assertIn("trusted_engagement", c)

    def test_C7_downgrade_fires(self):
        c = self._conclusions(
            attendance=95.0, confidence_level=5,
            planner_found_goal=True
        )
        self.assertIn("DOWNGRADE_RISK", c)


# =============================================================================
# Risk level derivation
# =============================================================================

class TestRiskLevelDerivation(unittest.TestCase):

    def test_demo_a_low_or_medium(self):
        pa = load_demo_profile("A")
        r  = forward_chain(pa, {"astar": {
            "found_goal": True, "not_scheduled": [], "partial": []
        }})
        self.assertIn(r["risk_level"], ("Low", "Medium"))

    def test_demo_b_high_risk(self):
        pb = load_demo_profile("B")
        r  = forward_chain(pb, {"astar": {
            "found_goal": False,
            "not_scheduled": ["Research paper","Stats prep"],
            "partial": []
        }})
        self.assertEqual(r["risk_level"], "High")

    def test_confidence_in_range(self):
        r = forward_chain(_facts())
        self.assertGreaterEqual(r["confidence"], 0.0)
        self.assertLessEqual(r["confidence"],    1.0)

    def test_report_has_all_keys(self):
        r = forward_chain(_facts())
        for key in ("risk_level","confidence","signals",
                    "rules_fired","recommendations","explanation"):
            self.assertIn(key, r)

    def test_signals_match_rules_fired(self):
        r = forward_chain(_facts(stress_level=5, deadlines=4))
        self.assertEqual(len(r["signals"]), len(r["rules_fired"]))


# =============================================================================
# Backward chaining
# =============================================================================

class TestBackwardChaining(unittest.TestCase):

    def test_returns_list(self):
        self.assertIsInstance(backward_chain("HIGH_RISK", _facts()), list)

    def test_detects_missing_attendance(self):
        p = _facts()
        del p["attendance"]
        gaps = backward_chain("HIGH_RISK", p)
        self.assertIn("attendance", [g["missing_fact"] for g in gaps])

    def test_detects_missing_stress(self):
        p = _facts()
        del p["stress_level"]
        gaps = backward_chain("HIGH_RISK", p)
        self.assertIn("stress_level", [g["missing_fact"] for g in gaps])

    def test_complete_profile_returns_empty(self):
        full = _facts(
            attendance=75.0, confidence_level=3, stress_level=3,
            quiz_score_avg=65.0, workload_hours=10.0, deadlines=2,
            exams=[{"subject":"X","days_until":7}],
            planner_found_goal=False,
            planner_not_scheduled=["T1"],
        )
        self.assertEqual(backward_chain("HIGH_RISK", full), [])

    def test_each_item_has_required_keys(self):
        p = _facts()
        del p["attendance"]
        for item in backward_chain("HIGH_RISK", p):
            for key in ("missing_fact","question","needed_by","to_prove"):
                self.assertIn(key, item)

    def test_needed_by_is_valid_rule_id(self):
        p = _facts()
        del p["confidence_level"]
        valid_ids = {r["id"] for r in load_rule_base()}
        for item in backward_chain("HIGH_RISK", p):
            self.assertIn(item["needed_by"], valid_ids)

    def test_medium_risk_goal(self):
        self.assertIsInstance(backward_chain("MEDIUM_RISK", _facts()), list)

    def test_low_risk_goal(self):
        self.assertIsInstance(backward_chain("LOW_RISK", _facts()), list)


# =============================================================================
# Explanation
# =============================================================================

class TestExplanation(unittest.TestCase):

    def test_is_string(self):
        self.assertIsInstance(forward_chain(_facts())["explanation"], str)

    def test_not_empty(self):
        r = forward_chain(_facts(stress_level=5))
        self.assertTrue(r["explanation"].strip())

    def test_contains_risk_level(self):
        r = forward_chain(_facts(
            attendance=40.0, confidence_level=1, deadlines=5,
            planner_found_goal=False,
            planner_not_scheduled=["T"],
        ))
        self.assertIn(r["risk_level"], r["explanation"])

    def test_contains_name(self):
        r = forward_chain(_facts(name="Jordan"))
        self.assertIn("Jordan", r["explanation"])

    def test_mentions_layers(self):
        r = forward_chain(_facts())
        self.assertIn("O→T→D→C", r["explanation"])


# =============================================================================
# Integration — full pipeline with both demo profiles
# =============================================================================

class TestIntegration(unittest.TestCase):

    def test_demo_a_full_pipeline(self):
        pa  = load_demo_profile("A")
        cmp = {"astar": {"found_goal": True, "not_scheduled": [],
                         "partial": [], "score": 0}}
        r   = forward_chain(pa, cmp)
        self.assertIn(r["risk_level"], ("Low", "Medium"))
        self.assertTrue(r["explanation"])
        self.assertGreater(len(r["rules_fired"]), 0)

    def test_demo_b_full_pipeline(self):
        pb  = load_demo_profile("B")
        cmp = {"astar": {"found_goal": False,
                         "not_scheduled": ["Research paper","Stats exam prep"],
                         "partial": ["Case study (2/8 h)"],
                         "score": 24.4}}
        r   = forward_chain(pb, cmp)
        self.assertEqual(r["risk_level"], "High")
        self.assertGreater(len(r["recommendations"]), 0)

    def test_layer_order_preserved(self):
        """O-layer conclusions must appear before T-layer conclusions."""
        r       = forward_chain(_facts(
            attendance=45.0, quiz_score_avg=40.0,
            planner_found_goal=False,
            planner_not_scheduled=["T1"]
        ))
        rules   = load_rule_base()
        id_to_layer = {rule["id"]: rule["layer"] for rule in rules}
        fired_layers = [id_to_layer[fid] for fid, _, _ in r["rules_fired"]
                        if fid in id_to_layer]
        order_map   = {"O": 0, "T": 1, "D": 2, "C": 3}
        for i in range(1, len(fired_layers)):
            self.assertGreaterEqual(
                order_map[fired_layers[i]],
                order_map[fired_layers[i-1]],
                f"Layer order violated: {fired_layers}"
            )


class TestOverdueRule(unittest.TestCase):
    """O15 — overdue task fires correctly."""

    def test_O15_fires_when_task_overdue(self):
        p = _facts()
        p["tasks"] = [{"name": "Late essay", "subject": "X",
                       "hours": 5, "deadline_days": 0, "overdue": True}]
        r = forward_chain(p)
        conclusions = {c for _, c, _ in r["rules_fired"]}
        self.assertIn("overdue_task", conclusions)

    def test_O15_does_not_fire_without_overdue_flag(self):
        p = _facts()
        p["tasks"] = [{"name": "Normal task", "subject": "X",
                       "hours": 5, "deadline_days": 3}]
        r = forward_chain(p)
        conclusions = {c for _, c, _ in r["rules_fired"]}
        self.assertNotIn("overdue_task", conclusions)

    def test_O15_risk_vote_is_high(self):
        rules = load_rule_base()
        o15   = next(r for r in rules if r["id"] == "O15")
        self.assertEqual(o15["risk_vote"], "High")
        self.assertGreaterEqual(o15["confidence"], 0.90)


if __name__ == "__main__":
    unittest.main(verbosity=2)
