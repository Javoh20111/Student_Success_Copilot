"""
tests/test_pipeline.py
======================
Integration tests: run both demo scenarios end-to-end through
the full pipeline (planner → rules → ML) and assert the combined
result is structurally correct and internally consistent.

Run from project root:
    python -m unittest tests.test_pipeline -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.pipeline import run_pipeline, run_demo, train_once
from ui.input_layer import load_demo_profile


# Train models once for the whole test module
_MODELS = None

def _get_models():
    global _MODELS
    if _MODELS is None:
        _MODELS = train_once()
    return _MODELS


class TestPipelineStructure(unittest.TestCase):
    """Result dict has the right shape for both scenarios."""

    def _run(self, scenario):
        profile = load_demo_profile(scenario)
        return run_pipeline(profile, models=_get_models(), verbose=False)

    def test_result_has_all_keys(self):
        r = self._run("A")
        for key in ("profile", "planner", "report", "ml_predictions"):
            self.assertIn(key, r)

    def test_planner_has_bfs_and_astar(self):
        r = self._run("A")
        self.assertIn("bfs",   r["planner"])
        self.assertIn("astar", r["planner"])

    def test_report_has_all_keys(self):
        r = self._run("A")
        for key in ("risk_level","confidence","signals",
                    "rules_fired","recommendations","explanation"):
            self.assertIn(key, r["report"])

    def test_ml_predictions_not_empty(self):
        r = self._run("A")
        self.assertGreater(len(r["ml_predictions"]), 0)

    def test_ml_prediction_fields(self):
        r = self._run("A")
        for _, pred in r["ml_predictions"].items():
            self.assertIn(pred["prediction"], ("Pass", "Fail"))
            self.assertGreaterEqual(pred["confidence"], 0.5)


class TestDemoA(unittest.TestCase):
    """Demo A: Alex, manageable workload — expect low or medium risk."""

    @classmethod
    def setUpClass(cls):
        profile = load_demo_profile("A")
        cls.r = run_pipeline(profile, models=_get_models(), verbose=False)

    def test_planner_reaches_goal(self):
        self.assertTrue(self.r["planner"]["astar"]["found_goal"])

    def test_planner_no_conflicts(self):
        self.assertEqual(self.r["planner"]["astar"]["conflicts"], 0)

    def test_planner_no_unscheduled(self):
        self.assertEqual(self.r["planner"]["astar"]["not_scheduled"], [])

    def test_risk_level_low_or_medium(self):
        self.assertIn(self.r["report"]["risk_level"], ("Low", "Medium"))

    def test_rules_fired(self):
        self.assertGreater(len(self.r["report"]["rules_fired"]), 0)

    def test_explanation_not_empty(self):
        self.assertTrue(self.r["report"]["explanation"].strip())

    def test_confidence_in_range(self):
        conf = self.r["report"]["confidence"]
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)


class TestDemoB(unittest.TestCase):
    """Demo B: Jordan, overloaded — expect high risk."""

    @classmethod
    def setUpClass(cls):
        profile = load_demo_profile("B")
        cls.r = run_pipeline(profile, models=_get_models(), verbose=False)

    def test_planner_cannot_complete(self):
        self.assertFalse(self.r["planner"]["astar"]["found_goal"])

    def test_planner_has_unscheduled(self):
        # Either not_scheduled or partial must be non-empty
        ns = self.r["planner"]["astar"].get("not_scheduled", [])
        pt = self.r["planner"]["astar"].get("partial", [])
        self.assertTrue(len(ns) + len(pt) > 0)

    def test_risk_level_high(self):
        self.assertEqual(self.r["report"]["risk_level"], "High")

    def test_recommendations_not_empty(self):
        self.assertGreater(len(self.r["report"]["recommendations"]), 0)

    def test_explanation_mentions_immediate_action(self):
        self.assertIn("Immediate action", self.r["report"]["explanation"])

    def test_ml_predictions_present(self):
        self.assertIsNotNone(self.r["ml_predictions"])
        self.assertGreater(len(self.r["ml_predictions"]), 0)


class TestPipelineConsistency(unittest.TestCase):
    """Cross-component consistency checks."""

    def test_planner_overload_triggers_rules_high_risk(self):
        """When planner fails, rules engine must flag HIGH risk."""
        profile = load_demo_profile("B")
        r = run_pipeline(profile, models=_get_models(), verbose=False)
        if not r["planner"]["astar"]["found_goal"]:
            self.assertEqual(r["report"]["risk_level"], "High")

    def test_planner_success_allows_low_risk(self):
        """When planner succeeds, risk may be Low or Medium (never forced High)."""
        profile = load_demo_profile("A")
        r = run_pipeline(profile, models=_get_models(), verbose=False)
        if r["planner"]["astar"]["found_goal"]:
            self.assertIn(r["report"]["risk_level"], ("Low", "Medium"))

    def test_models_reused_across_scenarios(self):
        """Running both scenarios with same models object produces consistent results."""
        models = _get_models()
        pa = load_demo_profile("A")
        pb = load_demo_profile("B")
        ra = run_pipeline(pa, models=models, verbose=False)
        rb = run_pipeline(pb, models=models, verbose=False)
        # A should be lower risk than B
        risk_order = {"Low": 0, "Medium": 1, "High": 2}
        self.assertLessEqual(
            risk_order.get(ra["report"]["risk_level"], 0),
            risk_order.get(rb["report"]["risk_level"], 2),
        )

    def test_same_scenario_deterministic(self):
        """Running the same scenario twice produces the same risk level."""
        models = _get_models()
        profile = load_demo_profile("A")
        r1 = run_pipeline(profile, models=models, verbose=False)
        r2 = run_pipeline(profile, models=models, verbose=False)
        self.assertEqual(r1["report"]["risk_level"],
                         r2["report"]["risk_level"])


class TestMixedSignals(unittest.TestCase):
    """_detect_mixed_signals produces the right verdict in each conflict case."""

    def _report(self, risk="Low", conf=0.7, n_fired=3):
        return {"risk_level": risk, "confidence": conf,
                "rules_fired": [("R1","X",0.8)] * n_fired,
                "signals": [], "recommendations": []}

    def _planner(self, goal=True):
        return {"astar": {"found_goal": goal, "not_scheduled": [] if goal else ["T1"],
                          "partial": [], "completed": []}}

    def _ml(self, verdict="Pass"):
        return {"DT": {"prediction": verdict, "confidence": 0.9, "probability": [0.1, 0.9]}}

    def test_no_conflict_both_agree_pass(self):
        from ui.pipeline import _detect_mixed_signals
        r = _detect_mixed_signals(self._report("Low"), self._planner(True), self._ml("Pass"))
        self.assertFalse(r["has_conflict"])

    def test_no_conflict_both_agree_fail(self):
        from ui.pipeline import _detect_mixed_signals
        r = _detect_mixed_signals(self._report("High"), self._planner(False), self._ml("Fail"))
        self.assertFalse(r["has_conflict"])

    def test_high_severity_ml_pass_rules_high(self):
        from ui.pipeline import _detect_mixed_signals
        r = _detect_mixed_signals(self._report("High", conf=0.75),
                                   self._planner(True), self._ml("Pass"))
        self.assertTrue(r["has_conflict"])
        self.assertEqual(r["severity"], "high")
        self.assertIn("Mixed signals", r["message"])
        self.assertLess(r["adjusted_confidence"], 0.75)

    def test_high_severity_ml_pass_planner_failed(self):
        from ui.pipeline import _detect_mixed_signals
        r = _detect_mixed_signals(self._report("Medium", conf=0.65),
                                   self._planner(False), self._ml("Pass"))
        self.assertTrue(r["has_conflict"])
        self.assertEqual(r["severity"], "high")

    def test_medium_severity_ml_fail_rules_low(self):
        from ui.pipeline import _detect_mixed_signals
        r = _detect_mixed_signals(self._report("Low", conf=0.6),
                                   self._planner(True), self._ml("Fail"))
        self.assertTrue(r["has_conflict"])
        self.assertEqual(r["severity"], "medium")

    def test_confidence_is_adjusted_downward(self):
        from ui.pipeline import _detect_mixed_signals
        r = _detect_mixed_signals(self._report("High", conf=0.80),
                                   self._planner(True), self._ml("Pass"))
        self.assertLess(r["adjusted_confidence"], 0.80)
        self.assertGreaterEqual(r["adjusted_confidence"], 0.30)


class TestOverdueValidation(unittest.TestCase):
    """Negative deadline is clamped to 0 and flagged overdue."""

    def test_negative_deadline_clamped(self):
        from ui.input_layer import validate_inputs
        p = {
            "name": "Test", "gender": 0, "attendance": 80.0,
            "confidence_level": 3, "stress_level": 2, "quiz_score_avg": 70.0,
            "workload_credits": 40,
            "tasks": [{"name": "Late essay", "subject": "X",
                       "hours": 5.0, "deadline_days": -3}],
            "exams": [], "fixed_activities": [],
            "available_days": ["Monday","Tuesday","Wednesday"],
            "daily_free_hours": 5.0,
        }
        validate_inputs(p)
        # After validation the task deadline should be clamped to 0
        self.assertEqual(p["tasks"][0]["deadline_days"], 0)
        self.assertTrue(p["tasks"][0].get("overdue"))

    def test_negative_deadline_raises_contradiction_not_oor(self):
        from ui.input_layer import validate_inputs
        p = {
            "name": "Test", "gender": 0, "attendance": 80.0,
            "confidence_level": 3, "stress_level": 2, "quiz_score_avg": 70.0,
            "workload_credits": 40,
            "tasks": [{"name": "Late essay", "subject": "X",
                       "hours": 5.0, "deadline_days": -5}],
            "exams": [], "fixed_activities": [],
            "available_days": ["Monday","Tuesday"],
            "daily_free_hours": 5.0,
        }
        issues = validate_inputs(p)
        kinds = {i.kind for i in issues}
        self.assertIn("CONTRADICTION", kinds)
        self.assertNotIn("OUT_OF_RANGE", kinds)

    def test_overdue_task_shown_in_display(self):
        """display_profile should not crash with overdue task."""
        from ui.input_layer import load_demo_profile
        from ui.display import display_profile
        import io, sys
        pa = load_demo_profile("A")
        pa["tasks"][0]["overdue"] = True
        pa["tasks"][0]["deadline_days"] = 0
        buf = io.StringIO()
        sys.stdout = buf
        try:
            display_profile(pa)
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn("OVERDUE", buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
