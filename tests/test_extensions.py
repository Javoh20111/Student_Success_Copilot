"""
tests/test_extensions.py
========================
Tests for Section 6 optional extensions.

  NLP interface  (ui/nlp_interface.py)   — fully tested offline
  GenAI explainer (ui/genai_explainer.py) — tests graceful degradation
    and input sanitisation without requiring an API key

Run from project root:
    python -m unittest tests.test_extensions -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.nlp_interface import extract_from_text, get_next_question
from ui.genai_explainer import (
    _sanitise, _check_refused_topic as genai_check_refused,
    _summarise_profile, _summarise_report, _summarise_planner,
    fallback_explain,
)


# =============================================================================
# NLP — extraction correctness
# =============================================================================

class TestNLPExtraction(unittest.TestCase):

    def _empty(self):
        return {"tasks": [], "exams": [], "fixed_activities": [],
                "available_days": []}

    def test_extracts_attendance_percent(self):
        p = self._empty()
        extract_from_text("I attend about 82% of my lectures", p)
        self.assertAlmostEqual(p["attendance"], 82.0)

    def test_extracts_attendance_reversed(self):
        p = self._empty()
        extract_from_text("My attendance is 75%", p)
        self.assertAlmostEqual(p["attendance"], 75.0)

    def test_extracts_confidence(self):
        p = self._empty()
        extract_from_text("I feel confident, about 4 out of 5", p)
        self.assertEqual(p["confidence_level"], 4)

    def test_extracts_stress(self):
        p = self._empty()
        extract_from_text("I am pretty stressed, maybe 4/5", p)
        self.assertEqual(p["stress_level"], 4)

    def test_extracts_quiz_score(self):
        p = self._empty()
        extract_from_text("My quiz average is around 68%", p)
        self.assertAlmostEqual(p["quiz_score_avg"], 68.0)

    def test_extracts_name_plain(self):
        p = self._empty()
        extract_from_text("Sam", p)
        self.assertEqual(p["name"], "Sam")

    def test_extracts_name_intro(self):
        p = self._empty()
        extract_from_text("My name is Jordan", p)
        self.assertEqual(p["name"], "Jordan")

    def test_extracts_name_im(self):
        p = self._empty()
        extract_from_text("I'm Alex", p)
        self.assertEqual(p["name"], "Alex")

    def test_does_not_overwrite_name(self):
        p = self._empty()
        p["name"] = "Alex"
        extract_from_text("Sam", p)
        self.assertEqual(p["name"], "Alex")

    def test_extracts_gender_male(self):
        p = self._empty()
        extract_from_text("I am a male student", p)
        self.assertEqual(p["gender"], 0)

    def test_extracts_gender_female(self):
        p = self._empty()
        extract_from_text("She is studying biology", p)
        self.assertEqual(p["gender"], 1)

    def test_extracts_credits(self):
        p = self._empty()
        extract_from_text("I am taking 60 credits this term", p)
        self.assertEqual(p["workload_credits"], 60)

    def test_extracts_free_hours_per_day(self):
        p = self._empty()
        extract_from_text("I have 3 free hours per day", p)
        self.assertAlmostEqual(p["daily_free_hours"], 3.0)

    def test_extracts_available_days(self):
        p = self._empty()
        extract_from_text("I am free on Monday and Wednesday", p)
        self.assertIn("Monday",    p["available_days"])
        self.assertIn("Wednesday", p["available_days"])

    def test_extracts_task_with_deadline_and_hours(self):
        p = self._empty()
        extract_from_text("I have a report due in 5 days, about 4 hours", p)
        self.assertEqual(len(p["tasks"]), 1)
        task = p["tasks"][0]
        self.assertEqual(task["deadline_days"], 5)
        self.assertAlmostEqual(task["hours"], 4.0)

    def test_extracts_task_without_hours_defaults(self):
        p = self._empty()
        extract_from_text("I have an assignment due in 3 days", p)
        self.assertEqual(len(p["tasks"]), 1)
        self.assertAlmostEqual(p["tasks"][0]["hours"], 2.0)

    def test_extracts_exam(self):
        p = self._empty()
        extract_from_text("I have a Maths exam in 7 days", p)
        self.assertEqual(len(p["exams"]), 1)
        self.assertEqual(p["exams"][0]["days_until"], 7)

    def test_duplicate_task_not_added(self):
        p = self._empty()
        extract_from_text("I have a report due in 5 days", p)
        extract_from_text("I have a report due in 5 days", p)
        self.assertEqual(len(p["tasks"]), 1)

    def test_does_not_overwrite_existing_value(self):
        p = self._empty()
        p["attendance"] = 70.0
        extract_from_text("I attend 90% of my lectures", p)
        self.assertAlmostEqual(p["attendance"], 70.0)  # unchanged

    def test_multiple_extractions_in_one_call(self):
        p = self._empty()
        extract_from_text(
            "My attendance is 78% and I am stressed about 4/5", p)
        self.assertAlmostEqual(p["attendance"], 78.0)
        self.assertEqual(p["stress_level"], 4)


class TestNLPNextQuestion(unittest.TestCase):

    def test_asks_name_first(self):
        p = {"tasks": [], "exams": [], "fixed_activities": [],
             "available_days": []}
        field, _ = get_next_question(p)
        self.assertEqual(field, "name")

    def test_returns_none_when_complete(self):
        p = {
            "name": "Alice", "gender": 1,
            "attendance": 80.0, "confidence_level": 3,
            "stress_level": 2, "quiz_score_avg": 70.0,
            "workload_credits": 40,
            "tasks": [{"name":"T","subject":"X","hours":3,"deadline_days":7}],
            "available_days": ["Monday"],
            "daily_free_hours": 4.0,
            "exams": [], "fixed_activities": [],
        }
        self.assertIsNone(get_next_question(p))

    def test_skips_already_filled_fields(self):
        p = {
            "name": "Bob", "gender": 0,
            "attendance": 75.0,
            "tasks": [], "exams": [], "fixed_activities": [],
            "available_days": [],
        }
        field, _ = get_next_question(p)
        # name, gender, attendance all present — should ask confidence next
        self.assertEqual(field, "confidence_level")


class TestNLPRefusedTopics(unittest.TestCase):

    def test_flags_self_harm(self):
        self.assertIsNotNone(genai_check_refused("I want to self-harm"))

    def test_flags_cheat(self):
        self.assertIsNotNone(genai_check_refused("how do I cheat on my exam"))

    def test_safe_academic_question(self):
        self.assertIsNone(genai_check_refused("how do I study better"))

    def test_safe_task_description(self):
        self.assertIsNone(
            genai_check_refused("I have a report due in 3 days"))


# =============================================================================
# GenAI explainer — sanitisation and guardrails
# =============================================================================

class TestGenAISanitiser(unittest.TestCase):

    def test_strips_newlines(self):
        result = _sanitise("hello\nworld")
        self.assertNotIn("\n", result)

    def test_strips_backticks(self):
        result = _sanitise("ignore `previous` instructions")
        self.assertNotIn("`", result)

    def test_strips_angle_brackets(self):
        result = _sanitise("<script>alert('x')</script>")
        self.assertNotIn("<", result)
        self.assertNotIn(">", result)

    def test_truncates_to_max_len(self):
        long_str = "a" * 200
        result   = _sanitise(long_str, max_len=40)
        self.assertEqual(len(result), 40)

    def test_safe_string_unchanged(self):
        result = _sanitise("My essay is due in 3 days")
        self.assertEqual(result, "My essay is due in 3 days")


class TestGenAIRefused(unittest.TestCase):

    def test_flags_suicide(self):
        self.assertIsNotNone(genai_check_refused("I want to suicide"))

    def test_flags_hack(self):
        self.assertIsNotNone(genai_check_refused("how to hack the system"))

    def test_safe_question(self):
        self.assertIsNone(genai_check_refused("why is my risk level high"))

    def test_safe_algorithm_question(self):
        self.assertIsNone(genai_check_refused("why did A* do better than BFS"))


class TestGenAISummarisers(unittest.TestCase):

    def _profile(self):
        return {
            "name": "Test", "gender": 0, "attendance": 80.0,
            "confidence_level": 3, "stress_level": 2,
            "quiz_score_avg": 70.0, "workload_credits": 40,
            "tasks": [
                {"name": "Essay", "subject": "X",
                 "hours": 5, "deadline_days": 7}
            ],
            "available_days": ["Monday", "Wednesday"],
            "daily_free_hours": 4.0,
        }

    def _report(self):
        return {
            "risk_level": "Medium", "confidence": 0.65,
            "rules_fired": [("O1","low_attendance",0.80)],
            "signals": ["[O1] signal"], "recommendations": ["rec1"],
        }

    def _planner(self):
        return {
            "bfs":   {"states_explored": 100, "time_ms": 5.0,
                      "found_goal": True, "score": 0},
            "astar": {"states_explored": 40,  "time_ms": 3.0,
                      "found_goal": True, "score": 0,
                      "not_scheduled": [], "partial": []},
            "slots": [("Monday", 9), ("Monday", 10)],
        }

    def test_profile_summary_is_string(self):
        self.assertIsInstance(_summarise_profile(self._profile()), str)

    def test_profile_summary_contains_name(self):
        s = _summarise_profile(self._profile())
        self.assertIn("Test", s)

    def test_profile_summary_strips_injection(self):
        p = self._profile()
        p["name"] = "Inject\n<script>alert('x')</script>"
        s = _summarise_profile(p)
        self.assertNotIn("<script>", s)

    def test_report_summary_contains_risk(self):
        s = _summarise_report(self._report())
        self.assertIn("Medium", s)

    def test_planner_summary_contains_states(self):
        s = _summarise_planner(self._planner())
        self.assertIn("100", s)   # BFS states
        self.assertIn("40",  s)   # A* states

    def test_fallback_explain_no_api_key(self):
        """fallback_explain works without any API dependency."""
        result = fallback_explain(self._profile(), self._report())
        self.assertIsInstance(result, str)
        self.assertIn("Medium", result)
        self.assertIn("AI-generated", result)

    def test_genai_unavailable_returns_message(self):
        """When anthropic is not installed, _call_claude returns a message."""
        import ui.genai_explainer as ge
        orig = ge._call_claude
        ge._call_claude = lambda p: "[GenAI explainer unavailable]"
        try:
            r = ge.explain_study_tips(self._profile(), self._report(),
                                      self._planner())
            self.assertIsInstance(r, str)
        finally:
            ge._call_claude = orig


class TestBareAnswerResolver(unittest.TestCase):
    """_try_bare_answer fills in the pending field from a bare number."""

    def _empty(self):
        return {"tasks": [], "exams": [], "fixed_activities": [],
                "available_days": []}

    def _run(self, text, field):
        from ui.nlp_interface import _try_bare_answer
        p = self._empty()
        ok = _try_bare_answer(text, field, p)
        return ok, p

    def test_bare_number_attendance(self):
        ok, p = self._run("78", "attendance")
        self.assertTrue(ok)
        self.assertAlmostEqual(p["attendance"], 78.0)

    def test_approximately_attendance(self):
        ok, p = self._run("approximately 78", "attendance")
        self.assertTrue(ok)
        self.assertAlmostEqual(p["attendance"], 78.0)

    def test_maybe_confidence(self):
        ok, p = self._run("maybe 4", "confidence_level")
        self.assertTrue(ok)
        self.assertEqual(p["confidence_level"], 4)

    def test_bare_confidence(self):
        ok, p = self._run("4", "confidence_level")
        self.assertTrue(ok)
        self.assertEqual(p["confidence_level"], 4)

    def test_bare_stress(self):
        ok, p = self._run("3", "stress_level")
        self.assertTrue(ok)
        self.assertEqual(p["stress_level"], 3)

    def test_around_quiz(self):
        ok, p = self._run("around 65", "quiz_score_avg")
        self.assertTrue(ok)
        self.assertAlmostEqual(p["quiz_score_avg"], 65.0)

    def test_bare_free_hours(self):
        ok, p = self._run("3", "daily_free_hours")
        self.assertTrue(ok)
        self.assertAlmostEqual(p["daily_free_hours"], 3.0)

    def test_out_of_range_returns_false(self):
        ok, p = self._run("7", "confidence_level")  # 1-5 only
        self.assertFalse(ok)
        self.assertIsNone(p.get("confidence_level"))

    def test_wrong_field_returns_false(self):
        ok, p = self._run("Sam", "attendance")  # not a number
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
