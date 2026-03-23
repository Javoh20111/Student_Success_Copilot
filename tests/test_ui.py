"""
tests/test_ui.py
================
Unit tests for ui/input_layer.py and ui/display.py.
Uses only stdlib unittest — no external dependencies required.

Run from the project root:
    python -m unittest tests.test_ui -v
"""

import sys
import unittest
from pathlib import Path

# Make sure the project root is on the path regardless of where we run from
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.input_layer import (
    validate_inputs,
    compute_derived_fields,
    load_demo_profile,
    Issue,
)


# =============================================================================
# Helper
# =============================================================================

def _clean_profile(**overrides):
    """Return a fully valid base profile, with optional field overrides."""
    base = {
        "name": "Test", "gender": 0,
        "attendance": 80.0, "confidence_level": 3,
        "stress_level": 3,  "quiz_score_avg": 70.0,
        "workload_credits": 40,
        "tasks": [
            {"name": "Essay", "subject": "History",
             "hours": 5.0, "deadline_days": 8},
            {"name": "PS",    "subject": "Math",
             "hours": 3.0, "deadline_days": 12},
        ],
        "exams": [{"subject": "Math", "days_until": 15}],
        "fixed_activities": [],
        "available_days":   ["Monday", "Tuesday", "Wednesday", "Friday"],
        "daily_free_hours": 5.0,
    }
    base.update(overrides)
    return base


# =============================================================================
# Test cases
# =============================================================================

class TestDemoProfiles(unittest.TestCase):

    def test_demo_a_loads(self):
        pa = load_demo_profile("A")
        self.assertEqual(pa["name"], "Alex")
        self.assertEqual(pa["workload_tasks"], 4)
        self.assertEqual(pa["workload_hours"], 11.5)
        self.assertEqual(pa["deadlines"], 1)
        self.assertEqual(pa["availability_constraints"], 1)

    def test_demo_b_loads(self):
        pb = load_demo_profile("B")
        self.assertEqual(pb["name"], "Jordan")
        self.assertEqual(pb["workload_tasks"], 6)
        self.assertEqual(pb["workload_hours"], 45.0)
        self.assertEqual(pb["deadlines"], 5)
        self.assertEqual(pb["availability_constraints"], 2)
        self.assertEqual(pb["stress_level"], 5)

    def test_demo_invalid_scenario_raises(self):
        with self.assertRaises(ValueError):
            load_demo_profile("Z")


class TestValidationMissing(unittest.TestCase):

    def test_missing_name(self):
        issues = validate_inputs(_clean_profile(name=""))
        self.assertTrue(
            any(i.field == "name" and i.kind == "MISSING" for i in issues))

    def test_missing_gender(self):
        p = _clean_profile()
        del p["gender"]
        issues = validate_inputs(p)
        self.assertTrue(any(i.field == "gender" for i in issues))

    def test_missing_tasks(self):
        issues = validate_inputs(_clean_profile(tasks=[]))
        self.assertTrue(
            any(i.field == "tasks" and i.kind == "MISSING" for i in issues))

    def test_missing_available_days(self):
        issues = validate_inputs(_clean_profile(available_days=[]))
        self.assertTrue(any(i.field == "available_days" for i in issues))

    def test_missing_stress(self):
        p = _clean_profile()
        del p["stress_level"]
        issues = validate_inputs(p)
        self.assertTrue(any(i.field == "stress_level" for i in issues))

    def test_missing_daily_free_hours(self):
        p = _clean_profile()
        del p["daily_free_hours"]
        issues = validate_inputs(p)
        self.assertTrue(any(i.field == "daily_free_hours" for i in issues))


class TestValidationOutOfRange(unittest.TestCase):

    def test_attendance_over_100(self):
        issues = validate_inputs(_clean_profile(attendance=150.0))
        self.assertTrue(
            any(i.field == "attendance" and i.kind == "OUT_OF_RANGE"
                for i in issues))

    def test_attendance_negative(self):
        issues = validate_inputs(_clean_profile(attendance=-5.0))
        self.assertTrue(any(i.field == "attendance" for i in issues))

    def test_stress_over_5(self):
        issues = validate_inputs(_clean_profile(stress_level=9))
        self.assertTrue(any(i.field == "stress_level" for i in issues))

    def test_confidence_zero(self):
        issues = validate_inputs(_clean_profile(confidence_level=0))
        self.assertTrue(any(i.field == "confidence_level" for i in issues))

    def test_quiz_negative(self):
        issues = validate_inputs(_clean_profile(quiz_score_avg=-10.0))
        self.assertTrue(any(i.field == "quiz_score_avg" for i in issues))

    def test_quiz_over_100(self):
        issues = validate_inputs(_clean_profile(quiz_score_avg=110.0))
        self.assertTrue(any(i.field == "quiz_score_avg" for i in issues))

    def test_invalid_credits_non_multiple(self):
        issues = validate_inputs(_clean_profile(workload_credits=35))
        self.assertTrue(any(i.field == "workload_credits" for i in issues))

    def test_invalid_credits_over_80(self):
        issues = validate_inputs(_clean_profile(workload_credits=100))
        self.assertTrue(any(i.field == "workload_credits" for i in issues))

    def test_gender_invalid(self):
        issues = validate_inputs(_clean_profile(gender=5))
        self.assertTrue(any(i.field == "gender" for i in issues))

    def test_daily_free_hours_zero(self):
        issues = validate_inputs(_clean_profile(daily_free_hours=0))
        self.assertTrue(any(i.field == "daily_free_hours" for i in issues))


class TestValidationContradiction(unittest.TestCase):

    def test_overload_impossible_schedule(self):
        p = _clean_profile(
            tasks=[{"name": f"T{i}", "subject": "X",
                    "hours": 20, "deadline_days": 5}
                   for i in range(5)],       # 100 h of work
            available_days=["Monday", "Tuesday"],
            daily_free_hours=3.0,            # only 6 h free
        )
        issues = validate_inputs(p)
        contra = [i for i in issues if i.kind == "CONTRADICTION"]
        self.assertGreaterEqual(len(contra), 1)
        self.assertTrue(any(i.severity == "error" for i in contra))

    def test_high_confidence_low_attendance(self):
        issues = validate_inputs(
            _clean_profile(confidence_level=5, attendance=25.0))
        contra = [i for i in issues
                  if i.kind == "CONTRADICTION" and "confidence" in i.field]
        self.assertGreaterEqual(len(contra), 1)

    def test_urgent_tasks_low_stress(self):
        p = _clean_profile(
            stress_level=1,
            tasks=[{"name": "Due tomorrow", "subject": "X",
                    "hours": 4.0, "deadline_days": 1}],
        )
        issues = validate_inputs(p)
        contra = [i for i in issues
                  if i.kind == "CONTRADICTION" and "stress" in i.field]
        self.assertGreaterEqual(len(contra), 1)

    def test_low_quiz_high_attendance(self):
        issues = validate_inputs(
            _clean_profile(quiz_score_avg=30.0, attendance=90.0))
        contra = [i for i in issues
                  if i.kind == "CONTRADICTION" and "quiz" in i.field]
        self.assertGreaterEqual(len(contra), 1)


class TestCleanProfile(unittest.TestCase):

    def test_zero_issues_on_valid_profile(self):
        issues = validate_inputs(_clean_profile())
        self.assertEqual(issues, [])

    def test_all_valid_credits(self):
        for credits in (0, 20, 40, 60, 80):
            with self.subTest(credits=credits):
                issues = validate_inputs(_clean_profile(workload_credits=credits))
                self.assertFalse(
                    any(i.field == "workload_credits" for i in issues))

    def test_boundary_attendance(self):
        for att in (0.0, 50.0, 100.0):
            with self.subTest(att=att):
                issues = validate_inputs(_clean_profile(attendance=att))
                self.assertFalse(
                    any(i.field == "attendance" for i in issues))


class TestDerivedFields(unittest.TestCase):

    def test_workload_tasks_count(self):
        p = compute_derived_fields(_clean_profile())
        self.assertEqual(p["workload_tasks"], 2)

    def test_workload_hours_sum(self):
        p = compute_derived_fields(_clean_profile())
        self.assertEqual(p["workload_hours"], 8.0)

    def test_deadlines_count(self):
        # Both tasks have deadlines > 7 days — count should be 0
        p = compute_derived_fields(_clean_profile())
        self.assertEqual(p["deadlines"], 0)

    def test_deadline_count_with_urgent_task(self):
        p = _clean_profile(
            tasks=[
                {"name": "Urgent", "subject": "X",
                 "hours": 2.0, "deadline_days": 3},   # ≤ 7 → counted
                {"name": "Relaxed", "subject": "Y",
                 "hours": 3.0, "deadline_days": 14},  # > 7 → not counted
            ]
        )
        p = compute_derived_fields(p)
        self.assertEqual(p["deadlines"], 1)

    def test_availability_low_constraint(self):
        # 5 h/day → constraint = 1 (low)
        p = compute_derived_fields(_clean_profile(daily_free_hours=5.0))
        self.assertEqual(p["availability_constraints"], 1)

    def test_availability_medium_constraint(self):
        # 3 h/day → constraint = 2 (medium)
        p = compute_derived_fields(_clean_profile(daily_free_hours=3.0))
        self.assertEqual(p["availability_constraints"], 2)

    def test_availability_high_constraint(self):
        # 1.5 h/day → constraint = 3 (high)
        p = compute_derived_fields(_clean_profile(daily_free_hours=1.5))
        self.assertEqual(p["availability_constraints"], 3)


class TestIssueDataclass(unittest.TestCase):

    def test_issue_fields(self):
        i = Issue(
            kind="MISSING", field="name",
            message="name is missing", question="What is your name?",
            severity="error",
        )
        self.assertEqual(i.kind, "MISSING")
        self.assertEqual(i.severity, "error")

    def test_issue_warning_severity(self):
        i = Issue(
            kind="CONTRADICTION", field="stress_level",
            message="low stress but urgent tasks",
            question="Is your stress correct?",
            severity="warning",
        )
        self.assertEqual(i.severity, "warning")


# =============================================================================
# Entry point — run directly with: python tests/test_ui.py
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
