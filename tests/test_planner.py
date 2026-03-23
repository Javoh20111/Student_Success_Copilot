"""
tests/test_planner.py
=====================
Unit tests for planner/planner.py.

Run from the project root:
    python -m unittest tests.test_planner -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.input_layer import load_demo_profile
from planner.planner import (
    build_available_slots,
    bfs_planner,
    astar_planner,
    compare_planners,
    display_schedule,
    build_tight_comparison_profile,
    _initial_state,
    _heuristic,
    _urgency_weight,
    State,
)


class TestSlotBuilder(unittest.TestCase):

    def setUp(self):
        self.pa = load_demo_profile("A")
        self.pb = load_demo_profile("B")

    def test_demo_a_slots_positive(self):
        slots = build_available_slots(self.pa)
        self.assertGreater(len(slots), 0)

    def test_demo_a_slots_under_cap(self):
        # 5 days × 5 h = 25 max
        slots = build_available_slots(self.pa)
        self.assertLessEqual(len(slots), 25)

    def test_fixed_activity_blocked(self):
        # Demo A: Monday 9-11 is a lecture
        slots = build_available_slots(self.pa)
        self.assertNotIn(("Monday", 9),  slots)
        self.assertNotIn(("Monday", 10), slots)

    def test_slot_format(self):
        slots = build_available_slots(self.pa)
        for day, hour in slots:
            self.assertIsInstance(day,  str)
            self.assertIsInstance(hour, int)
            self.assertGreaterEqual(hour, 8)
            self.assertLess(hour, 21)

    def test_only_available_days(self):
        slots = build_available_slots(self.pa)
        days_in_slots = {d for d, _ in slots}
        for day in days_in_slots:
            self.assertIn(day, self.pa["available_days"])

    def test_demo_b_slots_positive(self):
        slots = build_available_slots(self.pb)
        self.assertGreater(len(slots), 0)


class TestInitialState(unittest.TestCase):

    def setUp(self):
        self.pa = load_demo_profile("A")

    def test_slot_idx_zero(self):
        s = _initial_state(self.pa["tasks"])
        self.assertEqual(s.slot_idx, 0)

    def test_all_tasks_in_remaining(self):
        s = _initial_state(self.pa["tasks"])
        self.assertEqual(len(s.remaining), len(self.pa["tasks"]))

    def test_not_goal_at_start(self):
        s = _initial_state(self.pa["tasks"])
        self.assertFalse(s.is_goal())

    def test_all_hours_positive(self):
        s = _initial_state(self.pa["tasks"])
        for _, h in s.remaining:
            self.assertGreater(h, 0)


class TestHeuristic(unittest.TestCase):

    def setUp(self):
        self.pa = load_demo_profile("A")

    def test_urgent_weight_greater(self):
        t_urgent  = {"name": "U", "hours": 3, "deadline_days": 1}
        t_relaxed = {"name": "R", "hours": 3, "deadline_days": 14}
        self.assertGreater(_urgency_weight(t_urgent), _urgency_weight(t_relaxed))

    def test_heuristic_positive_at_start(self):
        s  = _initial_state(self.pa["tasks"])
        h0 = _heuristic(s, self.pa["tasks"])
        self.assertGreater(h0, 0)

    def test_heuristic_zero_when_done(self):
        done_tasks = [{"name": "T", "hours": 0, "deadline_days": 5}]
        s = _initial_state(done_tasks)
        self.assertEqual(_heuristic(s, done_tasks), 0.0)


class TestBFS(unittest.TestCase):

    def setUp(self):
        self.pa = load_demo_profile("A")
        self.pb = load_demo_profile("B")

    def test_returns_dict(self):
        r = bfs_planner(self.pa)
        self.assertIsInstance(r, dict)

    def test_required_keys(self):
        r = bfs_planner(self.pa)
        for key in ("schedule", "states_explored", "time_ms",
                    "found_goal", "conflicts", "score", "unscheduled"):
            self.assertIn(key, r)

    def test_demo_a_reaches_goal(self):
        r = bfs_planner(self.pa)
        self.assertTrue(r["found_goal"])

    def test_demo_a_no_conflicts(self):
        r = bfs_planner(self.pa)
        self.assertEqual(r["conflicts"], 0)

    def test_demo_a_finishes_fast(self):
        r = bfs_planner(self.pa)
        self.assertLess(r["time_ms"], 10_000)

    def test_demo_b_cannot_fully_schedule(self):
        r = bfs_planner(self.pb)
        self.assertFalse(r["found_goal"])

    def test_demo_b_has_unscheduled(self):
        r = bfs_planner(self.pb)
        self.assertGreater(len(r["unscheduled"]), 0)

    def test_schedule_entries_are_tuples(self):
        r = bfs_planner(self.pa)
        for entry in r["schedule"]:
            self.assertEqual(len(entry), 3)   # (task_name, day, hour)


class TestAStar(unittest.TestCase):

    def setUp(self):
        self.pa = load_demo_profile("A")
        self.pb = load_demo_profile("B")

    def test_demo_a_reaches_goal(self):
        r = astar_planner(self.pa)
        self.assertTrue(r["found_goal"])

    def test_demo_a_no_conflicts(self):
        r = astar_planner(self.pa)
        self.assertEqual(r["conflicts"], 0)

    def test_demo_a_finishes_fast(self):
        r = astar_planner(self.pa)
        self.assertLess(r["time_ms"], 10_000)

    def test_score_le_bfs(self):
        bfs   = bfs_planner(self.pa)
        astar = astar_planner(self.pa)
        self.assertLessEqual(astar["score"], bfs["score"])

    def test_demo_b_cannot_fully_schedule(self):
        r = astar_planner(self.pb)
        self.assertFalse(r["found_goal"])

    def test_demo_b_unscheduled_le_bfs(self):
        bfs   = bfs_planner(self.pb)
        astar = astar_planner(self.pb)
        self.assertLessEqual(len(astar["unscheduled"]), len(bfs["unscheduled"]))


class TestTightComparison(unittest.TestCase):
    """
    The tight profile has tasks in reversed urgency order.
    BFS tries them in input order (relaxed first).
    A* should place the urgent task on Monday (the only day before deadline).
    """

    def setUp(self):
        self.pt   = build_tight_comparison_profile()
        self.bfs  = bfs_planner(self.pt)
        self.astr = astar_planner(self.pt)

    def test_astar_schedules_urgent_on_monday(self):
        monday_tasks = [
            name for name, day, _ in self.astr["schedule"]
            if day == "Monday"
        ]
        self.assertIn("Past paper", monday_tasks,
                      f"A* Monday assignments: {monday_tasks}")

    def test_bfs_misses_urgent_on_monday(self):
        monday_tasks = [
            name for name, day, _ in self.bfs["schedule"]
            if day == "Monday"
        ]
        # BFS takes the first task in input order (Background reading)
        self.assertNotIn("Past paper", monday_tasks,
                         "BFS should NOT place urgent task on Monday "
                         f"(it got: {monday_tasks})")

    def test_astar_score_le_bfs(self):
        self.assertLessEqual(self.astr["score"], self.bfs["score"])


class TestComparePlanners(unittest.TestCase):

    def test_returns_both_keys(self):
        pa  = load_demo_profile("A")
        cmp = compare_planners(pa)
        self.assertIn("bfs",   cmp)
        self.assertIn("astar", cmp)
        self.assertIn("slots", cmp)

    def test_display_schedule_runs(self):
        pa  = load_demo_profile("A")
        cmp = compare_planners(pa)
        # Should not raise
        display_schedule(cmp["astar"], strategy="A*")
        display_schedule(cmp["bfs"],   strategy="BFS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
