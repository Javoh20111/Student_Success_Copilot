"""
planner/planner.py
==================
Search-based weekly study scheduler.

Problem formulation
-------------------
State   : (slot_idx, remaining_hours_per_task)
            slot_idx  — index into the ordered free-slot list
            remaining — tuple of (task_name, hours_left) sorted by name
Action  : assign current slot to task T  |  skip (IDLE)
Goal    : all tasks have hours_left == 0
Cost    : g(n) = cumulative urgency penalty of assigned slots

BFS  — FIFO queue, input order, finds first valid schedule.
A*   — min-heap on f(n)=g(n)+h(n), urgency-aware, schedules urgent first.

Both share the same slot builder, successor generator, and result format.
"""

from __future__ import annotations

import heapq
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ui.display import print_subheader

# ── Constants ──────────────────────────────────────────────────────────────────
STUDY_START  = 8     # 08:00
STUDY_END    = 21    # slots run 08:00 → 20:00  (last = 20–21)
MAX_STATES   = 25_000

_DAY_ORDER = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday",
]


# =============================================================================
# 1.  Slot builder
# =============================================================================

def build_available_slots(profile: Dict[str, Any]) -> List[Tuple[str, int]]:
    """
    Return an ordered list of (day, hour) free 1-hour study slots.

    Rules
    -----
    - Only days in profile['available_days']
    - Hours STUDY_START to STUDY_END-1
    - Slots overlapping fixed_activities are blocked
    - Each day contributes at most floor(daily_free_hours) slots
    - Order: Monday first, then earliest hour first within each day
    """
    available   = set(profile.get("available_days", []))
    max_per_day = int(profile.get("daily_free_hours", 4))
    fixed       = profile.get("fixed_activities", [])

    blocked: set = set()
    for act in fixed:
        for h in range(act.get("start_hour", 0), act.get("end_hour", 0)):
            blocked.add((act.get("day", ""), h))

    slots: List[Tuple[str, int]] = []
    for day in _DAY_ORDER:
        if day not in available:
            continue
        count = 0
        for hour in range(STUDY_START, STUDY_END):
            if count >= max_per_day:
                break
            if (day, hour) not in blocked:
                slots.append((day, hour))
                count += 1

    # Explain discretization so the maths is transparent
    raw_total = profile.get("daily_free_hours", 0) * len(available)
    profile.setdefault("_slot_note",
        f"  Note: {profile.get('daily_free_hours', 0)} h/day discretized to "
        f"{max_per_day} slot(s)/day (floor) "
        f"× {len([d for d in _DAY_ORDER if d in available])} days "
        f"= {len(slots)} slots  (raw total {raw_total:.1f} h)"
    )

    return slots


# =============================================================================
# 2.  State
# =============================================================================

@dataclass(frozen=True)
class State:
    """
    Immutable, hashable search state.

    slot_idx  : next slot to decide on
    remaining : sorted tuple of (task_name, hours_left)
    assigned  : sorted tuple of (task_name, day, hour) — schedule so far
    """
    slot_idx:  int
    remaining: tuple
    assigned:  tuple

    def rem_dict(self) -> Dict[str, float]:
        return {n: h for n, h in self.remaining}

    def is_goal(self) -> bool:
        return all(h <= 0 for _, h in self.remaining)

    def slots_exhausted(self, n: int) -> bool:
        return self.slot_idx >= n


def _initial_state(tasks: List[Dict]) -> State:
    # Hours are ceiled to whole slots so 5.6h → 6 slots, 3.3h → 4 slots.
    # This guarantees we never under-allocate slots for a task.
    # The rounding strategy is ceiling (safe) not floor (lossy).
    remaining = tuple(sorted(
        (t["name"], float(math.ceil(t["hours"]))) for t in tasks
    ))
    return State(slot_idx=0, remaining=remaining, assigned=())


# =============================================================================
# 3.  Successor generator  (shared by BFS and A*)
# =============================================================================

def _successors(
    state: State,
    slots: List[Tuple[str, int]],
    tasks: List[Dict],
) -> List[State]:
    """
    For the current slot:
      (a) assign it to any unfinished task whose deadline has not passed
      (b) skip it (IDLE — leave the slot unused)
    """
    if state.slot_idx >= len(slots):
        return []

    day, hour = slots[state.slot_idx]
    day_idx   = _DAY_ORDER.index(day) if day in _DAY_ORDER else 6
    rem       = state.rem_dict()
    nexts: List[State] = []

    for task in tasks:
        name     = task["name"]
        hrs_left = rem.get(name, 0.0)
        if hrs_left <= 0:
            continue
        if day_idx > task.get("deadline_days", 999):
            continue

        new_rem  = tuple(sorted(
            (n, round(h - 1.0, 1) if n == name else h)
            for n, h in state.remaining
        ))
        new_asgn = tuple(sorted(state.assigned + ((name, day, hour),)))
        nexts.append(State(
            slot_idx  = state.slot_idx + 1,
            remaining = new_rem,
            assigned  = new_asgn,
        ))

    # IDLE — skip this slot
    nexts.append(State(
        slot_idx  = state.slot_idx + 1,
        remaining = state.remaining,
        assigned  = state.assigned,
    ))

    return nexts


# =============================================================================
# 4.  Cost and heuristic
# =============================================================================

def _urgency_weight(task: Dict) -> float:
    """Higher = more urgent.  deadline 1 d → ~0.93,  14+ d → 0.05."""
    days = max(1, task.get("deadline_days", 14))
    return max(0.05, min(1.0, 1.0 - days / 14.0))


def _heuristic(state: State, tasks: List[Dict]) -> float:
    """
    Admissible h(n): hours_remaining × urgency_weight summed over
    all unfinished tasks.  Never over-estimates remaining cost.
    """
    task_map = {t["name"]: t for t in tasks}
    return sum(
        h * _urgency_weight(task_map[n])
        for n, h in state.remaining
        if h > 0 and n in task_map
    )


def _score(state: State, tasks: List[Dict]) -> float:
    """Schedule quality score — lower is better, 0 = fully scheduled."""
    return _heuristic(state, tasks)


def _conflicts(state: State, tasks: List[Dict]) -> int:
    """Count tasks with remaining hours + slots assigned past deadline."""
    day_idx  = {d: i for i, d in enumerate(_DAY_ORDER)}
    deadline = {t["name"]: t.get("deadline_days", 999) for t in tasks}

    n  = sum(1 for _, h in state.remaining if h > 0)
    n += sum(
        1 for name, day, _ in state.assigned
        if day_idx.get(day, 0) > deadline.get(name, 999)
    )
    return n


# =============================================================================
# 5.  Result builder  (shared)
# =============================================================================

def _make_result(
    state:      State,
    tasks:      List[Dict],
    explored:   int,
    time_ms:    float,
    found_goal: bool,
) -> Dict[str, Any]:
    # --- Fix 1: accurate completion breakdown ---
    # original hours (ceiled) per task
    orig = {t["name"]: float(math.ceil(t["hours"])) for t in tasks}
    # hours left per task from final state
    rem  = {n: h for n, h in state.remaining}
    # slots assigned per task
    assigned_hrs: Dict[str, float] = {}
    for name, _, _ in state.assigned:
        assigned_hrs[name] = assigned_hrs.get(name, 0) + 1.0

    completed:     List[str] = []
    partial:       List[str] = []   # (name, done_h, total_h)
    not_scheduled: List[str] = []

    for name, orig_h in orig.items():
        left = rem.get(name, 0.0)
        done = assigned_hrs.get(name, 0.0)
        if left <= 0:
            completed.append(name)
        elif done > 0:
            partial.append(f"{name} ({done:.0f}/{orig_h:.0f} h)")
        else:
            not_scheduled.append(name)

    return {
        "schedule":        list(state.assigned),
        "states_explored": explored,
        "time_ms":         round(time_ms, 2),
        "found_goal":      found_goal,
        "conflicts":       _conflicts(state, tasks),
        "score":           round(_score(state, tasks), 3),
        # legacy flat list — kept for backward compat with tests
        "unscheduled":     [n for n, h in state.remaining if h > 0],
        # new accurate breakdown
        "completed":       completed,
        "partial":         partial,
        "not_scheduled":   not_scheduled,
        "_state":          state,
    }


# =============================================================================
# 6.  BFS
# =============================================================================

def bfs_planner(
    profile:    Dict[str, Any],
    max_states: int = MAX_STATES,
) -> Dict[str, Any]:
    """
    Breadth-first search scheduler.

    Explores states level-by-level with no heuristic.
    Tasks are tried in input order — no urgency awareness.
    Returns the first complete goal state found, or the best partial
    schedule if the goal is unreachable within max_states.
    """
    t0    = time.perf_counter()
    tasks = profile.get("tasks", [])
    slots = build_available_slots(profile)

    init     = _initial_state(tasks)
    queue    = deque([init])
    visited  = {(init.slot_idx, init.remaining)}
    explored = 0
    best     = init

    while queue:
        state = queue.popleft()
        explored += 1

        if _score(state, tasks) < _score(best, tasks):
            best = state

        if state.is_goal():
            elapsed = (time.perf_counter() - t0) * 1000
            return _make_result(state, tasks, explored, elapsed, True)

        if state.slots_exhausted(len(slots)) or explored >= max_states:
            continue

        for s in _successors(state, slots, tasks):
            key = (s.slot_idx, s.remaining)
            if key not in visited:
                visited.add(key)
                queue.append(s)

    elapsed = (time.perf_counter() - t0) * 1000
    return _make_result(best, tasks, explored, elapsed, False)


# =============================================================================
# 7.  A*
# =============================================================================

def astar_planner(
    profile:    Dict[str, Any],
    max_states: int = MAX_STATES,
) -> Dict[str, Any]:
    """
    A* scheduler.

    Uses f(n) = g(n) + h(n) to prioritise urgent tasks.
    g(n) = cumulative urgency_weight of all assigned slots.
    h(n) = admissible urgency-hours estimate for unfinished tasks.
    """
    t0    = time.perf_counter()
    tasks = profile.get("tasks", [])
    slots = build_available_slots(profile)

    init     = _initial_state(tasks)
    counter  = 0
    heap     = [(_heuristic(init, tasks), counter, init)]
    visited  = {(init.slot_idx, init.remaining)}
    g_map    = {(init.slot_idx, init.remaining): 0.0}
    explored = 0
    best     = init

    while heap:
        _, _, state = heapq.heappop(heap)
        explored += 1

        if _score(state, tasks) < _score(best, tasks):
            best = state

        if state.is_goal():
            elapsed = (time.perf_counter() - t0) * 1000
            return _make_result(state, tasks, explored, elapsed, True)

        if state.slots_exhausted(len(slots)) or explored >= max_states:
            continue

        key_cur = (state.slot_idx, state.remaining)
        g_cur   = g_map.get(key_cur, 0.0)
        task_map = {t["name"]: t for t in tasks}

        for s in _successors(state, slots, tasks):
            key_s  = (s.slot_idx, s.remaining)
            newly  = [a for a in s.assigned if a not in state.assigned]
            delta_g = sum(
                _urgency_weight(task_map[a[0]])
                for a in newly if a[0] in task_map
            )
            g_new = g_cur + delta_g

            if key_s not in visited or g_new < g_map.get(key_s, float("inf")):
                visited.add(key_s)
                g_map[key_s] = g_new
                f_new = g_new + _heuristic(s, tasks)
                counter += 1
                heapq.heappush(heap, (f_new, counter, s))

    elapsed = (time.perf_counter() - t0) * 1000
    return _make_result(best, tasks, explored, elapsed, False)


# =============================================================================
# 8.  Comparison
# =============================================================================

def compare_planners(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run BFS and A* on the same profile, print a comparison table,
    and return both results.
    """
    tasks = profile.get("tasks", [])
    slots = build_available_slots(profile)

    print_subheader(f"Running planners — {profile.get('name', '?')}")
    print(f"  Tasks          : {len(tasks)}")
    print(f"  Total hours    : {profile.get('workload_hours', '?')} h")
    print(f"  Available slots: {len(slots)} × 1 h")
    if "_slot_note" in profile:
        print(profile["_slot_note"])

    bfs   = bfs_planner(profile)
    astar = astar_planner(profile)

    W = 26
    print_subheader("BFS vs A* — comparison table")
    print(f"  {'Metric':<{W}} {'BFS':>12}   {'A*':>12}")
    print("  " + "─" * (W + 28))
    for label, bv, av in [
        ("Time (ms)",         bfs["time_ms"],          astar["time_ms"]),
        ("States explored",   bfs["states_explored"],  astar["states_explored"]),
        ("Goal reached",      bfs["found_goal"],       astar["found_goal"]),
        ("Conflicts",         bfs["conflicts"],         astar["conflicts"]),
        ("Schedule score",    bfs["score"],             astar["score"]),
        ("Unscheduled tasks", len(bfs["unscheduled"]),  len(astar["unscheduled"])),
    ]:
        print(f"  {label:<{W}} {str(bv):>12}   {str(av):>12}")

    print()
    _analysis(bfs, astar)
    _overload_recommendations(profile, astar)   # use A* result (better quality)
    return {"bfs": bfs, "astar": astar, "slots": slots}


def _analysis(bfs: Dict, astar: Dict) -> None:
    print("  Analysis")
    print("  " + "─" * 40)

    # Fix 4 — context-aware: different explanation for overloaded vs normal
    overloaded = bool(bfs.get("partial") or bfs.get("not_scheduled"))
    if overloaded:
        print("  • In overloaded scenarios both BFS and A* struggle because "
              "no complete solution exists — every search path leads to partial "
              "failure. A* cannot prune aggressively when all branches are "
              "equally incomplete, so state counts converge.")
    elif astar["states_explored"] < bfs["states_explored"]:
        ratio = bfs["states_explored"] / max(1, astar["states_explored"])
        print(f"  • A* explored {ratio:.1f}× fewer states. The urgency "
              "heuristic pruned branches where low-priority tasks occupied "
              "slots needed by urgent ones.")
    else:
        print(f"  • BFS explored {bfs['states_explored']} states vs "
              f"A*'s {astar['states_explored']}. On small or uniform-urgency "
              "inputs BFS can be faster — A*'s heap overhead outweighs its "
              "pruning benefit.")

    if astar["score"] < bfs["score"]:
        print(f"  • A* schedule is higher quality "
              f"(score {astar['score']:.3f} vs BFS {bfs['score']:.3f}): "
              "urgent tasks were placed in earlier slots.")
    elif bfs["score"] < astar["score"]:
        print(f"  • BFS score ({bfs['score']:.3f}) beat A* ({astar['score']:.3f}) "
              "on this input.")
    else:
        print("  • Both strategies scored equally. "
              "A*'s urgency advantage is most visible in the Tight scenario.")
    print()


def _overload_recommendations(profile: Dict, result: Dict) -> None:
    """
    Fix 5 — when overload is detected print actionable copilot recommendations.
    Called with the A* result (higher quality than BFS).
    """
    partial       = result.get("partial", [])
    not_scheduled = result.get("not_scheduled", [])
    if not partial and not not_scheduled:
        return

    tasks      = profile.get("tasks", [])
    urgent     = [t for t in tasks if t.get("deadline_days", 99) <= 3]
    low_urg    = [t for t in tasks if t.get("deadline_days", 99) > 7]
    total_h    = profile.get("workload_hours", 0)
    free_total = (profile.get("daily_free_hours", 0)
                  * len(profile.get("available_days", [])))
    shortfall  = round(total_h - free_total, 1)

    print_subheader("⚠  Overload detected — copilot recommendations")

    if partial:
        print("  Partially completed (started but not finished):")
        for p in partial:
            print(f"    • {p}")
    if not_scheduled:
        print("  Not started (no slots assigned):")
        for n in not_scheduled:
            print(f"    • {n}")

    print("\n  Suggested actions:")

    step = 1
    if urgent:
        names = ", ".join(t["name"] for t in urgent)
        print(f"  {step}. Prioritise the {len(urgent)} task(s) due within "
              f"3 days first: {names}")
        step += 1

    if low_urg:
        names = ", ".join(t["name"] for t in low_urg)
        print(f"  {step}. Consider deferring low-urgency tasks (deadline > 7 d): "
              f"{names}")
        step += 1

    if shortfall > 0:
        print(f"  {step}. You need {shortfall} h more than you currently have free "
              f"({total_h} h needed vs {free_total:.1f} h available). "
              "Increase daily study hours or reduce task estimates.")
        step += 1

    weekend = {"Saturday", "Sunday"} - set(profile.get("available_days", []))
    if weekend:
        print(f"  {step}. Add {' and '.join(sorted(weekend))} as study days "
              "to gain extra slots.")
        step += 1

    print(f"  {step}. Communicate with your instructor early if any deadline "
          "cannot be met — partial submissions are usually better than none.")
    print()


# =============================================================================
# 9.  Display schedule
# =============================================================================

def display_schedule(
    result:   Dict[str, Any],
    strategy: str = "Schedule",
) -> None:
    """Print a formatted weekly timetable from a planner result."""
    schedule = result.get("schedule", [])

    day_map: Dict[str, List] = {d: [] for d in _DAY_ORDER}
    for name, day, hour in schedule:
        if day in day_map:
            day_map[day].append((hour, name))

    print_subheader(f"{strategy} — weekly view")
    printed = False
    for day in _DAY_ORDER:
        entries = sorted(day_map[day])
        if not entries:
            continue
        printed = True
        print(f"\n  {day}")
        for hour, name in entries:
            print(f"    {hour:02d}:00 – {hour+1:02d}:00   {name}")

    if not printed:
        print("  (no slots assigned)")

    # Fix 1: accurate completion breakdown
    completed     = result.get("completed", [])
    partial       = result.get("partial", [])
    not_scheduled = result.get("not_scheduled", [])

    print(f"\n  Slots used    : {len(schedule)}")
    print(f"  Goal reached  : {result.get('found_goal', '?')}")
    print(f"  Conflicts     : {result.get('conflicts', '?')}")
    print(f"  Score         : {result.get('score', '?')}")
    if completed:
        print(f"  Completed     : {', '.join(completed)}")
    if partial:
        print(f"  Partial       : {', '.join(partial)}")
    if not_scheduled:
        print(f"  Not scheduled : {', '.join(not_scheduled)}")
    print()


# =============================================================================
# 10. Tight comparison profile
# =============================================================================

def build_tight_comparison_profile() -> Dict[str, Any]:
    """
    A crafted 3-task profile where urgency order matters visibly.

    Tasks are listed in REVERSED urgency order (relaxed → urgent) so BFS,
    which tries tasks in input order, wastes Monday's slot on the
    low-urgency task and misses the 1-day deadline.
    A* reorders by urgency and schedules the urgent task on Monday.
    """
    return {
        "name": "Tight-Test", "gender": 0,
        "attendance": 75.0, "confidence_level": 3,
        "stress_level": 3,  "quiz_score_avg": 65.0,
        "workload_credits": 40,
        "tasks": [
            {"name": "Background reading", "subject": "History",
             "hours": 3.0, "deadline_days": 12},   # relaxed — input first
            {"name": "Revision notes",     "subject": "Chemistry",
             "hours": 2.0, "deadline_days": 4},    # moderate
            {"name": "Past paper",         "subject": "Chemistry",
             "hours": 2.0, "deadline_days": 1},    # URGENT — input last
        ],
        "exams":            [{"subject": "Chemistry", "days_until": 1}],
        "fixed_activities": [],
        "available_days":   ["Monday", "Tuesday", "Wednesday"],
        "daily_free_hours": 2.0,   # 6 slots total — tight but solvable
        "workload_tasks":   3,
        "workload_hours":   7.0,
        "deadlines":        2,
        "availability_constraints": 2,
    }
