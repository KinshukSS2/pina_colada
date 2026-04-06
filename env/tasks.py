from __future__ import annotations

from typing import Dict, List

from env.models import TaskConfig


def _cycle(values: List[int], length: int) -> List[int]:
    return [values[index % len(values)] for index in range(length)]


def build_easy_task() -> TaskConfig:
    max_steps = 50
    return TaskConfig(
        task_id="easy",
        difficulty="easy",
        description="Balanced daytime flow with low emergency pressure.",
        max_steps=max_steps,
        min_green=3,
        max_green=10,
        base_capacity=8,
        target_passed=320,
        target_wait=420.0,
        target_emergency_wait=25.0,
        target_fairness_gap=0.22,
        target_max_wait=70.0,
        target_backlog_end=45.0,
        target_starvation_events=4.0,
        target_flicker_events=5.0,
        target_stability_penalty=7.0,
        emergency_priority_window=2,
        starvation_threshold=7,
        flicker_window=2,
        catastrophic_emergency_wait=70.0,
        catastrophic_backlog=400,
        arrivals_ns=_cycle([3, 4, 3, 5], max_steps),
        arrivals_ew=_cycle([3, 3, 4, 4], max_steps),
        emergency_ns_steps=[18, 37],
        emergency_ew_steps=[30],
    )


def build_medium_task() -> TaskConfig:
    max_steps = 60
    return TaskConfig(
        task_id="medium",
        difficulty="medium",
        description="Directional rush hour with intermittent emergency arrivals.",
        max_steps=max_steps,
        min_green=3,
        max_green=12,
        base_capacity=10,
        target_passed=390,
        target_wait=620.0,
        target_emergency_wait=35.0,
        target_fairness_gap=0.28,
        target_max_wait=90.0,
        target_backlog_end=70.0,
        target_starvation_events=5.0,
        target_flicker_events=7.0,
        target_stability_penalty=10.0,
        emergency_priority_window=2,
        starvation_threshold=8,
        flicker_window=2,
        catastrophic_emergency_wait=90.0,
        catastrophic_backlog=500,
        arrivals_ns=_cycle([5, 6, 6, 4, 5], max_steps),
        arrivals_ew=_cycle([2, 3, 4, 3, 2], max_steps),
        emergency_ns_steps=[14, 29, 44],
        emergency_ew_steps=[23, 52],
    )


def build_hard_task() -> TaskConfig:
    max_steps = 70
    return TaskConfig(
        task_id="hard",
        difficulty="hard",
        description="Peak congestion with emergency contention and fairness constraints.",
        max_steps=max_steps,
        min_green=2,
        max_green=14,
        base_capacity=14,
        target_passed=410,
        target_wait=890.0,
        target_emergency_wait=45.0,
        target_fairness_gap=0.18,
        target_max_wait=120.0,
        target_backlog_end=110.0,
        target_starvation_events=6.0,
        target_flicker_events=8.0,
        target_stability_penalty=12.0,
        emergency_priority_window=2,
        starvation_threshold=8,
        flicker_window=2,
        catastrophic_emergency_wait=110.0,
        catastrophic_backlog=600,
        arrivals_ns=_cycle([7, 5, 6, 8, 4, 7], max_steps),
        arrivals_ew=_cycle([6, 7, 5, 6, 8, 4], max_steps),
        emergency_ns_steps=[10, 26, 40, 59],
        emergency_ew_steps=[17, 34, 48, 65],
    )


def build_chaos_task() -> TaskConfig:
    max_steps = 75

    arrivals_ns: List[int] = []
    arrivals_ew: List[int] = []
    for index in range(max_steps):
        wave = index % 15
        if wave < 5:
            arrivals_ns.append(9 + (index % 3))
            arrivals_ew.append(2 + (index % 2))
        elif wave < 10:
            arrivals_ns.append(3 + (index % 2))
            arrivals_ew.append(9 + ((index + 1) % 3))
        else:
            arrivals_ns.append(4 + (index % 3))
            arrivals_ew.append(4 + ((index + 2) % 3))

    return TaskConfig(
        task_id="chaos",
        difficulty="extreme",
        description="Extreme stress test with simultaneous emergencies, platoons, and sensor blackouts.",
        max_steps=max_steps,
        min_green=2,
        max_green=12,
        base_capacity=16,
        target_passed=500,
        target_wait=900.0,
        target_emergency_wait=35.0,
        target_fairness_gap=0.15,
        target_max_wait=140.0,
        target_backlog_end=130.0,
        target_starvation_events=8.0,
        target_flicker_events=10.0,
        target_stability_penalty=14.0,
        emergency_priority_window=2,
        starvation_threshold=8,
        flicker_window=2,
        catastrophic_emergency_wait=120.0,
        catastrophic_backlog=800,
        arrivals_ns=arrivals_ns,
        arrivals_ew=arrivals_ew,
        emergency_ns_steps=[10, 11, 40, 65],
        emergency_ew_steps=[10, 25, 40, 65],
    )


def task_catalog() -> Dict[str, TaskConfig]:
    easy = build_easy_task()
    medium = build_medium_task()
    hard = build_hard_task()
    chaos = build_chaos_task()
    return {
        easy.task_id: easy,
        medium.task_id: medium,
        hard.task_id: hard,
        chaos.task_id: chaos,
    }


def get_task(task_id: str) -> TaskConfig:
    catalog = task_catalog()
    key = (task_id or "easy").lower()
    if key not in catalog:
        return catalog["easy"]
    return catalog[key]
