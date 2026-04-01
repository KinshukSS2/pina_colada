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
        base_capacity=5,
        target_passed=320,
        target_wait=420.0,
        target_emergency_wait=25.0,
        target_fairness_gap=0.22,
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
        base_capacity=5,
        target_passed=390,
        target_wait=620.0,
        target_emergency_wait=35.0,
        target_fairness_gap=0.28,
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
        base_capacity=4,
        target_passed=410,
        target_wait=890.0,
        target_emergency_wait=45.0,
        target_fairness_gap=0.18,
        arrivals_ns=_cycle([7, 5, 6, 8, 4, 7], max_steps),
        arrivals_ew=_cycle([6, 7, 5, 6, 8, 4], max_steps),
        emergency_ns_steps=[10, 26, 40, 59],
        emergency_ew_steps=[17, 34, 48, 65],
    )


def task_catalog() -> Dict[str, TaskConfig]:
    easy = build_easy_task()
    medium = build_medium_task()
    hard = build_hard_task()
    return {
        easy.task_id: easy,
        medium.task_id: medium,
        hard.task_id: hard,
    }


def get_task(task_id: str) -> TaskConfig:
    catalog = task_catalog()
    key = (task_id or "easy").lower()
    if key not in catalog:
        return catalog["easy"]
    return catalog[key]
