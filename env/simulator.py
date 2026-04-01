from __future__ import annotations

from typing import Optional, Tuple

from env.models import Action, IntersectionState, TaskConfig

VALID_ACTIONS = [
    "hold",
    "switch",
    "set_ns_green:<n>",
    "set_ew_green:<n>",
    "prioritize_emergency",
]


def parse_action(raw_action: str) -> Tuple[Action, bool, Optional[str]]:
    if raw_action is None:
        return Action(raw="hold", name="hold"), False, "empty_action"

    text = str(raw_action).strip().lower()
    if not text:
        return Action(raw="hold", name="hold"), False, "blank_action"

    if text == "hold":
        return Action(raw=text, name="hold"), True, None

    if text == "switch":
        return Action(raw=text, name="switch"), True, None

    if text == "prioritize_emergency":
        return Action(raw=text, name="prioritize_emergency"), True, None

    if text.startswith("set_ns_green:"):
        value = text.replace("set_ns_green:", "", 1)
        if value.isdigit():
            return Action(raw=text, name="set_ns_green", value=int(value)), True, None
        return Action(raw=text, name="hold"), False, "invalid_ns_duration"

    if text.startswith("set_ew_green:"):
        value = text.replace("set_ew_green:", "", 1)
        if value.isdigit():
            return Action(raw=text, name="set_ew_green", value=int(value)), True, None
        return Action(raw=text, name="hold"), False, "invalid_ew_duration"

    return Action(raw=text, name="hold"), False, "unknown_action"


def _clamp_green(value: int, task: TaskConfig) -> int:
    return max(task.min_green, min(task.max_green, int(value)))


def _toggle_phase(current_phase: str) -> str:
    return "ew" if current_phase == "ns" else "ns"


def _inject_arrivals(state: IntersectionState, task: TaskConfig) -> None:
    index = min(state.timestep, task.max_steps - 1)
    state.queue_ns += task.arrivals_ns[index]
    state.queue_ew += task.arrivals_ew[index]
    if index in task.emergency_ns_steps:
        state.emergency_ns += 1
    if index in task.emergency_ew_steps:
        state.emergency_ew += 1


def _phase_capacity(state: IntersectionState, task: TaskConfig) -> Tuple[int, int]:
    if state.current_phase == "ns":
        bonus = 1 if state.emergency_ns > 0 else 0
        return task.base_capacity + bonus, 0
    bonus = 1 if state.emergency_ew > 0 else 0
    return 0, task.base_capacity + bonus


def _move_vehicles(state: IntersectionState, task: TaskConfig) -> Tuple[int, int]:
    cap_ns, cap_ew = _phase_capacity(state, task)

    moved_ns = min(state.queue_ns, cap_ns)
    moved_ew = min(state.queue_ew, cap_ew)

    state.queue_ns -= moved_ns
    state.queue_ew -= moved_ew
    state.moved_ns += moved_ns
    state.moved_ew += moved_ew

    if state.current_phase == "ns" and state.emergency_ns > 0 and moved_ns > 0:
        state.emergency_ns = max(0, state.emergency_ns - 1)
    if state.current_phase == "ew" and state.emergency_ew > 0 and moved_ew > 0:
        state.emergency_ew = max(0, state.emergency_ew - 1)

    return moved_ns, moved_ew


def _update_wait_and_fairness(state: IntersectionState) -> None:
    state.total_wait_time += state.queue_ns + state.queue_ew
    state.emergency_wait_time += 2 * (state.emergency_ns + state.emergency_ew)

    moved_total = max(1, state.moved_ns + state.moved_ew)
    state.fairness_gap = abs(state.moved_ns - state.moved_ew) / moved_total


def apply_action(state: IntersectionState, task: TaskConfig, action: Action, valid: bool) -> None:
    if not valid:
        state.invalid_actions += 1
        return

    if action.name == "hold":
        return

    if action.name == "switch":
        if state.phase_remaining <= 1:
            state.current_phase = _toggle_phase(state.current_phase)
            state.phase_remaining = task.min_green
        return

    if action.name == "set_ns_green":
        value = _clamp_green(action.value or task.min_green, task)
        state.current_phase = "ns"
        state.phase_remaining = value
        return

    if action.name == "set_ew_green":
        value = _clamp_green(action.value or task.min_green, task)
        state.current_phase = "ew"
        state.phase_remaining = value
        return

    if action.name == "prioritize_emergency":
        if state.emergency_ns <= 0 and state.emergency_ew <= 0:
            state.invalid_actions += 1
            return
        if state.emergency_ns >= state.emergency_ew:
            state.current_phase = "ns"
        else:
            state.current_phase = "ew"
        state.phase_remaining = max(task.min_green, 2)


def simulate_step(state: IntersectionState, task: TaskConfig, action: Action, valid: bool) -> Tuple[int, int]:
    apply_action(state, task, action, valid)

    _inject_arrivals(state, task)

    if state.phase_remaining <= 0:
        state.phase_remaining = task.min_green
    moved_ns, moved_ew = _move_vehicles(state, task)
    _update_wait_and_fairness(state)

    state.phase_remaining = max(0, state.phase_remaining - 1)
    state.timestep += 1
    state.done = state.timestep >= task.max_steps

    return moved_ns, moved_ew
