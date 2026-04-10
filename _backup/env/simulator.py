from __future__ import annotations

from typing import Optional, Tuple

from env.models import SimAction, IntersectionState, TaskConfig

VALID_ACTIONS = [
    "hold",
    "switch",
    "prioritize_emergency",
    "set_ns_green:<n>",
    "set_ew_green:<n>",
]


def parse_action(raw_action: str) -> Tuple[SimAction, bool, Optional[str]]:
    if raw_action is None:
        return SimAction(raw="hold", name="hold"), False, "empty_action"

    text = str(raw_action).strip().lower()
    if not text:
        return SimAction(raw="hold", name="hold"), False, "blank_action"

    if text == "hold":
        return SimAction(raw=text, name="hold"), True, None

    if text == "switch":
        return SimAction(raw=text, name="switch"), True, None

    if text == "prioritize_emergency":
        return SimAction(raw=text, name="prioritize_emergency"), True, None

    if text.startswith("set_ns_green:"):
        value = text.replace("set_ns_green:", "", 1)
        if value.isdigit():
            return SimAction(raw=text, name="set_ns_green", value=int(value)), True, None
        return SimAction(raw=text, name="hold"), False, "invalid_ns_duration"

    if text.startswith("set_ew_green:"):
        value = text.replace("set_ew_green:", "", 1)
        if value.isdigit():
            return SimAction(raw=text, name="set_ew_green", value=int(value)), True, None
        return SimAction(raw=text, name="hold"), False, "invalid_ew_duration"

    return SimAction(raw=text, name="hold"), False, "unknown_action"


def _clamp_green(value: int, task: TaskConfig) -> int:
    return max(task.min_green, min(task.max_green, int(value)))


def _toggle_phase(current_phase: str) -> str:
    return "ew" if current_phase == "ns" else "ns"


def _signal_label(state: IntersectionState, lane: str) -> str:
    if state.yellow_remaining > 0 and state.current_phase == lane:
        return "YEL"
    if state.current_phase == lane and state.yellow_remaining == 0:
        return f"GRN({state.phase_remaining})"
    return "RED"


def _deterministic_u01(seed: int, timestep: int, salt: int) -> float:
    mixed = (seed * 1664525 + (timestep + 1) * 1013904223 + salt * 2654435761) & 0xFFFFFFFF
    return (mixed % 10000) / 10000.0


def _deterministic_jitter(seed: int, timestep: int, salt: int) -> int:
    value = (seed * 1103515245 + (timestep + 1) * 12345 + salt * 1013) & 0x7FFFFFFF
    return int(value % 3) - 1


def _generate_ascii(state: IntersectionState) -> str:
    ns_light = _signal_label(state, "ns")
    ew_light = _signal_label(state, "ew")
    q_ns = "?" if state.sensor_status == "OFFLINE" else str(state.queue_ns)
    q_ew = "?" if state.sensor_status == "OFFLINE" else str(state.queue_ew)
    ped = "🚶 WARNING" if state.pedestrian_waiting else ""
    return (
        f"\n"
        f"     [ NS: {ns_light} ]\n"
        f"       |   | ↓ |\n"
        f"       |   | ↓ | ({q_ns})\n"
        f"------ +---+---+ ------ {ped}\n"
        f" EW: {ew_light} |   |   | EW: {ew_light}\n"
        f" ({q_ew})→ |   |\n"
        f"------ +---+---+ ------\n"
        f"       | ↑ |   |\n"
    )


def _inject_arrivals(state: IntersectionState, task: TaskConfig) -> None:
    index = min(state.timestep, task.max_steps - 1)
    jitter_ns = _deterministic_jitter(state.seed, state.timestep, 31)
    jitter_ew = _deterministic_jitter(state.seed, state.timestep, 32)
    state.queue_ns = max(0, state.queue_ns + task.arrivals_ns[index] + jitter_ns)
    state.queue_ew = max(0, state.queue_ew + task.arrivals_ew[index] + jitter_ew)
    if index in task.emergency_ns_steps:
        state.emergency_ns += 1
        state.emergency_appearances += 1
    if index in task.emergency_ew_steps:
        state.emergency_ew += 1
        state.emergency_appearances += 1

    if _deterministic_u01(state.seed, state.timestep, 33) < 0.05:
        state.sensor_status = "OFFLINE"
    else:
        state.sensor_status = "ONLINE"

    if not state.pedestrian_waiting and _deterministic_u01(state.seed, state.timestep, 34) < 0.08:
        state.pedestrian_waiting = True
        state.pedestrian_patience = 3


def _phase_capacity(state: IntersectionState, task: TaskConfig) -> Tuple[int, int]:
    if state.yellow_remaining > 0:
        return 0, 0

    kinematic_eff = 0.5 if state.green_duration <= 2 else 1.0
    base_eff = 0.6 if state.lane_health < 1.0 else 1.0
    effective_capacity = max(1, int(task.base_capacity * kinematic_eff * base_eff))

    if state.current_phase == "ns":
        bonus = 1 if state.emergency_ns > 0 else 0
        return effective_capacity + bonus, 0
    bonus = 1 if state.emergency_ew > 0 else 0
    return 0, effective_capacity + bonus


def _move_vehicles(state: IntersectionState, task: TaskConfig) -> Tuple[int, int]:
    cap_ns, cap_ew = _phase_capacity(state, task)

    moved_ns = min(state.queue_ns, cap_ns)
    moved_ew = min(state.queue_ew, cap_ew)

    state.queue_ns -= moved_ns
    state.queue_ew -= moved_ew
    state.moved_ns += moved_ns
    state.moved_ew += moved_ew

    if state.current_phase == "ns" and state.yellow_remaining == 0 and state.emergency_ns > 0 and moved_ns > 0:
        state.emergency_ns = max(0, state.emergency_ns - 1)
        state.emergency_priority_hits += 1
    if state.current_phase == "ew" and state.yellow_remaining == 0 and state.emergency_ew > 0 and moved_ew > 0:
        state.emergency_ew = max(0, state.emergency_ew - 1)
        state.emergency_priority_hits += 1

    return moved_ns, moved_ew


def _update_wait_and_fairness(state: IntersectionState) -> None:
    state.total_wait_time += state.queue_ns + state.queue_ew
    state.queue_wait_ns += state.queue_ns
    state.queue_wait_ew += state.queue_ew
    state.emergency_wait_time += 2 * (state.emergency_ns + state.emergency_ew)
    state.max_wait_seen = max(
        state.max_wait_seen,
        state.queue_wait_ns,
        state.queue_wait_ew,
    )
    state.backlog_total = state.queue_ns + state.queue_ew

    moved_total = max(1, state.moved_ns + state.moved_ew)
    state.fairness_gap = abs(state.moved_ns - state.moved_ew) / moved_total


def _register_phase_change(state: IntersectionState, task: TaskConfig) -> None:
    state.phase_switches += 1
    dt = state.timestep - state.last_switch_timestep
    if dt <= task.flicker_window:
        state.flicker_events += 1
    if state.current_phase_run < task.min_green:
        state.stability_penalty += float(task.min_green - state.current_phase_run)
    state.last_switch_timestep = state.timestep
    state.current_phase_run = 0


def _update_starvation_and_stability(
    state: IntersectionState,
    task: TaskConfig,
    moved_ns: int,
    moved_ew: int,
) -> None:
    if moved_ns > 0:
        state.last_service_ns = state.timestep
    if moved_ew > 0:
        state.last_service_ew = state.timestep

    ns_idle = state.timestep - state.last_service_ns
    ew_idle = state.timestep - state.last_service_ew
    if ns_idle >= task.starvation_threshold and ns_idle % task.starvation_threshold == 0:
        state.starvation_events += 1
        state.stability_penalty += 0.5
    if ew_idle >= task.starvation_threshold and ew_idle % task.starvation_threshold == 0:
        state.starvation_events += 1
        state.stability_penalty += 0.5

    moved_total = moved_ns + moved_ew
    if moved_total <= 0:
        state.no_progress_steps += 1
    else:
        state.no_progress_steps = 0
    if state.no_progress_steps >= task.starvation_threshold:
        state.stability_penalty += 0.5


def _apply_safety_kills(state: IntersectionState, task: TaskConfig) -> None:
    if state.catastrophic_event:
        state.done = True
        return

    emergency_delay = state.emergency_wait_time / max(1, state.timestep + 1)
    if state.collision_detected:
        state.catastrophic_event = True
        state.catastrophic_reason = "safety_violation"

    if emergency_delay >= task.catastrophic_emergency_wait:
        state.catastrophic_event = True
        state.catastrophic_reason = "emergency_delay_breach"

    if state.backlog_total >= task.catastrophic_backlog:
        state.catastrophic_event = True
        state.catastrophic_reason = "backlog_overflow"

    if state.flicker_events >= int(3 * task.target_flicker_events):
        state.catastrophic_event = True
        state.catastrophic_reason = "signal_flicker_instability"

    if state.catastrophic_event:
        state.done = True


def apply_action(state: IntersectionState, task: TaskConfig, action: SimAction, valid: bool) -> None:
    if state.pedestrian_waiting:
        if action.name not in {"hold", "switch"}:
            state.collision_detected = True
            state.safety_violations += 1
            return
        if action.name == "hold" and state.yellow_remaining == 0 and state.phase_remaining > 0:
            state.pedestrian_patience -= 1
            if state.pedestrian_patience <= 0:
                state.collision_detected = True
                state.safety_violations += 1
                return
        elif action.name == "switch":
            state.pedestrian_waiting = False
            state.pedestrian_patience = 0

    if state.yellow_remaining > 0:
        if action.name != "hold":
            state.collision_detected = True
            state.safety_violations += 1
        return

    if not valid:
        state.invalid_actions += 1
        return

    if action.name == "hold":
        return

    if action.name == "switch":
        state.current_phase = _toggle_phase(state.current_phase)
        state.phase_remaining = task.min_green
        state.yellow_remaining = 2
        _register_phase_change(state, task)
        return

    if action.name == "set_ns_green":
        value = _clamp_green(action.value or task.min_green, task)
        state.current_phase = "ns"
        state.phase_remaining = value
        state.yellow_remaining = 0
        _register_phase_change(state, task)
        return

    if action.name == "set_ew_green":
        value = _clamp_green(action.value or task.min_green, task)
        state.current_phase = "ew"
        state.phase_remaining = value
        state.yellow_remaining = 0
        _register_phase_change(state, task)
        return

    if action.name == "prioritize_emergency":
        if state.emergency_ns <= 0 and state.emergency_ew <= 0:
            state.invalid_actions += 1
            return
        previous_phase = state.current_phase
        if state.emergency_ns >= state.emergency_ew:
            state.current_phase = "ns"
        else:
            state.current_phase = "ew"
        state.phase_remaining = max(task.min_green, 2)
        if state.current_phase != previous_phase:
            state.yellow_remaining = 2
        if state.current_phase != previous_phase:
            _register_phase_change(state, task)


def simulate_step(state: IntersectionState, task: TaskConfig, action: SimAction, valid: bool) -> Tuple[int, int]:
    apply_action(state, task, action, valid)

    _inject_arrivals(state, task)

    state.lane_health = 0.6 if (state.timestep // 10) % 2 == 1 else 1.0

    if state.phase_remaining <= 0:
        state.phase_remaining = task.min_green

    if state.yellow_remaining > 0:
        state.green_duration = 0
    else:
        state.green_duration += 1

    moved_ns, moved_ew = _move_vehicles(state, task)
    _update_wait_and_fairness(state)
    _update_starvation_and_stability(state, task, moved_ns, moved_ew)
    state.current_phase_run += 1

    state.phase_remaining = max(0, state.phase_remaining - 1)
    state.yellow_remaining = max(0, state.yellow_remaining - 1)
    _apply_safety_kills(state, task)
    state.timestep += 1
    if state.timestep >= task.max_steps or state.collision_detected:
        state.done = True

    return moved_ns, moved_ew
