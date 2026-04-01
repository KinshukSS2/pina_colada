from __future__ import annotations

from env.models import IntersectionState, TaskConfig


def compute_reward(
    previous_state: IntersectionState,
    current_state: IntersectionState,
    moved_ns: int,
    moved_ew: int,
    action_valid: bool,
) -> float:
    moved_total = moved_ns + moved_ew
    throughput_component = 0.08 * moved_total

    wait_delta = current_state.total_wait_time - previous_state.total_wait_time
    delay_component = -0.01 * wait_delta

    emergency_delta = current_state.emergency_wait_time - previous_state.emergency_wait_time
    emergency_component = -0.03 * emergency_delta

    fairness_component = -0.2 * current_state.fairness_gap

    invalid_component = -0.25 if not action_valid else 0.0
    done_component = 0.0
    if current_state.done:
        done_component = 0.2

    reward = (
        throughput_component
        + delay_component
        + emergency_component
        + fairness_component
        + invalid_component
        + done_component
    )
    return float(max(-2.0, min(2.0, reward)))
