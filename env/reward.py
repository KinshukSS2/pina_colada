from __future__ import annotations

from env.models import IntersectionState

_EPS = 0.01

def _clip_reward(value: float) -> float:
    return max(_EPS, min(1.0 - _EPS, float(value)))


def compute_reward(
    previous_state: IntersectionState,
    current_state: IntersectionState,
    moved_ns: int,
    moved_ew: int,
    action_valid: bool,
) -> float:
    if current_state.collision_detected:
        return _EPS

    if not action_valid:
        return 0.10

    queue_prev = previous_state.queue_ns + previous_state.queue_ew
    queue_curr = current_state.queue_ns + current_state.queue_ew
    reduction_reward = (queue_prev - queue_curr) * 0.05

    stability_penalty = 0.0
    if previous_state.current_phase != current_state.current_phase:
        if current_state.emergency_ns <= 0 and current_state.emergency_ew <= 0:
            stability_penalty = -0.3

    flow_reward = (moved_ns + moved_ew) * 0.1
    emergency_wait_penalty = -0.2 * (current_state.emergency_wait_time - previous_state.emergency_wait_time)

    total_reward = reduction_reward + stability_penalty + flow_reward + emergency_wait_penalty
    if current_state.catastrophic_event:
        total_reward = min(total_reward, _EPS)

    # Shift negative rewards to be safely within the (0, 1) interval.
    # Linear scale from [-1.0, 1.0] -> [0.01, 0.99] to preserve relative signal.
    scaled_reward = _EPS + ((total_reward + 1.0) / 2.0) * (1.0 - 2 * _EPS)
    return _clip_reward(scaled_reward)
