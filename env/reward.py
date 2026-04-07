from __future__ import annotations
import math

from env.models import IntersectionState

_EPS = 0.01

def _clip_reward(value: float) -> float:
    """Ensures the reward never exactly hits 0.0 or 1.0."""
    return max(_EPS, min(1.0 - _EPS, float(value)))

def compute_reward(
    previous_state: IntersectionState,
    current_state: IntersectionState,
    moved_ns: int,
    moved_ew: int,
    action_valid: bool,
) -> float:
    # 1. Continuous Components
    # Reward for moving vehicles (throughput)
    flow_reward = (moved_ns + moved_ew) * 0.1
    
    # Penalty for unnecessary phase changes (stability)
    stability_penalty = 0.0
    if previous_state.current_phase != current_state.current_phase:
        if current_state.emergency_ns <= 0 and current_state.emergency_ew <= 0:
            stability_penalty = -0.3

    # Penalty for increasing emergency vehicle wait times
    emergency_wait_penalty = -0.2 * (
        current_state.emergency_wait_time - previous_state.emergency_wait_time
    )

    # Base raw reward
    total_raw_reward = flow_reward + stability_penalty + emergency_wait_penalty

    # 2. Discrete Failure Penalties
    # Instead of early returns, subtract heavily from the raw reward.
    # This prevents the scaling logic bugs from the previous version.
    if current_state.collision_detected:
        total_raw_reward -= 10.0
        
    if current_state.catastrophic_event:
        total_raw_reward -= 5.0
        
    if not action_valid:
        total_raw_reward -= 1.0

    # 3. Squashing
    # math.tanh(x) smoothly maps (-inf, inf) to (-1, 1).
    # Adding 1 and dividing by 2 maps it smoothly to (0, 1).
    # This preserves the gradient so the agent still knows a "good" move 
    # is better than an "okay" move, even at the high end of the scale.
    squashed_reward = (math.tanh(total_raw_reward) + 1.0) / 2.0

    # 4. Final Clip
    # Guarantee the values are safely inside the [0.01, 0.99] interval.
    return _clip_reward(squashed_reward)