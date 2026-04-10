from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action as OpenEnvAction
from openenv.core.env_server.types import Observation as OpenEnvObservation
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# OpenEnv-compliant Action & Observation (extend framework base classes)
# ---------------------------------------------------------------------------

class TrafficAction(OpenEnvAction):
    """Action sent by the agent to the traffic control environment."""
    action: str = Field(default="hold", description="Traffic control action: hold, switch, prioritize_emergency, set_ns_green:<n>, set_ew_green:<n>")
    session_id: Optional[str] = Field(default=None, description="Session identifier for multi-step episodes")


class TrafficObservation(OpenEnvObservation):
    """Observation returned by the traffic control environment."""
    session_id: str = ""
    task_id: str = ""
    timestep: int = 0
    max_steps: int = 60
    current_phase: str = "ns"
    phase_remaining: int = 0
    yellow_active: bool = False
    green_duration: int = 0
    ascii_minimap: str = ""
    sensor_status: str = "ONLINE"
    pedestrian_waiting: bool = False
    action_mask: Dict[str, bool] = Field(default_factory=dict)
    queue_ns: int = 0
    queue_ew: int = 0
    emergency_ns: int = 0
    emergency_ew: int = 0
    lane_health: float = 1.0
    moved_ns: int = 0
    moved_ew: int = 0
    total_wait_time: int = 0
    emergency_wait_time: int = 0
    invalid_actions: int = 0
    fairness_gap: float = 0.0
    max_wait_seen: int = 0
    backlog_total: int = 0
    phase_switches: int = 0
    flicker_events: int = 0
    starvation_events: int = 0
    stability_penalty: float = 0.0
    catastrophic_event: bool = False
    seed_value: int = 0
    last_action_valid: bool = True
    valid_actions: List[str] = Field(default_factory=list)
    # Grader info (included in observation since OpenEnv has no separate info channel)
    score_estimate: float = 0.0
    reasoning_logs: List[str] = Field(default_factory=list)
    grader_breakdown: Dict[str, float] = Field(default_factory=dict)
    grader_reasons: List[str] = Field(default_factory=list)
    last_action: str = "reset"
    last_action_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal models (used by simulator, grader, reward — unchanged)
# ---------------------------------------------------------------------------

class SimAction(BaseModel):
    """Internal parsed action for the simulator."""
    raw: str = Field(default="hold")
    name: str = Field(default="hold")
    value: Optional[int] = None


class IntersectionState(BaseModel):
    session_id: str
    task_id: str
    timestep: int = 0
    max_steps: int = 60
    current_phase: str = "ns"
    phase_remaining: int = 0
    yellow_remaining: int = 0
    green_duration: int = 0
    lane_health: float = 1.0
    sensor_status: str = "ONLINE"
    pedestrian_waiting: bool = False
    pedestrian_patience: int = 0
    queue_ns: int = 0
    queue_ew: int = 0
    emergency_ns: int = 0
    emergency_ew: int = 0
    moved_ns: int = 0
    moved_ew: int = 0
    total_wait_time: int = 0
    emergency_wait_time: int = 0
    invalid_actions: int = 0
    safety_violations: int = 0
    fairness_gap: float = 0.0
    max_wait_seen: int = 0
    queue_wait_ns: int = 0
    queue_wait_ew: int = 0
    backlog_total: int = 0
    phase_switches: int = 0
    flicker_events: int = 0
    starvation_events: int = 0
    stability_penalty: float = 0.0
    emergency_appearances: int = 0
    emergency_priority_hits: int = 0
    last_service_ns: int = 0
    last_service_ew: int = 0
    last_switch_timestep: int = -999
    current_phase_run: int = 0
    no_progress_steps: int = 0
    collision_detected: bool = False
    catastrophic_event: bool = False
    catastrophic_reason: Optional[str] = None
    seed: int = 0
    sensor_noise: bool = False
    ood_start: bool = False
    done: bool = False


class TaskConfig(BaseModel):
    task_id: str
    difficulty: str
    description: str
    max_steps: int
    min_green: int
    max_green: int
    base_capacity: int
    target_passed: int
    target_wait: float
    target_emergency_wait: float
    target_fairness_gap: float
    target_max_wait: float = 120.0
    target_backlog_end: float = 90.0
    target_starvation_events: float = 6.0
    target_flicker_events: float = 8.0
    target_stability_penalty: float = 12.0
    emergency_priority_window: int = 2
    starvation_threshold: int = 8
    flicker_window: int = 2
    catastrophic_emergency_wait: float = 90.0
    catastrophic_backlog: int = 220
    arrivals_ns: List[int]
    arrivals_ew: List[int]
    emergency_ns_steps: List[int]
    emergency_ew_steps: List[int]
    has_grader: bool = True


class EpisodeSummary(BaseModel):
    task_id: str
    steps: int
    moved_total: int
    avg_wait: float
    emergency_wait: float = 0.0
    max_wait: float
    backlog_end: float
    emergency_delay: float
    emergency_priority: float
    fairness_gap: float
    starvation: float
    flicker: float
    stability: float
    invalid_actions: int
    safety_violations: int = 0
    catastrophic_event: bool = False
    catastrophic_reason: Optional[str] = None


class GradeResult(BaseModel):
    score: float
    breakdown: Dict[str, float]
    reasons: List[str]
