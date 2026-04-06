from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Action(BaseModel):
    raw: str = Field(default="hold")
    name: str = Field(default="hold")
    value: Optional[int] = None


class Observation(BaseModel):
    session_id: str
    task_id: str
    timestep: int
    max_steps: int
    current_phase: str
    phase_remaining: int
    yellow_active: bool = False
    green_duration: int = 0
    ascii_minimap: str = ""
    sensor_status: str = "ONLINE"
    pedestrian_waiting: bool = False
    action_mask: Dict[str, bool]
    queue_ns: int
    queue_ew: int
    emergency_ns: int
    emergency_ew: int
    lane_health: float = 1.0
    moved_ns: int
    moved_ew: int
    total_wait_time: int
    emergency_wait_time: int
    invalid_actions: int
    fairness_gap: float
    max_wait_seen: int
    backlog_total: int
    phase_switches: int
    flicker_events: int
    starvation_events: int
    stability_penalty: float
    catastrophic_event: bool
    seed: int
    last_action_valid: bool
    valid_actions: List[str]


class StepInfo(BaseModel):
    action: str
    action_valid: bool
    fallback_status: str = Field(default="unknown")
    agent_reason: Optional[str] = Field(default=None)
    reason: Optional[str] = None
    task_id: str
    score_estimate: float
    reasoning_logs: List[str] = Field(default_factory=list)
    grader_breakdown: Dict[str, float] = Field(default_factory=dict)
    grader_reasons: List[str] = Field(default_factory=list)


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: StepInfo


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


class ResetRequest(BaseModel):
    session_id: Optional[str] = None
    task_id: str = "easy"
    seed: Optional[int] = None
    sensor_noise: bool = False
    ood_start: bool = False


class GradeResult(BaseModel):
    score: float
    breakdown: Dict[str, float]
    reasons: List[str]


class StepRequest(BaseModel):
    session_id: str
    action: str
    fallback_status: Optional[str] = "unknown"
    agent_reason: Optional[str] = None


class StateRequest(BaseModel):
    session_id: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[Dict[str, str]] = None
