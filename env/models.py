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
    queue_ns: int
    queue_ew: int
    emergency_ns: int
    emergency_ew: int
    moved_ns: int
    moved_ew: int
    total_wait_time: int
    emergency_wait_time: int
    invalid_actions: int
    fairness_gap: float
    last_action_valid: bool
    valid_actions: List[str]


class StepInfo(BaseModel):
    action: str
    action_valid: bool
    reason: Optional[str] = None
    task_id: str
    score_estimate: float


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
    queue_ns: int = 0
    queue_ew: int = 0
    emergency_ns: int = 0
    emergency_ew: int = 0
    moved_ns: int = 0
    moved_ew: int = 0
    total_wait_time: int = 0
    emergency_wait_time: int = 0
    invalid_actions: int = 0
    fairness_gap: float = 0.0
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
    arrivals_ns: List[int]
    arrivals_ew: List[int]
    emergency_ns_steps: List[int]
    emergency_ew_steps: List[int]


class EpisodeSummary(BaseModel):
    task_id: str
    steps: int
    moved_total: int
    avg_wait: float
    emergency_wait: float
    fairness_gap: float
    invalid_actions: int


class ResetRequest(BaseModel):
    session_id: Optional[str] = None
    task_id: str = "easy"


class StepRequest(BaseModel):
    session_id: str
    action: str


class StateRequest(BaseModel):
    session_id: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[Dict[str, str]] = None
