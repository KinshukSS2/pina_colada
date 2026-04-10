"""Pydantic schemas for TicketTriageEnv — actions, observations, state."""
from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action as OpenEnvAction  # pylint: disable=import-error
from openenv.core.env_server.types import Observation as OpenEnvObservation  # pylint: disable=import-error
from pydantic import BaseModel, Field  # pylint: disable=import-error


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Priority(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class TicketCategory(IntEnum):
    BILLING = 0
    TECHNICAL = 1
    GENERAL = 2
    ACCOUNT = 3
    SECURITY = 4


# ---------------------------------------------------------------------------
# OpenEnv-compliant Action & Observation
# ---------------------------------------------------------------------------

class TriageAction(OpenEnvAction):
    """Action sent by the agent to the ticket triage environment."""
    action: str = Field(
        default="skip",
        description=(
            "Triage action: assign:<priority>:<department>, escalate, "
            "defer, skip, resolve:<ticket_id>"
        ),
    )
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class TriageObservation(OpenEnvObservation):
    """Observation returned by the ticket triage environment."""
    session_id: str = ""
    task_id: str = ""
    timestep: int = 0
    max_steps: int = 30
    # Current ticket to triage
    current_ticket_id: int = -1
    current_ticket_subject: str = ""
    current_ticket_category_hint: str = ""
    current_ticket_urgency_hint: str = ""
    current_ticket_is_vip: bool = False
    current_ticket_sla_remaining: int = 0
    current_ticket_wait_time: int = 0
    current_ticket_info_hidden: bool = False
    # Queue state
    queue_size: int = 0
    queue_by_priority: Dict[str, int] = Field(default_factory=dict)
    queue_by_department: Dict[str, int] = Field(default_factory=dict)
    sla_breaches: int = 0
    sla_warnings: int = 0
    # Department load
    department_load: Dict[str, int] = Field(default_factory=dict)
    department_capacity: Dict[str, int] = Field(default_factory=dict)
    # Cumulative metrics
    tickets_resolved: int = 0
    tickets_assigned: int = 0
    tickets_escalated: int = 0
    tickets_deferred: int = 0
    correct_assignments: int = 0
    incorrect_assignments: int = 0
    vip_breaches: int = 0
    total_wait_time: int = 0
    max_wait_seen: int = 0
    # Action mask and valid actions
    action_mask: Dict[str, bool] = Field(default_factory=dict)
    valid_actions: List[str] = Field(default_factory=list)
    invalid_actions: int = 0
    # Grader info
    score_estimate: float = 0.0
    grader_breakdown: Dict[str, float] = Field(default_factory=dict)
    grader_reasons: List[str] = Field(default_factory=list)
    last_action: str = "reset"
    last_action_valid: bool = True
    last_action_reason: Optional[str] = None
    seed_value: int = 0


# ---------------------------------------------------------------------------
# Internal simulation models
# ---------------------------------------------------------------------------

class Ticket(BaseModel):
    """A single support ticket in the simulation."""
    ticket_id: int
    subject: str
    true_category: str  # ground truth department
    true_priority: str  # ground truth priority
    category_hint: str  # what agent sees (may be noisy)
    urgency_hint: str   # what agent sees
    is_vip: bool = False
    sla_deadline: int = 20  # steps until SLA breach
    arrival_step: int = 0
    wait_time: int = 0
    assigned_priority: Optional[str] = None
    assigned_department: Optional[str] = None
    resolved: bool = False
    escalated: bool = False
    deferred: bool = False
    sla_breached: bool = False
    info_hidden: bool = False


class SimState(BaseModel):
    """Full internal simulation state."""
    session_id: str
    task_id: str
    timestep: int = 0
    max_steps: int = 30
    seed: int = 42
    done: bool = False
    # Ticket queues
    pending_tickets: List[Ticket] = Field(default_factory=list)
    resolved_tickets: List[Ticket] = Field(default_factory=list)
    all_tickets: List[Ticket] = Field(default_factory=list)
    current_ticket_idx: int = -1
    next_ticket_id: int = 0
    # Cumulative counters
    tickets_assigned: int = 0
    tickets_resolved: int = 0
    tickets_escalated: int = 0
    tickets_deferred: int = 0
    correct_assignments: int = 0
    incorrect_assignments: int = 0
    invalid_actions: int = 0
    sla_breaches: int = 0
    sla_warnings: int = 0
    vip_breaches: int = 0
    total_wait_time: int = 0
    max_wait_seen: int = 0
    department_load: Dict[str, int] = Field(default_factory=dict)
    # Trajectory for grading
    trajectory: List[Dict[str, Any]] = Field(default_factory=list)


class TaskConfig(BaseModel):
    """Task-level configuration."""
    task_id: str
    difficulty: str
    description: str
    max_steps: int
    n_departments: int
    departments: List[str]
    priorities: List[str] = Field(default_factory=lambda: ["low", "medium", "high", "critical"])
    # Grading targets
    target_accuracy: float = 0.80
    target_sla_compliance: float = 0.90
    target_resolution_rate: float = 0.70
    target_vip_compliance: float = 0.95
    target_avg_wait: float = 5.0
    target_max_wait: float = 15.0
    # Ticket generation parameters
    ticket_subjects: List[str] = Field(default_factory=list)
    ticket_categories: List[str] = Field(default_factory=list)
    ticket_urgencies: List[str] = Field(default_factory=list)
    sla_deadlines: List[int] = Field(default_factory=list)
    arrival_pattern: List[int] = Field(default_factory=list)  # tickets per step
    vip_steps: List[int] = Field(default_factory=list)
    escalation_steps: List[int] = Field(default_factory=list)
    hidden_info_steps: List[int] = Field(default_factory=list)
    agent_capacity: int = 999
    has_grader: bool = True


class EpisodeSummary(BaseModel):
    """Summary of an episode for grading."""
    task_id: str
    steps: int
    tickets_seen: int
    tickets_assigned: int
    tickets_resolved: int
    tickets_escalated: int
    tickets_deferred: int
    correct_assignments: int
    incorrect_assignments: int
    accuracy: float
    sla_breaches: int
    sla_compliance: float
    vip_breaches: int
    vip_compliance: float
    avg_wait: float
    max_wait: float
    invalid_actions: int
    resolution_rate: float
    department_balance: float  # Jain's fairness across departments


class GradeResult(BaseModel):
    """Result of grading an episode."""
    score: float
    breakdown: Dict[str, float]
    reasons: List[str]
