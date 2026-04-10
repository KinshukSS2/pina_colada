"""Configuration dataclasses for TicketTriageEnv."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SimConfig:
    """Low-level simulation parameters."""
    max_steps: int = 30
    # Ticket arrival (Poisson-like, deterministic via seed)
    arrival_rate_base: float = 0.4
    arrival_rate_noise: float = 0.1
    # Queues
    max_queue_size: int = 50
    # SLA
    sla_warning_frac: float = 0.75  # fraction of SLA before warning
    sla_breach_penalty: float = 0.5
    # Escalation
    escalation_prob: float = 0.0  # per-step probability of escalation event
    # VIP
    vip_prob: float = 0.0
    vip_sla_multiplier: float = 0.5  # VIP SLAs are tighter
    # Partial observability
    info_hidden_prob: float = 0.0  # probability ticket description is masked
    # Resource constraints
    agent_capacity: int = 999  # max concurrent tickets per department
    # Determinism
    seed: int = 42


@dataclass
class EnvConfig:
    """Top-level environment configuration."""
    n_departments: int = 3
    departments: List[str] = field(default_factory=lambda: ["billing", "technical", "general"])
    priorities: List[str] = field(default_factory=lambda: ["low", "medium", "high", "critical"])
    task_id: str = "easy"
    sim: SimConfig = field(default_factory=SimConfig)
