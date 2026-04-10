"""Task-specific configuration presets."""
from __future__ import annotations

from config.env_config import EnvConfig, SimConfig


def easy_config() -> EnvConfig:
    """Task 1: single-queue ticket triage — assign priority and route."""
    return EnvConfig(
        n_departments=3,
        departments=["billing", "technical", "general"],
        task_id="easy",
        sim=SimConfig(
            max_steps=20,
            arrival_rate_base=0.35,
            arrival_rate_noise=0.10,
            max_queue_size=30,
            sla_warning_frac=0.75,
            escalation_prob=0.0,
            vip_prob=0.0,
            info_hidden_prob=0.0,
            agent_capacity=999,
            seed=42,
        ),
    )


def medium_config() -> EnvConfig:
    """Task 2: multi-queue with SLA deadlines and escalation."""
    return EnvConfig(
        n_departments=4,
        departments=["billing", "technical", "general", "account"],
        task_id="medium",
        sim=SimConfig(
            max_steps=20,
            arrival_rate_base=0.50,
            arrival_rate_noise=0.15,
            max_queue_size=40,
            sla_warning_frac=0.70,
            sla_breach_penalty=0.7,
            escalation_prob=0.08,
            vip_prob=0.0,
            info_hidden_prob=0.0,
            agent_capacity=10,
            seed=42,
        ),
    )


def hard_config() -> EnvConfig:
    """Task 3: VIP customers, overlapping SLAs, resource limits, partial info."""
    return EnvConfig(
        n_departments=5,
        departments=["billing", "technical", "general", "account", "security"],
        task_id="hard",
        sim=SimConfig(
            max_steps=20,
            arrival_rate_base=0.60,
            arrival_rate_noise=0.20,
            max_queue_size=50,
            sla_warning_frac=0.65,
            sla_breach_penalty=0.9,
            escalation_prob=0.12,
            vip_prob=0.10,
            vip_sla_multiplier=0.5,
            info_hidden_prob=0.10,
            agent_capacity=6,
            seed=42,
        ),
    )
