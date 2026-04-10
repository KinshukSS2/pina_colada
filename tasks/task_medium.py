"""Task 2 — Medium: multi-queue with SLA deadlines and escalation."""
from __future__ import annotations

from env.environment import TicketTriageEnvironment


def make_env(seed: int = 42) -> TicketTriageEnvironment:
    """Factory used by the evaluation runner."""
    env = TicketTriageEnvironment()
    env.reset(task_id="medium", seed=seed)
    return env
