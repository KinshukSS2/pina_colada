"""Task 1 — Easy: single-queue ticket triage."""
from __future__ import annotations

from env.environment import TicketTriageEnvironment


def make_env(seed: int = 42) -> TicketTriageEnvironment:
    """Factory used by the evaluation runner."""
    env = TicketTriageEnvironment()
    env.reset(task_id="easy", seed=seed)
    return env
