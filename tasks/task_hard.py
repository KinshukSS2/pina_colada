"""Task 3 — Hard: VIP customers, partial info, tight capacity."""
from __future__ import annotations

from env.environment import TicketTriageEnvironment


def make_env(seed: int = 42) -> TicketTriageEnvironment:
    """Factory used by the evaluation runner."""
    env = TicketTriageEnvironment()
    env.reset(task_id="hard", seed=seed)
    return env
