"""Dense reward function for TicketTriageEnv.

Produces a per-step scalar reward in [-1, 1] used primarily for RL
training loops.  The *grader* (not the reward) determines the official
score for the hackathon; this reward is an auxiliary signal.
"""
from __future__ import annotations

import math
from typing import Optional

from env.schemas import SimState, TaskConfig


def compute_reward(
    state: SimState,
    task: TaskConfig,
    action_name: str,
    params: dict,
    valid: bool,
    success: bool,
    _reason: Optional[str],
) -> float:
    """Return a scalar reward in [-1, 1] for the latest step."""

    r = 0.0

    # --- Invalid action penalty ---
    if not valid:
        r -= 0.30
        return _squash(r)

    # --- Action-level rewards ---
    if action_name == "assign" and success:
        dept = params.get("department", "")
        pri = params.get("priority", "")
        # Find the ticket that was just resolved (last in resolved list)
        ticket = state.resolved_tickets[-1] if state.resolved_tickets else None
        if ticket:
            if dept == ticket.true_category and pri == ticket.true_priority:
                r += 0.40  # perfect triage
            elif dept == ticket.true_category:
                r += 0.20  # correct department, wrong priority
            elif pri == ticket.true_priority:
                r += 0.10  # correct priority, wrong department
            else:
                r -= 0.10  # both wrong
            # VIP bonus
            if ticket.is_vip and dept == ticket.true_category:
                r += 0.15
            # SLA urgency bonus — handling near-deadline tickets well
            sla_remaining = ticket.sla_deadline - ticket.wait_time
            if sla_remaining <= 3 and dept == ticket.true_category:
                r += 0.10
    elif action_name == "escalate" and success:
        r += 0.05  # acceptable action
    elif action_name == "defer":
        r -= 0.05  # mildly negative — delays resolution
    elif action_name == "skip":
        # Skip is bad when there are pending tickets
        pending = len(state.pending_tickets)
        if pending > 0:
            r -= 0.15
        else:
            r += 0.02  # correct to skip when queue is empty
    elif action_name == "resolve" and success:
        r += 0.25

    # --- SLA pressure ---
    new_breaches = state.sla_breaches  # already incremented by simulator
    if new_breaches > 0:
        breach_rate = min(new_breaches / max(len(state.all_tickets), 1), 1.0)
        r -= 0.20 * breach_rate

    # --- Queue health bonus ---
    pending = len(state.pending_tickets)
    if pending == 0:
        r += 0.05
    elif pending > task.agent_capacity * 0.8:
        r -= 0.10

    return _squash(r)


def _squash(x: float) -> float:
    """Squash to (-1, 1) via tanh."""
    return float(math.tanh(x))
