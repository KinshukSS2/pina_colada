"""Core ticket triage simulation logic — deterministic given the same seed."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from env.schemas import Ticket, SimState, TaskConfig


VALID_ACTIONS = [
    "assign:<priority>:<department>",
    "escalate",
    "defer",
    "skip",
    "resolve:<ticket_id>",
]

# Ticket subject templates indexed by category
_SUBJECTS: Dict[str, List[str]] = {
    "billing": [
        "Incorrect charge on invoice",
        "Refund not processed",
        "Subscription billing error",
        "Payment method declined",
        "Duplicate charge detected",
    ],
    "technical": [
        "Application crash on login",
        "API returning 500 errors",
        "Data sync failure",
        "Performance degradation",
        "SSL certificate expired",
    ],
    "general": [
        "How to update account info",
        "Feature request submission",
        "Feedback on new UI",
        "Need documentation link",
        "Account settings question",
    ],
    "account": [
        "Cannot reset password",
        "Two-factor auth locked out",
        "Account suspended unexpectedly",
        "Merge duplicate accounts",
        "Change account ownership",
    ],
    "security": [
        "Suspicious login detected",
        "Data breach concern",
        "Unauthorized access report",
        "Phishing email received",
        "API key compromised",
    ],
}

_URGENCY_MAP: Dict[str, str] = {
    "low": "Not urgent — can wait",
    "medium": "Moderate — within business hours",
    "high": "Urgent — needs quick resolution",
    "critical": "CRITICAL — immediate attention required",
}


def _deterministic_int(seed: int, step: int, salt: int) -> int:
    """Deterministic PRNG returning a non-negative integer."""
    return ((seed * 1103515245 + (step + 1) * 12345 + salt * 1013904223) & 0x7FFFFFFF)


def _deterministic_u01(seed: int, step: int, salt: int) -> float:
    """Deterministic float in [0, 1)."""
    return (_deterministic_int(seed, step, salt) % 10000) / 10000.0


def _pick(items: list, seed: int, step: int, salt: int):
    """Pick a deterministic item from a list."""
    idx = _deterministic_int(seed, step, salt) % len(items)
    return items[idx]


# ---------------------------------------------------------------------------
# Ticket generation
# ---------------------------------------------------------------------------

def generate_ticket(
    state: SimState,
    task: TaskConfig,
    ticket_offset: int = 0,
) -> Ticket:
    """Generate a single deterministic ticket."""
    seed = state.seed
    step = state.timestep
    salt_base = state.next_ticket_id * 7 + ticket_offset * 13

    # Pick true category
    true_cat = _pick(task.departments, seed, step, salt_base + 1)
    # Pick true priority
    true_pri = _pick(task.priorities, seed, step, salt_base + 2)
    # Subject
    cat_subjects = _SUBJECTS.get(true_cat, _SUBJECTS["general"])
    subject = _pick(cat_subjects, seed, step, salt_base + 3)

    # Hints (may be noisy for harder tasks)
    category_hint = true_cat
    urgency_hint = _URGENCY_MAP.get(true_pri, _URGENCY_MAP["medium"])

    # Check if info should be hidden
    info_hidden = step in task.hidden_info_steps

    if info_hidden:
        category_hint = "unknown"
        urgency_hint = "Information unavailable — customer did not specify"

    # VIP check
    is_vip = step in task.vip_steps

    # SLA deadline based on priority
    sla_map = {"low": 15, "medium": 10, "high": 6, "critical": 3}
    sla_base = sla_map.get(true_pri, 10)
    if is_vip:
        sla_base = max(2, int(sla_base * 0.5))

    # Deterministic jitter on SLA
    jitter = (_deterministic_int(seed, step, salt_base + 4) % 3) - 1
    sla_deadline = max(2, sla_base + jitter)

    ticket = Ticket(
        ticket_id=state.next_ticket_id,
        subject=subject,
        true_category=true_cat,
        true_priority=true_pri,
        category_hint=category_hint,
        urgency_hint=urgency_hint,
        is_vip=is_vip,
        sla_deadline=sla_deadline,
        arrival_step=step,
        info_hidden=info_hidden,
    )
    state.next_ticket_id += 1
    return ticket


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_action(raw_action: str, _state: SimState, task: TaskConfig) -> Tuple[str, dict, bool, Optional[str]]:
    """Parse a raw action string into (action_name, params, valid, reason).

    Returns:
        (name, params_dict, is_valid, rejection_reason)
    """
    if raw_action is None:
        return "skip", {}, False, "empty_action"

    text = str(raw_action).strip().lower()
    if not text:
        return "skip", {}, False, "blank_action"

    if text == "skip":
        return "skip", {}, True, None

    if text == "defer":
        return "defer", {}, True, None

    if text == "escalate":
        return "escalate", {}, True, None

    if text.startswith("resolve:"):
        parts = text.split(":", 1)
        if len(parts) == 2 and parts[1].strip().isdigit():
            tid = int(parts[1].strip())
            return "resolve", {"ticket_id": tid}, True, None
        return "skip", {}, False, "invalid_resolve_format"

    if text.startswith("assign:"):
        parts = text.split(":")
        if len(parts) == 3:
            pri = parts[1].strip()
            dept = parts[2].strip()
            if pri in task.priorities and dept in task.departments:
                return "assign", {"priority": pri, "department": dept}, True, None
            if pri not in task.priorities:
                return "skip", {}, False, f"invalid_priority_{pri}"
            if dept not in task.departments:
                return "skip", {}, False, f"invalid_department_{dept}"
        return "skip", {}, False, "invalid_assign_format"

    return "skip", {}, False, "unknown_action"


# ---------------------------------------------------------------------------
# Step logic
# ---------------------------------------------------------------------------

def inject_tickets(state: SimState, task: TaskConfig) -> None:
    """Inject new tickets at the current timestep (deterministic)."""
    idx = min(state.timestep, len(task.arrival_pattern) - 1)
    n_arrivals = task.arrival_pattern[idx]
    for i in range(n_arrivals):
        ticket = generate_ticket(state, task, ticket_offset=i)
        state.pending_tickets.append(ticket)
        state.all_tickets.append(ticket)


def advance_tickets(state: SimState, _task: TaskConfig) -> None:
    """Advance wait times and check SLA breaches."""
    for ticket in state.pending_tickets:
        ticket.wait_time += 1
        state.total_wait_time += 1
        state.max_wait_seen = max(state.max_wait_seen, ticket.wait_time)
        # SLA check
        if ticket.wait_time >= ticket.sla_deadline and not ticket.sla_breached:
            ticket.sla_breached = True
            state.sla_breaches += 1
            if ticket.is_vip:
                state.vip_breaches += 1
        elif ticket.wait_time >= int(ticket.sla_deadline * 0.75):
            state.sla_warnings += 1


def select_current_ticket(state: SimState) -> None:
    """Select the next ticket for the agent to triage."""
    if not state.pending_tickets:
        state.current_ticket_idx = -1
        return
    # Pick the oldest unassigned ticket (FIFO with priority boost)
    best_idx = 0
    best_score = -1.0
    for i, t in enumerate(state.pending_tickets):
        if t.assigned_department is not None:
            continue
        # Score: higher = more urgent to show
        priority_boost = {"low": 0, "medium": 1, "high": 3, "critical": 6}.get(t.true_priority, 0)
        wait_score = t.wait_time * 2.0
        vip_boost = 10.0 if t.is_vip else 0.0
        sla_urgency = max(0, t.sla_deadline - t.wait_time)
        sla_score = (20.0 - sla_urgency) * 1.5
        score = wait_score + priority_boost + vip_boost + sla_score
        if score > best_score:
            best_score = score
            best_idx = i
    state.current_ticket_idx = best_idx


def apply_action(
    state: SimState,
    task: TaskConfig,
    action_name: str,
    params: dict,
    valid: bool,
) -> Tuple[bool, Optional[str]]:
    """Apply an action to the current state. Returns (success, reason)."""
    if not valid:
        state.invalid_actions += 1
        return False, "invalid_action"

    if state.current_ticket_idx < 0 or state.current_ticket_idx >= len(state.pending_tickets):
        if action_name == "skip":
            return True, None
        state.invalid_actions += 1
        return False, "no_ticket_available"

    ticket = state.pending_tickets[state.current_ticket_idx]

    if action_name == "skip":
        return True, None

    if action_name == "defer":
        ticket.deferred = True
        state.tickets_deferred += 1
        # Move to end of queue
        state.pending_tickets.pop(state.current_ticket_idx)
        state.pending_tickets.append(ticket)
        return True, None

    if action_name == "escalate":
        ticket.escalated = True
        state.tickets_escalated += 1
        state.pending_tickets.pop(state.current_ticket_idx)
        state.resolved_tickets.append(ticket)
        return True, None

    if action_name == "assign":
        pri = params.get("priority", "medium")
        dept = params.get("department", "general")

        # Check department capacity
        current_load = state.department_load.get(dept, 0)
        if current_load >= task.agent_capacity:
            state.invalid_actions += 1
            return False, f"department_{dept}_at_capacity"

        ticket.assigned_priority = pri
        ticket.assigned_department = dept
        state.tickets_assigned += 1
        state.department_load[dept] = current_load + 1

        # Check correctness
        correct_dept = (dept == ticket.true_category)
        correct_pri = (pri == ticket.true_priority)
        if correct_dept and correct_pri:
            state.correct_assignments += 1
        else:
            state.incorrect_assignments += 1

        # Mark as resolved (assigned = in-progress toward resolution)
        ticket.resolved = True
        state.tickets_resolved += 1
        state.pending_tickets.pop(state.current_ticket_idx)
        state.resolved_tickets.append(ticket)
        return True, None

    if action_name == "resolve":
        tid = params.get("ticket_id", -1)
        found = None
        for i, t in enumerate(state.pending_tickets):
            if t.ticket_id == tid:
                found = i
                break
        if found is not None:
            t = state.pending_tickets.pop(found)
            t.resolved = True
            state.tickets_resolved += 1
            state.resolved_tickets.append(t)
            return True, None
        state.invalid_actions += 1
        return False, "ticket_not_found"

    return False, "unhandled_action"


def simulate_step(
    state: SimState,
    task: TaskConfig,
    action_name: str,
    params: dict,
    valid: bool,
) -> Tuple[bool, Optional[str]]:
    """Execute one simulation step. Returns (action_success, reason)."""
    # 1. Apply the agent's action
    success, reason = apply_action(state, task, action_name, params, valid)

    # 2. Inject new tickets
    inject_tickets(state, task)

    # 3. Advance waiting/SLA
    advance_tickets(state, task)

    # 4. Select next ticket to present
    select_current_ticket(state)

    # 5. Record trajectory snapshot
    snapshot = _build_snapshot(state, task, action_name, params, valid, success, reason)
    state.trajectory.append(snapshot)

    # 6. Advance timestep
    state.timestep += 1
    if state.timestep >= task.max_steps:
        state.done = True

    return success, reason


def _build_snapshot(
    state: SimState,
    _task: TaskConfig,
    action_name: str,
    params: dict,
    valid: bool,
    success: bool,
    reason: Optional[str],
) -> dict:
    """Build a trajectory snapshot for grading."""
    return {
        "timestep": state.timestep,
        "action": action_name,
        "params": params,
        "valid": valid,
        "success": success,
        "reason": reason,
        "state_snapshot": {
            "queue_size": len(state.pending_tickets),
            "tickets_assigned": state.tickets_assigned,
            "tickets_resolved": state.tickets_resolved,
            "correct_assignments": state.correct_assignments,
            "incorrect_assignments": state.incorrect_assignments,
            "sla_breaches": state.sla_breaches,
            "sla_warnings": state.sla_warnings,
            "vip_breaches": state.vip_breaches,
            "total_wait_time": state.total_wait_time,
            "max_wait_seen": state.max_wait_seen,
            "department_load": dict(state.department_load),
            "pending_count": len(state.pending_tickets),
            "resolved_count": len(state.resolved_tickets),
        },
    }


def build_action_mask(state: SimState, task: TaskConfig) -> Dict[str, bool]:
    """Build action mask for current state."""
    has_ticket = (
        state.current_ticket_idx >= 0
        and state.current_ticket_idx < len(state.pending_tickets)
    )
    mask: Dict[str, bool] = {
        "skip": True,
        "defer": has_ticket,
        "escalate": has_ticket,
    }
    # assign for each priority-department combo
    for dept in task.departments:
        load = state.department_load.get(dept, 0)
        at_capacity = load >= task.agent_capacity
        for pri in task.priorities:
            key = f"assign:{pri}:{dept}"
            mask[key] = has_ticket and not at_capacity

    # resolve for pending tickets
    for t in state.pending_tickets:
        mask[f"resolve:{t.ticket_id}"] = True

    return mask
