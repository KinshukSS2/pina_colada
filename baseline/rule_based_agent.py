"""Rule-based baseline agent for ticket triage.

Acts as a reference policy:
- Assigns tickets with heuristic priority/department guessing
- Escalates VIP tickets with high urgency
- Defers when capacity is full
"""
from __future__ import annotations

from typing import Any, Dict


def baseline_action(observation: Dict[str, Any]) -> str:
    """Given an observation dict, return a triage action string."""
    tid = observation.get("current_ticket_id", -1)
    if tid < 0:
        return "skip"

    subject = str(observation.get("current_ticket_subject", "")).lower()
    hint = str(observation.get("current_ticket_category_hint", "general")).lower()
    urgency = str(observation.get("current_ticket_urgency_hint", "")).lower()
    is_vip = observation.get("current_ticket_is_vip", False)
    sla_rem = observation.get("current_ticket_sla_remaining", 99)
    info_hidden = observation.get("current_ticket_info_hidden", False)

    dept_load = observation.get("department_load", {})
    dept_cap = observation.get("department_capacity", {})

    # --- Priority heuristic ---
    if "critical" in urgency or "immediate" in urgency:
        priority = "critical"
    elif "urgent" in urgency or "quick" in urgency:
        priority = "high"
    elif "moderate" in urgency or "business" in urgency:
        priority = "medium"
    else:
        priority = "low"

    # VIP boost
    if is_vip and priority in ("low", "medium"):
        priority = "high"

    # --- Department heuristic ---
    if info_hidden or hint == "unknown":
        department = _guess_from_subject(subject)
    else:
        department = hint if hint in dept_load else _guess_from_subject(subject)

    # --- Capacity check ---
    load = dept_load.get(department, 0)
    cap = dept_cap.get(department, 999)
    if load >= cap:
        # Try another dept
        for d in dept_load:
            if dept_load.get(d, 0) < dept_cap.get(d, 999):
                department = d
                break
        else:
            # All full — escalate VIP, defer others
            if is_vip or sla_rem <= 2:
                return "escalate"
            return "defer"

    # --- SLA pressure ---
    if sla_rem <= 1:
        return "escalate" if is_vip else f"assign:{priority}:{department}"

    return f"assign:{priority}:{department}"


def _guess_from_subject(subject: str) -> str:
    """Guess department from subject keywords."""
    subject = subject.lower()
    if any(w in subject for w in ("invoice", "charge", "refund", "billing", "payment", "subscription")):
        return "billing"
    if any(w in subject for w in ("crash", "error", "api", "ssl", "sync", "performance", "bug")):
        return "technical"
    if any(w in subject for w in ("password", "account", "auth", "locked", "suspend", "merge", "ownership")):
        return "account"
    if any(w in subject for w in ("suspicious", "breach", "unauthorized", "phishing", "compromised", "security")):
        return "security"
    return "general"
