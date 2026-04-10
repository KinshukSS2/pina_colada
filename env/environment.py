"""OpenEnv-compliant TicketTriageEnv — the main environment class.

Implements the openenv.core.env_server.types.Environment ABC.
Manages stateful sessions via module-level _SESSIONS dict.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional, Tuple

from openenv.core.env_server import Environment  # pylint: disable=import-error

from config.task_configs import easy_config, medium_config, hard_config
from env.reward import compute_reward
from env.schemas import (
    SimState,
    TaskConfig,
    Ticket,
    TriageAction,
    TriageObservation,
)
from env.simulator import (
    build_action_mask,
    inject_tickets,
    parse_action,
    select_current_ticket,
    simulate_step,
)
from graders.easy_grader import EasyGrader
from graders.medium_grader import MediumGrader
from graders.hard_grader import HardGrader


# ---------------------------------------------------------------------------
# Task catalog
# ---------------------------------------------------------------------------

_TASK_CONFIGS: Dict[str, TaskConfig] = {}


def _build_task(env_cfg, difficulty: str, desc: str, arrival_pattern, **extra) -> TaskConfig:
    return TaskConfig(
        task_id=env_cfg.task_id,
        difficulty=difficulty,
        description=desc,
        max_steps=env_cfg.sim.max_steps,
        n_departments=env_cfg.n_departments,
        departments=env_cfg.departments,
        priorities=env_cfg.priorities,
        arrival_pattern=arrival_pattern,
        agent_capacity=env_cfg.sim.agent_capacity,
        **extra,
    )


def task_catalog() -> Dict[str, TaskConfig]:
    """Lazy-initialise and return the task catalog."""
    if not _TASK_CONFIGS:
        ec = easy_config()
        _TASK_CONFIGS["easy"] = _build_task(
            ec, "easy",
            "Single-queue ticket triage — assign priority and route tickets to the correct department.",
            arrival_pattern=[1] * ec.sim.max_steps,
            target_accuracy=0.75,
            target_sla_compliance=0.90,
            target_resolution_rate=0.70,
        )
        mc = medium_config()
        _TASK_CONFIGS["medium"] = _build_task(
            mc, "medium",
            "Multi-queue triage with SLA deadlines, escalation, and department capacity.",
            arrival_pattern=[1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2],
            target_accuracy=0.70,
            target_sla_compliance=0.80,
            target_resolution_rate=0.60,
            target_avg_wait=6.0,
            vip_steps=[],
            escalation_steps=[3, 7, 12, 16],
            hidden_info_steps=[],
        )
        hc = hard_config()
        _TASK_CONFIGS["hard"] = _build_task(
            hc, "hard",
            "VIP customers, tight SLAs, partial information, and limited department capacity.",
            arrival_pattern=[2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2],
            target_accuracy=0.65,
            target_sla_compliance=0.70,
            target_resolution_rate=0.55,
            target_vip_compliance=0.85,
            target_avg_wait=7.0,
            vip_steps=[2, 5, 9, 14, 18],
            escalation_steps=[1, 4, 8, 11, 15, 19],
            hidden_info_steps=[3, 6, 10, 13, 17],
        )
    return _TASK_CONFIGS


def get_task(task_id: str) -> TaskConfig:
    catalog = task_catalog()
    if task_id not in catalog:
        raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(catalog.keys())}")
    return catalog[task_id]


# ---------------------------------------------------------------------------
# Grader dispatch
# ---------------------------------------------------------------------------

_GRADERS = {
    "easy": EasyGrader(),
    "medium": MediumGrader(),
    "hard": HardGrader(),
}


def _grade_trajectory(task_id: str, trajectory: list) -> Tuple[float, Dict[str, float], list]:
    grader = _GRADERS.get(task_id, _GRADERS["easy"])
    score = grader.grade(trajectory)
    return score, {"final_score": score}, []


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

_SESSIONS: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TicketTriageEnvironment(Environment):
    """Customer-support ticket triage environment (OpenEnv-compliant)."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    # ------------------------------------------------------------------ #
    # OpenEnv interface
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, episode_id=None, *, task_id: str = "easy", **kwargs) -> TriageObservation:
        """Reset (or create) a session and return the initial observation."""
        session_id = kwargs.get("session_id") or episode_id or uuid.uuid4().hex[:12]
        seed = seed or kwargs.get("seed", 42) or 42

        task = get_task(task_id)
        state = SimState(
            session_id=session_id,
            task_id=task_id,
            max_steps=task.max_steps,
            seed=seed,
            department_load={d: 0 for d in task.departments},
        )

        # Inject initial tickets
        inject_tickets(state, task)
        select_current_ticket(state)

        _SESSIONS[session_id] = {"state": state, "task": task}

        return self._build_observation(state, task)

    def step(self, action: TriageAction, timeout_s=None, **kwargs) -> TriageObservation:
        """Apply an action and return the next observation."""
        _ = timeout_s, kwargs  # required by Environment ABC
        sid = action.session_id or ""
        session = _SESSIONS.get(sid)
        if session is None:
            obs = self.reset(task_id="easy", session_id=sid)
            session = _SESSIONS[sid]

        state: SimState = session["state"]
        task: TaskConfig = session["task"]

        if state.done:
            return self._build_observation(state, task, done=True)

        raw = action.action
        action_name, params, valid, reason = parse_action(raw, state, task)

        success, step_reason = simulate_step(state, task, action_name, params, valid)

        # Add current ticket info to trajectory snapshot
        if state.trajectory:
            last_snap = state.trajectory[-1]
            ticket = self._current_ticket(state)
            if ticket:
                last_snap["current_ticket"] = ticket.model_dump()
            last_snap["action_taken"] = raw

        obs = self._build_observation(
            state, task,
            last_action=raw,
            last_valid=valid,
            last_reason=reason or step_reason,
        )

        # Compute reward
        reward = compute_reward(state, task, action_name, params, valid, success, step_reason)
        obs.reward = reward

        return obs

    @property
    def state(self) -> Dict[str, Any]:
        """Return the current state as a dict."""
        # Return the last active session's state
        if not _SESSIONS:
            return {}
        last_key = list(_SESSIONS.keys())[-1]
        s = _SESSIONS[last_key].get("state")
        if s is None:
            return {}
        return s.model_dump()

    def get_state(self, *, session_id: str = "") -> Dict[str, Any]:
        """Return the current state for a specific session."""
        session = _SESSIONS.get(session_id, {})
        s = session.get("state")
        if s is None:
            return {"error": "session_not_found"}
        return s.model_dump()

    # ------------------------------------------------------------------ #
    # Legacy endpoints (session-based HTTP)
    # ------------------------------------------------------------------ #

    def legacy_reset(
        self,
        task_id: str = "easy",
        session_id: Optional[str] = None,
        seed: Optional[int] = None,
        **_kwargs,
    ) -> Dict[str, Any]:
        sid = session_id or uuid.uuid4().hex[:12]
        obs = self.reset(task_id=task_id, session_id=sid, seed=seed or 42)
        return {
            "observation": obs.model_dump(),
            "reward": 0.0,
            "done": False,
            "info": {"session_id": sid, "task_id": task_id},
        }

    def legacy_step(self, session_id: str, action_text: str) -> Dict[str, Any]:
        obs = self.step(TriageAction(action=action_text, session_id=session_id))
        session = _SESSIONS.get(session_id, {})
        s = session.get("state")
        done = s.done if s else True
        return {
            "observation": obs.model_dump(),
            "reward": getattr(obs, "reward", 0.0),
            "done": done,
            "info": {
                "session_id": session_id,
                "action_valid": obs.last_action_valid,
                "action_reason": obs.last_action_reason,
            },
        }

    def legacy_state(self, session_id: str) -> Dict[str, Any]:
        session = _SESSIONS.get(session_id)
        if session is None:
            return {"error": "session_not_found"}
        s: SimState = session["state"]
        t: TaskConfig = session["task"]
        obs = self._build_observation(s, t)
        return {
            "observation": obs.model_dump(),
            "reward": 0.0,
            "done": s.done,
            "info": {"session_id": session_id},
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _current_ticket(state: SimState) -> Optional[Ticket]:
        if state.current_ticket_idx < 0 or state.current_ticket_idx >= len(state.pending_tickets):
            return None
        return state.pending_tickets[state.current_ticket_idx]

    def _build_observation(
        self,
        state: SimState,
        task: TaskConfig,
        done: bool = False,
        last_action: str = "reset",
        last_valid: bool = True,
        last_reason: Optional[str] = None,
    ) -> TriageObservation:
        ticket = self._current_ticket(state)
        mask = build_action_mask(state, task)

        # Grading
        score, breakdown, reasons = _grade_trajectory(state.task_id, state.trajectory)

        obs = TriageObservation(
            session_id=state.session_id,
            task_id=state.task_id,
            timestep=state.timestep,
            max_steps=task.max_steps,
            # Current ticket
            current_ticket_id=ticket.ticket_id if ticket else -1,
            current_ticket_subject=ticket.subject if ticket else "",
            current_ticket_category_hint=ticket.category_hint if ticket else "",
            current_ticket_urgency_hint=ticket.urgency_hint if ticket else "",
            current_ticket_is_vip=ticket.is_vip if ticket else False,
            current_ticket_sla_remaining=max(0, ticket.sla_deadline - ticket.wait_time) if ticket else 0,
            current_ticket_wait_time=ticket.wait_time if ticket else 0,
            current_ticket_info_hidden=ticket.info_hidden if ticket else False,
            # Queue
            queue_size=len(state.pending_tickets),
            queue_by_priority=self._queue_by_priority(state),
            queue_by_department={},
            sla_breaches=state.sla_breaches,
            sla_warnings=state.sla_warnings,
            # Department
            department_load=dict(state.department_load),
            department_capacity={d: task.agent_capacity for d in task.departments},
            # Cumulative
            tickets_resolved=state.tickets_resolved,
            tickets_assigned=state.tickets_assigned,
            tickets_escalated=state.tickets_escalated,
            tickets_deferred=state.tickets_deferred,
            correct_assignments=state.correct_assignments,
            incorrect_assignments=state.incorrect_assignments,
            vip_breaches=state.vip_breaches,
            total_wait_time=state.total_wait_time,
            max_wait_seen=state.max_wait_seen,
            # Actions
            action_mask=mask,
            valid_actions=list(mask.keys()),
            invalid_actions=state.invalid_actions,
            # Grading
            score_estimate=score,
            grader_breakdown=breakdown,
            grader_reasons=reasons,
            last_action=last_action,
            last_action_valid=last_valid,
            last_action_reason=last_reason,
            seed_value=state.seed,
            done=done or state.done,
        )
        return obs

    @staticmethod
    def _queue_by_priority(state: SimState) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in state.pending_tickets:
            counts[t.true_priority] = counts.get(t.true_priority, 0) + 1
        return counts
