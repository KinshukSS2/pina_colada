from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from env.grader import compute_grade, compute_score
from env.models import (
    EpisodeSummary,
    IntersectionState,
    TaskConfig,
    TrafficAction,
    TrafficObservation,
)
from env.reward import compute_reward
from env.simulator import VALID_ACTIONS, _generate_ascii, parse_action, simulate_step
from env.tasks import get_task


# Module-level session store — persists across env instances created by the framework
_SESSIONS: Dict[str, Tuple[IntersectionState, TaskConfig]] = {}


class TrafficControlEnvironment(Environment):
    """OpenEnv-compliant traffic intersection control environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id="", step_count=0)

    @staticmethod
    def _noise_delta(seed: int, timestep: int, key: int) -> int:
        value = (seed * 1103515245 + (timestep + 1) * 12345 + key * 1013) & 0x7FFFFFFF
        return int(value % 3) - 1

    def _build_observation(
        self,
        state: IntersectionState,
        action_valid: bool,
        reward_value: float,
        is_done: bool,
        score: float,
        logs: list,
        grade_breakdown: dict,
        grade_reasons: list,
        action_name: str = "reset",
        reason: Optional[str] = None,
    ) -> TrafficObservation:
        queue_ns = state.queue_ns
        queue_ew = state.queue_ew
        emergency_ns = state.emergency_ns
        emergency_ew = state.emergency_ew
        if state.sensor_noise:
            queue_ns = max(0, queue_ns + self._noise_delta(state.seed, state.timestep, 1))
            queue_ew = max(0, queue_ew + self._noise_delta(state.seed, state.timestep, 2))
            emergency_ns = max(0, emergency_ns + self._noise_delta(state.seed, state.timestep, 3))
            emergency_ew = max(0, emergency_ew + self._noise_delta(state.seed, state.timestep, 4))

        yellow_active = state.yellow_remaining > 0
        pedestrian_waiting = state.pedestrian_waiting
        action_mask = {
            "hold": True,
            "switch": not yellow_active,
            "prioritize_emergency": (not yellow_active) and (not pedestrian_waiting),
            "set_ns_green": (not yellow_active) and (not pedestrian_waiting),
            "set_ew_green": (not yellow_active) and (not pedestrian_waiting),
        }

        observed_queue_ns = queue_ns if state.sensor_status != "OFFLINE" else -1
        observed_queue_ew = queue_ew if state.sensor_status != "OFFLINE" else -1

        return TrafficObservation(
            # OpenEnv base fields
            done=is_done,
            reward=reward_value,
            # Domain fields
            session_id=state.session_id,
            task_id=state.task_id,
            timestep=state.timestep,
            max_steps=state.max_steps,
            current_phase=state.current_phase,
            phase_remaining=state.phase_remaining,
            yellow_active=yellow_active,
            green_duration=state.green_duration,
            ascii_minimap=_generate_ascii(state),
            sensor_status=state.sensor_status,
            pedestrian_waiting=state.pedestrian_waiting,
            queue_ns=observed_queue_ns,
            queue_ew=observed_queue_ew,
            emergency_ns=emergency_ns,
            emergency_ew=emergency_ew,
            lane_health=state.lane_health,
            moved_ns=state.moved_ns,
            moved_ew=state.moved_ew,
            total_wait_time=state.total_wait_time,
            emergency_wait_time=state.emergency_wait_time,
            invalid_actions=state.invalid_actions,
            fairness_gap=state.fairness_gap,
            max_wait_seen=state.max_wait_seen,
            backlog_total=state.backlog_total,
            phase_switches=state.phase_switches,
            flicker_events=state.flicker_events,
            starvation_events=state.starvation_events,
            stability_penalty=state.stability_penalty,
            catastrophic_event=state.catastrophic_event,
            seed_value=state.seed,
            action_mask=action_mask,
            last_action_valid=action_valid,
            valid_actions=VALID_ACTIONS,
            # Grader info
            score_estimate=score,
            reasoning_logs=logs,
            grader_breakdown=grade_breakdown,
            grader_reasons=grade_reasons,
            last_action=action_name,
            last_action_reason=reason,
        )

    def _summary(self, state: IntersectionState) -> EpisodeSummary:
        moved_total = state.moved_ns + state.moved_ew
        avg_wait = state.total_wait_time / max(1, state.timestep)
        max_wait = state.max_wait_seen / max(1, state.timestep)
        backlog_end = state.queue_ns + state.queue_ew
        emergency_delay = state.emergency_wait_time / max(1, state.timestep)
        emergency_priority = state.emergency_priority_hits / max(1, state.emergency_appearances)
        stability = state.stability_penalty + (0.25 * state.no_progress_steps)
        return EpisodeSummary(
            task_id=state.task_id,
            steps=state.timestep,
            moved_total=moved_total,
            avg_wait=avg_wait,
            emergency_wait=state.emergency_wait_time / max(1, state.timestep),
            max_wait=max_wait,
            backlog_end=backlog_end,
            emergency_delay=emergency_delay,
            emergency_priority=emergency_priority,
            fairness_gap=state.fairness_gap,
            starvation=float(state.starvation_events),
            flicker=float(state.flicker_events),
            stability=stability,
            invalid_actions=state.invalid_actions,
            safety_violations=state.safety_violations,
            catastrophic_event=state.catastrophic_event,
            catastrophic_reason=state.catastrophic_reason,
        )

    def reset(
        self,
        task_id: str = "easy",
        session_id: Optional[str] = None,
        seed: Optional[int] = None,
        sensor_noise: bool = False,
        ood_start: bool = False,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TrafficObservation:
        sid = session_id or str(uuid.uuid4())
        task = get_task(task_id)
        episode_seed = int(seed if seed is not None else 0)

        istate = IntersectionState(
            session_id=sid,
            task_id=task.task_id,
            timestep=0,
            max_steps=task.max_steps,
            current_phase="ns",
            phase_remaining=task.min_green,
            seed=episode_seed,
            sensor_noise=sensor_noise,
            ood_start=ood_start,
        )
        if ood_start:
            offset = abs((episode_seed * 2654435761 + 1013904223) % 7)
            istate.queue_ns = task.arrivals_ns[0] * (2 + (offset % 2))
            istate.queue_ew = task.arrivals_ew[0] * (2 + ((offset // 2) % 2))
            istate.backlog_total = istate.queue_ns + istate.queue_ew

        _SESSIONS[sid] = (istate, task)
        self._state = State(episode_id=episode_id or sid, step_count=0)

        summary = self._summary(istate)
        score, logs = compute_score(summary, task)
        grade = compute_grade(summary, task)

        return self._build_observation(
            istate,
            action_valid=True,
            reward_value=0.0,
            is_done=False,
            score=score,
            logs=logs,
            grade_breakdown=grade.breakdown,
            grade_reasons=grade.reasons,
            action_name="reset",
        )

    def step(
        self,
        action: TrafficAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TrafficObservation:
        session_id = action.session_id
        action_text = action.action

        if session_id is None or session_id not in _SESSIONS:
            obs = self.reset(task_id="easy", session_id=session_id)
            obs.last_action_reason = "session_not_found_reset_applied"
            return obs

        istate, task = _SESSIONS[session_id]
        self._state = State(episode_id=session_id, step_count=istate.timestep)

        if istate.done:
            summary = self._summary(istate)
            score, logs = compute_score(summary, task)
            grade = compute_grade(summary, task)
            return self._build_observation(
                istate,
                action_valid=False,
                reward_value=0.0,
                is_done=True,
                score=score,
                logs=logs,
                grade_breakdown=grade.breakdown,
                grade_reasons=grade.reasons,
                action_name="hold",
                reason="episode_done",
            )

        sim_action, valid, reason = parse_action(action_text)
        previous_state = deepcopy(istate)
        moved_ns, moved_ew = simulate_step(istate, task, sim_action, valid)

        reward = compute_reward(previous_state, istate, moved_ns, moved_ew, valid)
        summary = self._summary(istate)
        score, logs = compute_score(summary, task)
        grade = compute_grade(summary, task)

        self._state.step_count = istate.timestep

        return self._build_observation(
            istate,
            action_valid=valid,
            reward_value=reward,
            is_done=istate.done,
            score=score,
            logs=logs,
            grade_breakdown=grade.breakdown,
            grade_reasons=grade.reasons,
            action_name=sim_action.raw,
            reason=reason,
        )

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self):
        from openenv.core.env_server.interfaces import EnvironmentMetadata

        return EnvironmentMetadata(
            name="traffic-control-openenv",
            description="Advanced deterministic traffic intersection control benchmark testing POMDP resilience, spatial reasoning, and dynamic constraints.",
            version="0.1.0",
        )

    # ------------------------------------------------------------------
    # Legacy helpers for direct access (used by tests, validation scripts)
    # ------------------------------------------------------------------

    def legacy_reset(
        self,
        task_id: str = "easy",
        session_id: Optional[str] = None,
        seed: Optional[int] = None,
        sensor_noise: bool = False,
        ood_start: bool = False,
    ) -> dict:
        """Returns data in the old dict format for backward-compatible callers."""
        obs = self.reset(
            task_id=task_id,
            session_id=session_id,
            seed=seed,
            sensor_noise=sensor_noise,
            ood_start=ood_start,
        )
        return _obs_to_legacy(obs)

    def legacy_step(
        self,
        session_id: str,
        action_text: str,
    ) -> dict:
        """Returns data in the old dict format for backward-compatible callers."""
        obs = self.step(TrafficAction(action=action_text, session_id=session_id))
        return _obs_to_legacy(obs)

    def legacy_state(self, session_id: str) -> dict:
        """Returns data in the old dict format for backward-compatible callers."""
        if session_id not in _SESSIONS:
            return self.legacy_reset(task_id="easy", session_id=session_id)
        istate, task = _SESSIONS[session_id]
        summary = self._summary(istate)
        score, logs = compute_score(summary, task)
        grade = compute_grade(summary, task)
        obs = self._build_observation(
            istate,
            action_valid=True,
            reward_value=0.0,
            is_done=istate.done,
            score=score,
            logs=logs,
            grade_breakdown=grade.breakdown,
            grade_reasons=grade.reasons,
            action_name="state",
        )
        return _obs_to_legacy(obs)


def _obs_to_legacy(obs: TrafficObservation) -> dict:
    """Convert a TrafficObservation to the legacy dict format with observation/reward/done/info keys."""
    d = obs.model_dump()
    # Remove base fields that go to top level
    done = d.pop("done", False)
    reward = d.pop("reward", 0.0)
    d.pop("metadata", None)
    # Extract info fields from observation
    info = {
        "action": d.pop("last_action", "reset"),
        "action_valid": d.pop("last_action_valid", True),
        "reason": d.pop("last_action_reason", None),
        "task_id": d.get("task_id", ""),
        "score_estimate": d.pop("score_estimate", 0.0),
        "reasoning_logs": d.pop("reasoning_logs", []),
        "grader_breakdown": d.pop("grader_breakdown", {}),
        "grader_reasons": d.pop("grader_reasons", []),
    }
    return {
        "observation": d,
        "reward": reward,
        "done": done,
        "info": info,
    }
