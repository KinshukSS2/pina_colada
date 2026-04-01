from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Dict, Tuple

from env.grader import compute_score
from env.models import EpisodeSummary, IntersectionState, Observation, StepInfo, StepResult, TaskConfig
from env.reward import compute_reward
from env.simulator import VALID_ACTIONS, parse_action, simulate_step
from env.tasks import get_task


class TrafficControlEnvironment:
    def __init__(self) -> None:
        self.sessions: Dict[str, Tuple[IntersectionState, TaskConfig]] = {}

    def _build_observation(self, state: IntersectionState, action_valid: bool) -> Observation:
        return Observation(
            session_id=state.session_id,
            task_id=state.task_id,
            timestep=state.timestep,
            max_steps=state.max_steps,
            current_phase=state.current_phase,
            phase_remaining=state.phase_remaining,
            queue_ns=state.queue_ns,
            queue_ew=state.queue_ew,
            emergency_ns=state.emergency_ns,
            emergency_ew=state.emergency_ew,
            moved_ns=state.moved_ns,
            moved_ew=state.moved_ew,
            total_wait_time=state.total_wait_time,
            emergency_wait_time=state.emergency_wait_time,
            invalid_actions=state.invalid_actions,
            fairness_gap=state.fairness_gap,
            last_action_valid=action_valid,
            valid_actions=VALID_ACTIONS,
        )

    def _summary(self, state: IntersectionState) -> EpisodeSummary:
        moved_total = state.moved_ns + state.moved_ew
        avg_wait = state.total_wait_time / max(1, state.timestep)
        emergency_wait = state.emergency_wait_time / max(1, state.timestep)
        return EpisodeSummary(
            task_id=state.task_id,
            steps=state.timestep,
            moved_total=moved_total,
            avg_wait=avg_wait,
            emergency_wait=emergency_wait,
            fairness_gap=state.fairness_gap,
            invalid_actions=state.invalid_actions,
        )

    def reset(self, task_id: str = "easy", session_id: str | None = None) -> StepResult:
        sid = session_id or str(uuid.uuid4())
        task = get_task(task_id)

        state = IntersectionState(
            session_id=sid,
            task_id=task.task_id,
            timestep=0,
            max_steps=task.max_steps,
            current_phase="ns",
            phase_remaining=task.min_green,
        )
        self.sessions[sid] = (state, task)

        summary = self._summary(state)
        score_estimate = compute_score(summary, task)
        observation = self._build_observation(state, action_valid=True)
        info = StepInfo(
            action="reset",
            action_valid=True,
            reason=None,
            task_id=task.task_id,
            score_estimate=score_estimate,
        )
        return StepResult(observation=observation, reward=0.0, done=False, info=info)

    def step(self, session_id: str, action_text: str) -> StepResult:
        if session_id not in self.sessions:
            result = self.reset(task_id="easy", session_id=session_id)
            result.info.reason = "session_not_found_reset_applied"
            return result

        state, task = self.sessions[session_id]

        if state.done:
            summary = self._summary(state)
            score_estimate = compute_score(summary, task)
            observation = self._build_observation(state, action_valid=False)
            info = StepInfo(
                action="hold",
                action_valid=False,
                reason="episode_done",
                task_id=task.task_id,
                score_estimate=score_estimate,
            )
            return StepResult(observation=observation, reward=0.0, done=True, info=info)

        action, valid, reason = parse_action(action_text)
        previous_state = deepcopy(state)
        moved_ns, moved_ew = simulate_step(state, task, action, valid)

        reward = compute_reward(previous_state, state, moved_ns, moved_ew, valid)
        summary = self._summary(state)
        score_estimate = compute_score(summary, task)

        observation = self._build_observation(state, action_valid=valid)
        info = StepInfo(
            action=action.raw,
            action_valid=valid,
            reason=reason,
            task_id=task.task_id,
            score_estimate=score_estimate,
        )
        return StepResult(observation=observation, reward=reward, done=state.done, info=info)

    def state(self, session_id: str) -> StepResult:
        if session_id not in self.sessions:
            return self.reset(task_id="easy", session_id=session_id)

        state, task = self.sessions[session_id]
        summary = self._summary(state)
        score_estimate = compute_score(summary, task)
        observation = self._build_observation(state, action_valid=True)
        info = StepInfo(
            action="state",
            action_valid=True,
            reason=None,
            task_id=task.task_id,
            score_estimate=score_estimate,
        )
        return StepResult(observation=observation, reward=0.0, done=state.done, info=info)
