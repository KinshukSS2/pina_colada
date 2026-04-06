from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Dict, Tuple

from env.grader import compute_grade, compute_score
from env.models import EpisodeSummary, IntersectionState, Observation, StepInfo, StepResult, TaskConfig
from env.reward import compute_reward
from env.simulator import VALID_ACTIONS, _generate_ascii, parse_action, simulate_step
from env.tasks import get_task


class TrafficControlEnvironment:
    def __init__(self) -> None:
        self.sessions: Dict[str, Tuple[IntersectionState, TaskConfig]] = {}

    @staticmethod
    def _noise_delta(seed: int, timestep: int, key: int) -> int:
        value = (seed * 1103515245 + (timestep + 1) * 12345 + key * 1013) & 0x7FFFFFFF
        return int(value % 3) - 1

    def _build_observation(self, state: IntersectionState, action_valid: bool) -> Observation:
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

        return Observation(
            session_id=state.session_id,
            task_id=state.task_id,
            timestep=state.timestep,
            max_steps=state.max_steps,
            current_phase=state.current_phase,
            phase_remaining=state.phase_remaining,
            yellow_active=(state.yellow_remaining > 0),
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
            seed=state.seed,
            action_mask=action_mask,
            last_action_valid=action_valid,
            valid_actions=VALID_ACTIONS,
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
        session_id: str | None = None,
        seed: int | None = None,
        sensor_noise: bool = False,
        ood_start: bool = False,
    ) -> StepResult:
        sid = session_id or str(uuid.uuid4())
        task = get_task(task_id)
        episode_seed = int(seed if seed is not None else 0)

        state = IntersectionState(
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
            state.queue_ns = task.arrivals_ns[0] * (2 + (offset % 2))
            state.queue_ew = task.arrivals_ew[0] * (2 + ((offset // 2) % 2))
            state.backlog_total = state.queue_ns + state.queue_ew

        self.sessions[sid] = (state, task)

        summary = self._summary(state)
        score, logs = compute_score(summary, task)
        grade = compute_grade(summary, task)
        observation = self._build_observation(state, action_valid=True)
        info = StepInfo(
            action="reset",
            action_valid=True,
            reason=None,
            task_id=task.task_id,
            score_estimate=score,
            reasoning_logs=logs,
            grader_breakdown=grade.breakdown,
            grader_reasons=grade.reasons,
        )
        return StepResult(observation=observation, reward=0.0, done=False, info=info)

    def step(
        self,
        session_id: str,
        action_text: str,
        fallback_status: str = "unknown",
        agent_reason: str | None = None,
    ) -> StepResult:
        if session_id not in self.sessions:
            result = self.reset(task_id="easy", session_id=session_id)
            result.info.reason = "session_not_found_reset_applied"
            return result

        state, task = self.sessions[session_id]

        if state.done:
            summary = self._summary(state)
            score, logs = compute_score(summary, task)
            grade = compute_grade(summary, task)
            observation = self._build_observation(state, action_valid=False)
            info = StepInfo(
                action="hold",
                action_valid=False,
                fallback_status=fallback_status,
                agent_reason=agent_reason,
                reason="episode_done",
                task_id=task.task_id,
                score_estimate=score,
                reasoning_logs=logs,
                grader_breakdown=grade.breakdown,
                grader_reasons=grade.reasons,
            )
            return StepResult(observation=observation, reward=0.0, done=True, info=info)

        action, valid, reason = parse_action(action_text)
        previous_state = deepcopy(state)
        moved_ns, moved_ew = simulate_step(state, task, action, valid)

        reward = compute_reward(previous_state, state, moved_ns, moved_ew, valid)
        summary = self._summary(state)
        score, logs = compute_score(summary, task)
        grade = compute_grade(summary, task)

        observation = self._build_observation(state, action_valid=valid)
        info = StepInfo(
            action=action.raw,
            action_valid=valid,
            fallback_status=fallback_status,
            agent_reason=agent_reason,
            reason=reason,
            task_id=task.task_id,
            score_estimate=score,
            reasoning_logs=logs,
            grader_breakdown=grade.breakdown,
            grader_reasons=grade.reasons,
        )
        return StepResult(observation=observation, reward=reward, done=state.done, info=info)

    def state(self, session_id: str) -> StepResult:
        if session_id not in self.sessions:
            return self.reset(task_id="easy", session_id=session_id)

        state, task = self.sessions[session_id]
        summary = self._summary(state)
        score, logs = compute_score(summary, task)
        grade = compute_grade(summary, task)
        observation = self._build_observation(state, action_valid=True)
        info = StepInfo(
            action="state",
            action_valid=True,
            reason=None,
            task_id=task.task_id,
            score_estimate=score,
            reasoning_logs=logs,
            grader_breakdown=grade.breakdown,
            grader_reasons=grade.reasons,
        )
        return StepResult(observation=observation, reward=0.0, done=state.done, info=info)
