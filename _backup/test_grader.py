from __future__ import annotations

import unittest

from env.environment import TrafficControlEnvironment
from env.grader import compute_grade
from env.models import EpisodeSummary, TrafficAction
from env.tasks import get_task


class GraderTests(unittest.TestCase):
    def test_perfect_trajectory_scores_high(self) -> None:
        task = get_task("medium")
        summary = EpisodeSummary(
            task_id=task.task_id,
            steps=task.max_steps,
            moved_total=int(task.target_passed * 1.08),
            avg_wait=task.target_wait * 0.28,
            max_wait=task.target_max_wait * 0.30,
            backlog_end=task.target_backlog_end * 0.22,
            emergency_delay=task.target_emergency_wait * 0.20,
            emergency_priority=0.98,
            fairness_gap=task.target_fairness_gap * 0.22,
            starvation=0.0,
            flicker=0.0,
            stability=task.target_stability_penalty * 0.10,
            invalid_actions=0,
            catastrophic_event=False,
            catastrophic_reason=None,
        )

        grade = compute_grade(summary, task)

        self.assertGreaterEqual(grade.score, 0.88)
        self.assertLessEqual(grade.score, 1.0)

    def test_selfish_trajectory_scores_moderate(self) -> None:
        task = get_task("medium")
        summary = EpisodeSummary(
            task_id=task.task_id,
            steps=task.max_steps,
            moved_total=int(task.target_passed * 1.05),
            avg_wait=task.target_wait * 0.85,
            max_wait=task.target_max_wait * 1.05,
            backlog_end=task.target_backlog_end * 0.95,
            emergency_delay=task.target_emergency_wait * 1.15,
            emergency_priority=0.55,
            fairness_gap=task.target_fairness_gap * 1.55,
            starvation=task.target_starvation_events * 1.20,
            flicker=task.target_flicker_events * 0.65,
            stability=task.target_stability_penalty * 0.95,
            invalid_actions=2,
            catastrophic_event=False,
            catastrophic_reason=None,
        )

        grade = compute_grade(summary, task)

        self.assertGreaterEqual(grade.score, 0.20)
        self.assertLessEqual(grade.score, 0.72)

    def test_catastrophic_trajectory_scores_near_zero(self) -> None:
        task = get_task("hard")
        summary = EpisodeSummary(
            task_id=task.task_id,
            steps=task.max_steps,
            moved_total=int(task.target_passed * 0.75),
            avg_wait=task.target_wait * 1.40,
            max_wait=task.target_max_wait * 1.65,
            backlog_end=task.target_backlog_end * 1.50,
            emergency_delay=task.target_emergency_wait * 2.20,
            emergency_priority=0.15,
            fairness_gap=task.target_fairness_gap * 2.10,
            starvation=task.target_starvation_events * 2.00,
            flicker=task.target_flicker_events * 2.40,
            stability=task.target_stability_penalty * 1.90,
            invalid_actions=8,
            catastrophic_event=True,
            catastrophic_reason="unsafe_bypass_transition",
        )

        grade = compute_grade(summary, task)

        self.assertGreaterEqual(grade.score, 0.0)
        self.assertLessEqual(grade.score, 0.05)

    def test_same_trajectory_is_deterministic(self) -> None:
        env_a = TrafficControlEnvironment()
        env_b = TrafficControlEnvironment()
        actions = [
            "hold",
            "switch",
            "set_ns_green:4",
            "hold",
            "prioritize_emergency",
            "set_ew_green:3",
            "hold",
            "switch",
            "hold",
            "hold",
        ]

        first = env_a.reset(task_id="medium", session_id="det-a", seed=123, sensor_noise=True, ood_start=True)
        second = env_b.reset(task_id="medium", session_id="det-b", seed=123, sensor_noise=True, ood_start=True)

        for action in actions:
            first = env_a.step(TrafficAction(action=action, session_id="det-a"))
            second = env_b.step(TrafficAction(action=action, session_id="det-b"))

        self.assertEqual(first.score_estimate, second.score_estimate)
        self.assertEqual(first.grader_breakdown, second.grader_breakdown)
        self.assertEqual(first.grader_reasons, second.grader_reasons)


if __name__ == "__main__":
    unittest.main()
