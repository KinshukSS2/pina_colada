from __future__ import annotations

import unittest

from env.environment import TicketTriageEnvironment, _SESSIONS
from env.schemas import TriageAction
from graders.easy_grader import EasyGrader
from graders.medium_grader import MediumGrader
from graders.hard_grader import HardGrader
from baseline.rule_based_agent import baseline_action


def _run_episode(task_id: str, seed: int = 42, agent=None):
    """Run a full episode and return (score, state)."""
    if agent is None:
        agent = baseline_action
    env = TicketTriageEnvironment()
    sid = f"test-run-{task_id}-{seed}"
    obs = env.reset(task_id=task_id, session_id=sid, seed=seed)
    while not obs.done:
        action = agent(obs.model_dump())
        obs = env.step(TriageAction(action=action, session_id=sid))
    state = _SESSIONS[sid]["state"]
    grader = {"easy": EasyGrader(), "medium": MediumGrader(), "hard": HardGrader()}[task_id]
    score = grader.grade(state.trajectory)
    return score, state


class GraderTests(unittest.TestCase):
    def test_easy_grader_scores_positive(self) -> None:
        env = TicketTriageEnvironment()
        env.reset(task_id="easy", session_id="test-easy")
        actions = [
            "assign:medium:billing",
            "assign:high:technical",
            "assign:low:general",
            "assign:medium:billing",
            "assign:high:technical",
        ]
        for a in actions:
            env.step(TriageAction(action=a, session_id="test-easy"))

        grader = EasyGrader()
        state = _SESSIONS["test-easy"]["state"]
        score = grader.grade(state.trajectory)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(score, 0.01)

    def test_medium_grader_with_varied_actions(self) -> None:
        env = TicketTriageEnvironment()
        env.reset(task_id="medium", session_id="test-med")
        actions = [
            "assign:high:billing",
            "escalate",
            "assign:medium:technical",
            "assign:low:general",
            "assign:critical:account",
            "defer",
            "assign:high:technical",
        ]
        for a in actions:
            env.step(TriageAction(action=a, session_id="test-med"))

        grader = MediumGrader()
        state = _SESSIONS["test-med"]["state"]
        score = grader.grade(state.trajectory)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_hard_grader_penalizes_all_skip(self) -> None:
        env = TicketTriageEnvironment()
        env.reset(task_id="hard", session_id="test-hard")
        for _ in range(10):
            env.step(TriageAction(action="skip", session_id="test-hard"))

        grader = HardGrader()
        state = _SESSIONS["test-hard"]["state"]
        score = grader.grade(state.trajectory)
        self.assertLessEqual(score, 0.15)

    def test_deterministic_same_seed(self) -> None:
        env_a = TicketTriageEnvironment()
        env_b = TicketTriageEnvironment()
        actions = [
            "assign:medium:billing",
            "assign:high:technical",
            "escalate",
            "assign:low:general",
            "defer",
        ]

        obs_a = env_a.reset(task_id="easy", session_id="det-a", seed=42)
        obs_b = env_b.reset(task_id="easy", session_id="det-b", seed=42)

        for action in actions:
            obs_a = env_a.step(TriageAction(action=action, session_id="det-a"))
            obs_b = env_b.step(TriageAction(action=action, session_id="det-b"))

        self.assertEqual(obs_a.score_estimate, obs_b.score_estimate)
        self.assertEqual(obs_a.tickets_resolved, obs_b.tickets_resolved)
        self.assertEqual(obs_a.correct_assignments, obs_b.correct_assignments)

    # ------------------------------------------------------------------
    # Score ordering: Easy > Medium > Hard (baseline agent)
    # ------------------------------------------------------------------

    def test_baseline_score_ordering(self) -> None:
        """Easy must score strictly higher than Medium, Medium higher than Hard."""
        easy_score, _ = _run_episode("easy", seed=42)
        medium_score, _ = _run_episode("medium", seed=42)
        hard_score, _ = _run_episode("hard", seed=42)

        self.assertGreater(easy_score, medium_score,
                           f"Easy ({easy_score:.4f}) must be > Medium ({medium_score:.4f})")
        self.assertGreater(medium_score, hard_score,
                           f"Medium ({medium_score:.4f}) must be > Hard ({hard_score:.4f})")

    def test_ordering_across_seeds(self) -> None:
        """Ordering holds across multiple seeds."""
        for seed in [1, 7, 99, 256, 1337]:
            easy, _ = _run_episode("easy", seed=seed)
            medium, _ = _run_episode("medium", seed=seed)
            hard, _ = _run_episode("hard", seed=seed)
            self.assertGreater(easy, medium,
                               f"seed={seed}: Easy ({easy:.4f}) must be > Medium ({medium:.4f})")
            self.assertGreater(medium, hard,
                               f"seed={seed}: Medium ({medium:.4f}) must be > Hard ({hard:.4f})")

    # ------------------------------------------------------------------
    # Anti-exploit: degenerate strategies score very low
    # ------------------------------------------------------------------

    def test_all_skip_scores_low_all_tasks(self) -> None:
        """All-skip agent should score < 0.10 on every task."""
        def skip_agent(_obs):
            return "skip"
        for task_id in ["easy", "medium", "hard"]:
            score, _ = _run_episode(task_id, seed=42, agent=skip_agent)
            self.assertLess(score, 0.10,
                            f"All-skip on {task_id}: {score:.4f} should be < 0.10")

    def test_all_defer_scores_low(self) -> None:
        """All-defer agent should score < 0.15 on every task."""
        def defer_agent(_obs):
            return "defer"
        for task_id in ["easy", "medium", "hard"]:
            score, _ = _run_episode(task_id, seed=42, agent=defer_agent)
            self.assertLess(score, 0.15,
                            f"All-defer on {task_id}: {score:.4f} should be < 0.15")

    def test_single_dept_spam_penalised(self) -> None:
        """Agent that always assigns to one department should score lower than baseline."""
        def spam_agent(_obs):
            return "assign:medium:billing"
        for task_id in ["easy", "medium", "hard"]:
            baseline_score, _ = _run_episode(task_id, seed=42)
            spam_score, _ = _run_episode(task_id, seed=42, agent=spam_agent)
            self.assertLess(spam_score, baseline_score,
                            f"Spam on {task_id}: {spam_score:.4f} should be < baseline {baseline_score:.4f}")

    # ------------------------------------------------------------------
    # Scores bounded and deterministic
    # ------------------------------------------------------------------

    def test_scores_bounded_01(self) -> None:
        """All scores must be in [0, 1]."""
        for task_id in ["easy", "medium", "hard"]:
            score, _ = _run_episode(task_id, seed=42)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_deterministic_full_episode(self) -> None:
        """Same seed must yield identical scores."""
        for task_id in ["easy", "medium", "hard"]:
            s1, _ = _run_episode(task_id, seed=123)
            s2, _ = _run_episode(task_id, seed=123)
            self.assertEqual(s1, s2, f"{task_id}: {s1} != {s2}")


if __name__ == "__main__":
    unittest.main()
