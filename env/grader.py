from __future__ import annotations

from env.models import EpisodeSummary, TaskConfig


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_score(summary: EpisodeSummary, task: TaskConfig) -> float:
    throughput_score = _clip01(summary.moved_total / max(1.0, task.target_passed))
    delay_score = _clip01(1.0 - (summary.avg_wait / max(1.0, task.target_wait)))
    emergency_score = _clip01(1.0 - (summary.emergency_wait / max(1.0, task.target_emergency_wait)))
    fairness_score = _clip01(1.0 - (summary.fairness_gap / max(1e-6, task.target_fairness_gap)))

    invalid_budget = max(1.0, task.max_steps * 0.15)
    invalid_score = _clip01(1.0 - (summary.invalid_actions / invalid_budget))

    score = (
        0.35 * throughput_score
        + 0.20 * delay_score
        + 0.20 * emergency_score
        + 0.15 * fairness_score
        + 0.10 * invalid_score
    )
    return _clip01(score)
