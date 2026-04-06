from __future__ import annotations

from typing import Dict, List, Tuple

from env.models import EpisodeSummary, GradeResult, TaskConfig


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_badness(value: float, scale: float) -> float:
    return _clip01(value / max(1e-6, scale))


def _build_reasons(score_terms: Dict[str, float], summary: EpisodeSummary) -> List[str]:
    reasons: List[str] = []

    if score_terms["throughput"] >= 0.14:
        reasons.append(f"Bonus: +{score_terms['throughput']:.3f} | Throughput cleared vehicles effectively")
    if score_terms["emergency_priority"] >= 0.08:
        reasons.append(
            f"Bonus: +{score_terms['emergency_priority']:.3f} | Emergency prioritization succeeded"
        )
    if score_terms["fairness"] >= 0.07:
        reasons.append(f"Bonus: +{score_terms['fairness']:.3f} | Lane fairness maintained")

    avg_wait_penalty = 0.14 - score_terms["avg_wait"]
    if avg_wait_penalty > 0.06:
        reasons.append(f"Penalty: -{avg_wait_penalty:.3f} | Average wait remained high")

    emergency_penalty = 0.18 - score_terms["emergency_delay"]
    if emergency_penalty > 0.08:
        reasons.append(f"Penalty: -{emergency_penalty:.3f} | Emergency delay exceeded target")

    starvation_penalty = 0.08 - score_terms["starvation"]
    if starvation_penalty > 0.04:
        reasons.append(f"Penalty: -{starvation_penalty:.3f} | Lane starvation detected")

    flicker_penalty = 0.06 - score_terms["flicker"]
    if flicker_penalty > 0.03:
        reasons.append(f"Penalty: -{flicker_penalty:.3f} | Signal flickering/rapid switching detected")

    if summary.invalid_actions > 0:
        reasons.append(f"Penalty: variable | Invalid actions observed: {summary.invalid_actions}")

    if summary.catastrophic_event:
        reasons.append(f"Penalty: severe | Catastrophic event: {summary.catastrophic_reason or 'unsafe_event'}")

    if not reasons:
        reasons.append("Stable performance with mixed traffic outcomes")
    return reasons


def compute_score(summary: EpisodeSummary, task: TaskConfig) -> Tuple[float, List[str]]:
    logs: List[str] = []

    throughput_ratio = summary.moved_total / max(1.0, task.target_passed)
    throughput_score = _clip01(throughput_ratio**1.5)
    logs.append(f"Throughput Score: {throughput_score:.2f} ({summary.moved_total}/{task.target_passed})")

    emergency_ratio = summary.emergency_wait / max(1.0, task.target_emergency_wait)
    emergency_score = _clip01(1.0 - (emergency_ratio**2.5))
    if emergency_ratio > 1.0:
        logs.append(f"Penalty: Emergency wait exceeded by {(emergency_ratio - 1.0):.1%}")

    fairness_score = _clip01(1.0 - (summary.fairness_gap / max(1e-6, task.target_fairness_gap)))

    discipline_base = 1.0 - (summary.invalid_actions / max(1.0, task.max_steps * 0.05))
    safety_modifier = 0.0 if summary.safety_violations > 0 else 1.0
    discipline_score = _clip01(discipline_base * safety_modifier)

    final_score = (0.45 * throughput_score) + (0.30 * emergency_score) + (0.15 * fairness_score) + (0.10 * discipline_score)
    if summary.safety_violations > 0:
        logs.append("CRITICAL: Safety violation detected. Score crushed.")

    return _clip01(final_score), logs


def compute_grade(summary: EpisodeSummary, task: TaskConfig) -> GradeResult:
    score_from_diagnostic, diagnostic_logs = compute_score(summary, task)

    throughput = _clip01(summary.moved_total / max(1.0, task.target_passed))
    avg_wait_bad = _normalize_badness(summary.avg_wait, task.target_wait)
    max_wait_bad = _normalize_badness(summary.max_wait, task.target_max_wait)
    backlog_bad = _normalize_badness(summary.backlog_end, task.target_backlog_end)
    emergency_bad = _normalize_badness(summary.emergency_delay, task.target_emergency_wait)
    emergency_priority = _clip01(summary.emergency_priority)
    fairness = _clip01(1.0 - (summary.fairness_gap / max(1e-6, task.target_fairness_gap)))
    starvation_bad = _normalize_badness(summary.starvation, task.target_starvation_events)
    flicker_bad = _normalize_badness(summary.flicker, task.target_flicker_events)
    stability_bad = _normalize_badness(summary.stability, task.target_stability_penalty)

    invalid_budget = max(1.0, task.max_steps * 0.15)
    invalid_penalty = _clip01(summary.invalid_actions / invalid_budget)

    score_terms = {
        "throughput": 0.18 * throughput,
        "avg_wait": 0.14 * (1.0 - avg_wait_bad),
        "max_wait": 0.08 * (1.0 - max_wait_bad),
        "backlog_end": 0.10 * (1.0 - backlog_bad),
        "emergency_delay": 0.18 * (1.0 - emergency_bad),
        "emergency_priority": 0.12 * emergency_priority,
        "fairness": 0.10 * fairness,
        "starvation": 0.08 * (1.0 - starvation_bad),
        "flicker": 0.06 * (1.0 - flicker_bad),
        "stability": 0.04 * (1.0 - stability_bad),
    }

    score = sum(score_terms.values()) - 0.08 * invalid_penalty
    score = _clip01(score)
    score = (0.65 * score) + (0.35 * score_from_diagnostic)
    score = _clip01(score)

    if summary.catastrophic_event:
        score = min(score, 0.03)

    breakdown = {
        "throughput": throughput,
        "avg_wait": avg_wait_bad,
        "max_wait": max_wait_bad,
        "backlog_end": backlog_bad,
        "emergency_delay": emergency_bad,
        "emergency_priority": emergency_priority,
        "fairness": fairness,
        "starvation": starvation_bad,
        "flicker": flicker_bad,
        "stability": stability_bad,
    }
    reasons = _build_reasons(score_terms, summary)
    reasons.extend(log for log in diagnostic_logs if log not in reasons)
    return GradeResult(score=score, breakdown=breakdown, reasons=reasons)


def compute_scalar_score(summary: EpisodeSummary, task: TaskConfig) -> float:
    return compute_grade(summary, task).score
