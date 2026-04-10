"""Medium-task grader — multi-queue with SLA deadlines and escalation.

Architecture: gate × _soft_cap(weighted_sum, DIFFICULTY_CEIL)
=============================================================

  final_score = gate × _soft_cap(
      W_OUTCOME * outcome + W_PROCESS * process
      + W_IMPROVE * improvement + W_TAIL * tail_risk, DIFFICULTY_CEIL)

  outcome_score = weighted_sum(
      0.30 * accuracy_score,
      0.25 * sla_compliance_score,
      0.15 * queue_management_score,
      0.15 * fairness_score,
      0.10 * throughput_score,
      0.05 * diversity_score,
  )

Target ranges
-------------
  Rule-based baseline: ≈ 0.28–0.42
  Good LLM policy:     ≈ 0.42–0.58
  Random policy:        ≈ 0.08–0.25
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

from graders.base_grader import BaseGrader

W_OUTCOME  = 0.55
W_PROCESS  = 0.25
W_IMPROVE  = 0.10
W_TAIL     = 0.10

W_ACCURACY   = 0.30
W_SLA        = 0.25
W_QUEUE_MGMT = 0.15
W_FAIRNESS   = 0.15
W_THROUGHPUT = 0.10
W_DIVERSITY  = 0.05

DIFFICULTY_CEIL = 0.72


def _soft_cap(score: float, ceiling: float) -> float:
    if ceiling <= 0.0:
        return 0.0
    return ceiling * (1.0 - math.exp(-3.0 * score / ceiling))


class MediumGrader(BaseGrader):
    """Deterministic grader for Task 2 (Medium)."""

    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        if not trajectory:
            return 0.0

        n_steps = len(trajectory)

        # ------------------------------------------------------------------
        # 1. Gate
        # ------------------------------------------------------------------
        exploit_penalty = self._anti_exploit_penalty(trajectory)
        skip_r = self._skip_rate(trajectory)

        gate = exploit_penalty
        if skip_r > 0.50:
            gate *= 0.55
        elif skip_r > 0.30:
            gate *= 0.75

        if gate < 0.01:
            return 0.0

        # ------------------------------------------------------------------
        # 2. Process score
        # ------------------------------------------------------------------
        process_scores = self._compute_process_scores(trajectory)
        process_score = self._robust_mean(process_scores, default=0.5)

        # ------------------------------------------------------------------
        # 3. Outcome subscores
        # ------------------------------------------------------------------
        last_snap = trajectory[-1].get("state_snapshot", {})

        # Accuracy
        correct = float(last_snap.get("correct_assignments", 0))
        total_assigned = correct + float(last_snap.get("incorrect_assignments", 0))
        accuracy = correct / max(total_assigned, 1.0)
        accuracy_score = self._normalise(accuracy, 0.0, 1.0)

        # SLA compliance
        sla_breaches = float(last_snap.get("sla_breaches", 0))
        total_tickets = float(last_snap.get("resolved_count", 0)) + float(last_snap.get("pending_count", 0))
        sla_rate = sla_breaches / max(total_tickets, 1.0)
        sla_score = self._invert(self._normalise(sla_rate, 0.0, 0.5))

        # Queue management (trend-based, replaces routing_efficiency)
        queue_mgmt_score = self._queue_trend_score(trajectory)

        # Fairness across departments
        fairness_score = self._department_fairness(trajectory)

        # Throughput: resolved per step
        resolved = float(last_snap.get("resolved_count", 0))
        throughput = resolved / max(n_steps, 1)
        throughput_score = self._normalise(throughput, 0.0, 1.0)

        # Diversity
        diversity_score = self._action_entropy(trajectory)

        outcome_score = max(0.0, min(1.0,
            W_ACCURACY   * accuracy_score
            + W_SLA        * sla_score
            + W_QUEUE_MGMT * queue_mgmt_score
            + W_FAIRNESS   * fairness_score
            + W_THROUGHPUT * throughput_score
            + W_DIVERSITY  * diversity_score
        ))

        # ------------------------------------------------------------------
        # 4. Trajectory-aware terms
        # ------------------------------------------------------------------
        improve = self._improvement_score(trajectory)
        tail = self._tail_risk_score(trajectory)

        # ------------------------------------------------------------------
        # 5. Combined with soft cap
        # ------------------------------------------------------------------
        raw = (W_OUTCOME * outcome_score
               + W_PROCESS * process_score
               + W_IMPROVE * improve
               + W_TAIL * tail)
        capped = _soft_cap(raw, DIFFICULTY_CEIL)
        final = gate * capped
        return float(max(0.0, min(1.0, final)))
