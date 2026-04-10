"""Hard-task grader — VIP handling, partial information, tight resources.

Architecture: gate × _soft_cap(weighted_sum, DIFFICULTY_CEIL)
=============================================================

  final_score = gate × _soft_cap(
      W_OUTCOME * outcome + W_PROCESS * process
      + W_IMPROVE * improvement + W_TAIL * tail_risk, DIFFICULTY_CEIL)

  outcome_score = weighted_sum(
      0.25 * accuracy_score,
      0.25 * sla_compliance_score,
      0.15 * vip_handling_score,
      0.12 * escalation_quality_score,
      0.10 * resource_management_score,
      0.08 * fairness_score,
      0.05 * diversity_score,
  )

Target ranges
-------------
  Rule-based baseline: ≈ 0.15–0.30
  Good LLM policy:     ≈ 0.28–0.45
  Random policy:        ≈ 0.03–0.15
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

from graders.base_grader import BaseGrader

W_OUTCOME  = 0.45
W_PROCESS  = 0.25
W_IMPROVE  = 0.15
W_TAIL     = 0.15

W_ACCURACY     = 0.25
W_SLA          = 0.25
W_VIP          = 0.15
W_ESCALATION   = 0.12
W_RESOURCE     = 0.10
W_FAIRNESS     = 0.08
W_DIVERSITY    = 0.05

DIFFICULTY_CEIL = 0.55


def _soft_cap(score: float, ceiling: float) -> float:
    if ceiling <= 0.0:
        return 0.0
    return ceiling * (1.0 - math.exp(-3.0 * score / ceiling))


class HardGrader(BaseGrader):
    """Deterministic grader for Task 3 (Hard)."""

    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        if not trajectory:
            return 0.0

        # ------------------------------------------------------------------
        # 1. Gate (strict for hard)
        # ------------------------------------------------------------------
        exploit_penalty = self._anti_exploit_penalty(trajectory)
        skip_r = self._skip_rate(trajectory)
        defer_r = self._defer_rate(trajectory)

        gate = exploit_penalty
        if skip_r > 0.35:
            gate *= 0.45
        elif skip_r > 0.20:
            gate *= 0.70

        if defer_r > 0.40:
            gate *= 0.55
        elif defer_r > 0.25:
            gate *= 0.75

        if gate < 0.01:
            return 0.0

        # ------------------------------------------------------------------
        # 2. Process score
        # ------------------------------------------------------------------
        process_scores = self._compute_process_scores(trajectory)
        process_score = self._robust_mean(process_scores, default=0.4)

        # ------------------------------------------------------------------
        # 3. Outcome subscores
        # ------------------------------------------------------------------
        last = trajectory[-1].get("state_snapshot", {})

        # Accuracy
        correct = float(last.get("correct_assignments", 0))
        total_assigned = correct + float(last.get("incorrect_assignments", 0))
        accuracy = correct / max(total_assigned, 1.0)
        accuracy_score = self._normalise(accuracy, 0.0, 1.0)

        # SLA Compliance (stricter range for hard)
        sla_breaches = float(last.get("sla_breaches", 0))
        total_tickets = float(last.get("resolved_count", 0)) + float(last.get("pending_count", 0))
        sla_rate = sla_breaches / max(total_tickets, 1.0)
        sla_score = self._invert(self._normalise(sla_rate, 0.0, 0.4))

        # VIP handling (aggregate breach-based)
        vip_score = self._vip_handling_score(trajectory)

        # Escalation quality (rate-based)
        escalation_score = self._escalation_quality_score(trajectory)

        # Resource management
        resource_score = self._resource_management_score(trajectory)

        # Department fairness
        fairness_score = self._department_fairness(trajectory)

        # Diversity
        diversity_score = self._action_entropy(trajectory)

        outcome_score = max(0.0, min(1.0,
            W_ACCURACY   * accuracy_score
            + W_SLA        * sla_score
            + W_VIP        * vip_score
            + W_ESCALATION * escalation_score
            + W_RESOURCE   * resource_score
            + W_FAIRNESS   * fairness_score
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

    # ------------------------------------------------------------------ #
    #                   Hard-specific metrics                             #
    # ------------------------------------------------------------------ #

    def _vip_handling_score(self, trajectory: List[Dict[str, Any]]) -> float:
        """Aggregate VIP performance based on breach count.

        Uses vip_breaches from state snapshots — more reliable than
        per-ticket action matching since current_ticket in the snapshot
        refers to the *next* ticket rather than the one just processed.
        """
        if not trajectory:
            return 0.3
        last = trajectory[-1].get("state_snapshot", {})
        vip_breaches = float(last.get("vip_breaches", 0))
        total_tickets = float(last.get("resolved_count", 0)) + float(last.get("pending_count", 0))
        if total_tickets == 0:
            return 0.3
        # 0 breaches = 1.0, breaches >= 20% of tickets = 0.0
        vip_breach_rate = vip_breaches / max(total_tickets, 1.0)
        return self._invert(self._normalise(vip_breach_rate, 0.0, 0.20))

    def _escalation_quality_score(self, trajectory: List[Dict[str, Any]]) -> float:
        """Rate-based escalation evaluation.

        Moderate escalation (10–25%) is optimal for hard tasks.
        Too few → might miss complex cases.  Too many → dumping work.
        """
        if not trajectory:
            return 0.3
        n_escalate = sum(1 for s in trajectory if s.get("action") == "escalate")
        n_total = len(trajectory)
        if n_total == 0:
            return 0.3
        rate = n_escalate / n_total
        if 0.10 <= rate <= 0.25:
            return 0.85
        elif rate < 0.05:
            return 0.40
        elif rate < 0.10:
            return 0.65
        elif rate <= 0.40:
            return 0.50
        else:
            return 0.20

    def _resource_management_score(self, trajectory: List[Dict[str, Any]]) -> float:
        """Department load balance at end of episode."""
        if not trajectory:
            return 0.3
        last = trajectory[-1].get("state_snapshot", {})
        dept_loads = last.get("department_load", {})
        if not dept_loads:
            return 0.3
        loads = list(dept_loads.values())
        if not loads:
            return 0.3
        max_load = max(loads)
        avg_load = sum(loads) / len(loads)
        if avg_load <= 0:
            return 0.3
        spread = max_load / avg_load
        return self._invert(self._normalise(spread - 1.0, 0.0, 3.0))
