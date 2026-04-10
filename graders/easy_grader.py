"""Easy-task grader — single-queue ticket triage.

Architecture: gate × (outcome + process + improvement + tail_risk)
==================================================================

  final_score = gate × (
      W_OUTCOME * outcome + W_PROCESS * process
      + W_IMPROVE * improvement + W_TAIL * tail_risk)

  outcome_score = weighted_sum(
      0.35 * accuracy_score,
      0.25 * completeness_score,
      0.20 * speed_score,
      0.10 * queue_health_score,
      0.10 * diversity_score,
  )

Target ranges
-------------
  Rule-based baseline: ≈ 0.40–0.55
  Good LLM policy:     ≈ 0.55–0.75
  Random policy:        ≈ 0.10–0.30
"""
from __future__ import annotations

from typing import Any, Dict, List

from graders.base_grader import BaseGrader

W_OUTCOME  = 0.60
W_PROCESS  = 0.25
W_IMPROVE  = 0.10
W_TAIL     = 0.05

# Outcome sub-weights
W_ACCURACY     = 0.35
W_COMPLETENESS = 0.25
W_SPEED        = 0.20
W_QUEUE        = 0.10
W_DIVERSITY    = 0.10


class EasyGrader(BaseGrader):
    """Deterministic grader for Task 1 (Easy)."""

    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        if not trajectory:
            return 0.0

        n_steps = len(trajectory)

        # ------------------------------------------------------------------
        # 1. Safety gate (anti-exploit only — easy is lenient)
        # ------------------------------------------------------------------
        gate = self._anti_exploit_penalty(trajectory)

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

        # Completeness
        resolved = float(last_snap.get("resolved_count", 0))
        total_seen = resolved + float(last_snap.get("pending_count", 0))
        completeness = resolved / max(total_seen, 1.0)
        completeness_score = self._normalise(completeness, 0.0, 1.0)

        # Speed
        wait_vals = self._extract_stat(trajectory, "total_wait_time")
        if len(wait_vals) >= 2:
            wait_deltas = [wait_vals[i] - wait_vals[i - 1] for i in range(1, len(wait_vals))]
            mean_delta = self._robust_mean(wait_deltas, 3.0)
        else:
            mean_delta = float(last_snap.get("total_wait_time", 0)) / max(n_steps, 1)
        speed_score = self._invert(self._normalise(mean_delta, 0.0, 10.0))

        # Queue health (trend over episode)
        queue_score = self._queue_trend_score(trajectory)

        # Diversity (full-action entropy)
        diversity_score = self._action_entropy(trajectory)

        outcome_score = max(0.0, min(1.0,
            W_ACCURACY     * accuracy_score
            + W_COMPLETENESS * completeness_score
            + W_SPEED        * speed_score
            + W_QUEUE        * queue_score
            + W_DIVERSITY    * diversity_score
        ))

        # ------------------------------------------------------------------
        # 4. Trajectory-aware terms
        # ------------------------------------------------------------------
        improve = self._improvement_score(trajectory)
        tail = self._tail_risk_score(trajectory)

        # ------------------------------------------------------------------
        # 5. Combined score (no soft cap for easy)
        # ------------------------------------------------------------------
        raw = (W_OUTCOME * outcome_score
               + W_PROCESS * process_score
               + W_IMPROVE * improve
               + W_TAIL * tail)
        final = gate * raw
        return float(max(0.0, min(1.0, final)))
