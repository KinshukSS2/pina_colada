"""Base grader interface — shared helpers for all difficulty levels.

Design principles
-----------------
1. Deterministic given the same trajectory.
2. Scores always clamped to [0, 1].
3. Winsorization and trimmed means protect against outliers.
4. Anti-exploit helpers detect degenerate policies.
5. Per-step local-action scoring via _step_process_score().
"""
from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class BaseGrader(ABC):
    """Consumes an episode trajectory and returns a score in [0, 1]."""

    def __init__(self, calibration: Optional[Dict[str, Tuple[float, float]]] = None):
        self.calibration = calibration

    @abstractmethod
    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        """Grade a trajectory. Returns float in [0.0, 1.0]."""

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _get_bounds(self, key: str, default_lo: float, default_hi: float) -> Tuple[float, float]:
        if self.calibration and key in self.calibration:
            lo, hi = self.calibration[key]
            if hi > lo:
                return lo, hi
        return default_lo, default_hi

    @staticmethod
    def _normalise(value: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 1.0
        return max(0.0, min(1.0, (value - lo) / (hi - lo)))

    @staticmethod
    def _invert(score: float) -> float:
        return 1.0 - max(0.0, min(1.0, score))

    @staticmethod
    def _safe_mean(vals: List[float], default: float = 0.0) -> float:
        return statistics.mean(vals) if vals else default

    # ------------------------------------------------------------------
    # Robust statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _winsorize(data: List[float], lo_pct: float = 5.0, hi_pct: float = 95.0) -> List[float]:
        if len(data) < 4:
            return data
        sorted_d = sorted(data)
        n = len(sorted_d)
        lo_idx = max(0, int(n * lo_pct / 100.0))
        hi_idx = min(n - 1, int(n * hi_pct / 100.0))
        lo_val = sorted_d[lo_idx]
        hi_val = sorted_d[hi_idx]
        return [max(lo_val, min(hi_val, v)) for v in data]

    @staticmethod
    def _trimmed_mean(data: List[float], trim_pct: float = 0.10) -> float:
        if not data:
            return 0.0
        n = len(data)
        k = max(0, int(n * trim_pct))
        sdata = sorted(data)
        trimmed = sdata[k: n - k] if k > 0 else sdata
        return float(statistics.mean(trimmed)) if trimmed else float(statistics.mean(sdata))

    def _robust_mean(self, vals: List[float], default: float = 0.0) -> float:
        if not vals:
            return default
        ws = self._winsorize(vals)
        return self._trimmed_mean(ws)

    # ------------------------------------------------------------------
    # Metric extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_stat(trajectory: List[Dict], key: str) -> List[float]:
        vals = []
        for step in trajectory:
            snap = step.get("state_snapshot", {})
            if key in snap:
                vals.append(float(snap[key]))
        return vals

    # ------------------------------------------------------------------
    # Anti-exploit detection
    # ------------------------------------------------------------------

    @staticmethod
    def _skip_rate(trajectory: List[Dict]) -> float:
        """Fraction of steps where agent chose 'skip'."""
        if not trajectory:
            return 0.0
        skips = sum(1 for s in trajectory if s.get("action") == "skip")
        return skips / len(trajectory)

    @staticmethod
    def _defer_rate(trajectory: List[Dict]) -> float:
        """Fraction of steps where agent chose 'defer'."""
        if not trajectory:
            return 0.0
        defers = sum(1 for s in trajectory if s.get("action") == "defer")
        return defers / len(trajectory)

    @staticmethod
    def _action_entropy(trajectory: List[Dict]) -> float:
        """Diversity of full action strings (including parameters).

        Uses ``action_taken`` when present so that
        ``assign:high:billing`` and ``assign:low:technical`` count as
        distinct actions.  Avoids penalising agents that correctly
        route tickets to diverse departments/priorities.
        """
        if not trajectory:
            return 0.0
        action_counts: Dict[str, int] = {}
        for step in trajectory:
            a = step.get("action_taken", step.get("action", "skip"))
            action_counts[a] = action_counts.get(a, 0) + 1
        total = sum(action_counts.values())
        if total == 0:
            return 0.0
        n_distinct = len(action_counts)
        if n_distinct <= 1:
            return 0.0
        entropy = 0.0
        for count in action_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(n_distinct)
        return min(1.0, entropy / max(max_entropy, 0.01))

    @staticmethod
    def _dept_concentration(trajectory: List[Dict]) -> float:
        """Fraction of assignments routed to the single busiest dept.

        1.0 = all to one department (bad), low = well-distributed.
        Returns 0.0 when there are fewer than 2 assignments.
        """
        dept_counts: Dict[str, int] = {}
        total_assigns = 0
        for step in trajectory:
            if step.get("action") == "assign" and step.get("success", False):
                dept = step.get("params", {}).get("department", "")
                dept_counts[dept] = dept_counts.get(dept, 0) + 1
                total_assigns += 1
        if total_assigns < 2:
            return 0.0
        return max(dept_counts.values()) / total_assigns

    def _anti_exploit_penalty(self, trajectory: List[Dict]) -> float:
        """Multiplicative penalty in [0.2, 1.0] for degenerate policies."""
        if not trajectory:
            return 1.0

        skip_r = self._skip_rate(trajectory)
        defer_r = self._defer_rate(trajectory)
        entropy = self._action_entropy(trajectory)
        dept_conc = self._dept_concentration(trajectory)

        penalty = 1.0
        # Penalize always-skip
        if skip_r > 0.80:
            penalty = min(penalty, 0.25)
        elif skip_r > 0.60:
            penalty = min(penalty, 0.45)
        elif skip_r > 0.40:
            penalty = min(penalty, 0.70)

        # Penalize always-defer
        if defer_r > 0.70:
            penalty = min(penalty, 0.35)
        elif defer_r > 0.50:
            penalty = min(penalty, 0.55)

        # Penalize extremely low full-action entropy (constant action)
        if entropy < 0.10:
            penalty = min(penalty, 0.30)
        elif entropy < 0.20:
            penalty = min(penalty, 0.55)

        # Penalize routing everything to one department
        if dept_conc > 0.90:
            penalty = min(penalty, 0.40)
        elif dept_conc > 0.75:
            penalty = min(penalty, 0.65)

        return float(max(0.2, min(1.0, penalty)))

    # ------------------------------------------------------------------
    # Per-step scoring
    # ------------------------------------------------------------------

    def _step_process_score(self, step_data: Dict) -> float:
        """Score whether the action at this step was locally sensible.

        Returns a value in [0, 1].
        Default: 0.60 (neutral) when no violation is detected.
        """
        action = step_data.get("action", "skip")
        valid = step_data.get("valid", True)
        success = step_data.get("success", True)
        snap = step_data.get("state_snapshot", {})

        if not valid:
            return 0.0

        if action == "skip":
            if snap.get("pending_count", 0) == 0:
                return 0.75  # correct to skip empty queue
            return 0.15  # skipping when tickets are waiting

        if action == "assign" and success:
            return 0.90

        if action == "escalate" and success:
            return 0.60

        if action == "defer":
            return 0.30

        if action == "resolve" and success:
            return 0.85

        return 0.50  # neutral / failed action

    def _compute_process_scores(self, trajectory: List[Dict]) -> List[float]:
        return [self._step_process_score(step) for step in trajectory]

    # ------------------------------------------------------------------
    # Trajectory-aware scoring
    # ------------------------------------------------------------------

    def _improvement_score(self, trajectory: List[Dict]) -> float:
        """Compare second-half vs first-half process quality.

        Returns [0, 1]: 1.0 = significant improvement, 0.5 = flat.
        """
        if len(trajectory) < 4:
            return 0.5
        scores = self._compute_process_scores(trajectory)
        mid = len(scores) // 2
        first_half = self._safe_mean(scores[:mid], 0.5)
        second_half = self._safe_mean(scores[mid:], 0.5)
        delta = second_half - first_half
        return self._normalise(delta + 0.3, 0.0, 0.6)

    def _tail_risk_score(self, trajectory: List[Dict]) -> float:
        """25th-percentile of per-step process scores (worst quartile)."""
        scores = self._compute_process_scores(trajectory)
        if len(scores) < 4:
            return self._safe_mean(scores, 0.3)
        sorted_s = sorted(scores)
        q25_idx = max(0, len(sorted_s) // 4)
        return sorted_s[q25_idx]

    def _queue_trend_score(self, trajectory: List[Dict]) -> float:
        """Score queue-size trend.  Shrinking = good, growing = bad."""
        sizes = self._extract_stat(trajectory, "queue_size")
        if len(sizes) < 2:
            return 0.5
        n = len(sizes)
        mean_x = (n - 1) / 2.0
        mean_y = sum(sizes) / n
        numer = sum((i - mean_x) * (sizes[i] - mean_y) for i in range(n))
        denom = sum((i - mean_x) ** 2 for i in range(n))
        slope = numer / max(denom, 0.001)
        # slope > 0 = growing (bad), < 0 = shrinking (good)
        return self._invert(self._normalise(slope, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Department fairness (Jain's index)
    # ------------------------------------------------------------------

    @staticmethod
    def _department_fairness(trajectory: List[Dict]) -> float:
        """Jain's fairness index across department loads at end of episode."""
        if not trajectory:
            return 1.0
        last_snap = trajectory[-1].get("state_snapshot", {})
        loads = last_snap.get("department_load", {})
        if not loads or len(loads) <= 1:
            return 1.0
        vals = list(loads.values())
        s = sum(vals)
        sq = sum(v * v for v in vals)
        n = len(vals)
        if sq == 0:
            return 0.5
        return float(min(1.0, (s * s) / (n * sq)))
