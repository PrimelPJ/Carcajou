"""Trajectory error metrics.

The headline metric is **drift as a percentage of distance travelled** during a
GNSS outage. It is the number inertial navigation vendors quote, it is
scale-invariant, and it is the one an integrator can hold a supplier to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OutageResult:
    """Scoring for a single GNSS outage window."""

    t_start: float
    duration: float
    distance: float  # metres travelled during the outage
    final_error_3d: float
    final_error_horizontal: float
    final_error_vertical: float
    max_error_horizontal: float
    heading_error_deg: float

    @property
    def drift_pct(self) -> float:
        """Horizontal position error as a percentage of distance travelled."""
        if self.distance < 1.0:
            return float("nan")
        return 100.0 * self.final_error_horizontal / self.distance


def horizontal(err: np.ndarray) -> np.ndarray:
    return np.linalg.norm(err[..., :2], axis=-1)


def ate_rmse(est_p: np.ndarray, true_p: np.ndarray) -> float:
    """Absolute trajectory error, RMSE over all epochs (3D)."""
    return float(np.sqrt(np.mean(np.sum((est_p - true_p) ** 2, axis=1))))


def summarize(results: list[OutageResult]) -> dict[str, float]:
    """Aggregate a set of outage windows. p95 matters more than the mean here:
    a navigation system is judged on its bad minutes, not its average one."""
    if not results:
        return {}
    drift = np.array([r.drift_pct for r in results], float)
    err = np.array([r.final_error_horizontal for r in results], float)
    hdg = np.array([r.heading_error_deg for r in results], float)
    drift = drift[np.isfinite(drift)]
    return {
        "n_windows": float(len(results)),
        "drift_pct_median": float(np.median(drift)),
        "drift_pct_p95": float(np.percentile(drift, 95)),
        "drift_pct_max": float(np.max(drift)),
        "err_m_median": float(np.median(err)),
        "err_m_p95": float(np.percentile(err, 95)),
        "err_m_max": float(np.max(err)),
        "heading_err_deg_p95": float(np.percentile(np.abs(hdg), 95)),
    }
