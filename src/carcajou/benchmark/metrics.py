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

    @property
    def max_drift_pct(self) -> float:
        """Peak horizontal error as a percentage of distance travelled.

        Unlike ``drift_pct`` which uses the final error, this uses the
        worst-case error seen during the outage. A system that drifted
        badly mid-outage but happened to recover by the end will show a
        low ``drift_pct`` but a high ``max_drift_pct``, which is the
        number a safety case cares about.
        """
        if self.distance < 1.0:
            return float("nan")
        return 100.0 * self.max_error_horizontal / self.distance

    @property
    def vertical_drift_pct(self) -> float:
        """Vertical position error as a percentage of distance travelled.

        Complements ``drift_pct`` (horizontal) for applications where the
        vertical channel matters, such as tunnel grades or multi-storey
        parking structures.
        """
        if self.distance < 1.0:
            return float("nan")
        return 100.0 * self.final_error_vertical / self.distance

    def meets_budget(
        self,
        drift_pct: float = 1.0,
        heading_deg: float = 2.0,
        max_error_m: float | None = None,
    ) -> bool:
        """Check whether this outage stays within an error budget.

        Parameters
        ----------
        drift_pct : float
            Maximum allowed horizontal drift as a percentage of distance
            travelled.  Default 1 % — the typical automotive target.
        heading_deg : float
            Maximum allowed absolute heading error in degrees.  Default
            2 deg.
        max_error_m : float or None
            Optional hard cap on horizontal error in metres, regardless
            of distance.  Useful for low-speed manoeuvres where drift-%
            is undefined or misleading.

        Returns
        -------
        bool
            True if all specified budgets are met.
        """
        dp = self.drift_pct
        if np.isfinite(dp) and dp > drift_pct:
            return False
        if abs(self.heading_error_deg) > heading_deg:
            return False
        if max_error_m is not None:
            if self.final_error_horizontal > max_error_m:
                return False
        return True

    def summary_str(self) -> str:
        """One-line human-readable summary for logging."""
        return (
            f"t={self.t_start:.0f}s dur={self.duration:.0f}s "
            f"dist={self.distance:.0f}m "
            f"h_err={self.final_error_horizontal:.2f}m "
            f"drift={self.drift_pct:.3f}%"
        )


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
        "err_m_mean": float(np.mean(err)),
        "drift_pct_mean": float(np.mean(drift)) if len(drift) else float("nan"),
        "heading_err_deg_p95": float(np.percentile(np.abs(hdg), 95)),
    }
