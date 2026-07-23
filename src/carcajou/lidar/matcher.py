"""Online scan-to-map matching, driven by the filter's own estimate.

Unlike the VO source, this cannot be precomputed. ICP is a local method whose
basin is set by the correspondence gate, so its initial pose must be the
navigation filter's *current, drifting* estimate — initialising from truth
would quietly hand the experiment its answer. The cost of honesty is a
coupling: registration quality depends on how far the filter has drifted,
which is exactly the coupling a real system lives with, divergence and all.
A filter that drifts past the basin loses the map and does not get it back;
that failure mode is allowed to happen and shows up in the tail statistics
rather than being defined away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..vision.segmentation import MASK_OFF, SimulatedMask
from .mapping import LandmarkMap
from .registration import icp_point_to_point
from .scanner import LidarScanner


@dataclass
class MapMatch:
    """One accepted registration, converted to body-frame measurements."""

    index: int
    p_b: np.ndarray  # body position in nav frame
    R_b: np.ndarray  # body attitude, body -> nav
    cov_p: np.ndarray  # 3x3 position covariance
    cov_r: np.ndarray  # 3x3 rotation covariance (left/global convention)
    n_matched: int
    rms: float


class MapMatcher:
    """Register live scans against a persisted map.

    ``mask`` filters the *live* scan before registration. MASK_OFF leans
    entirely on ICP trimming to reject dynamic returns; a real mask removes
    them up front. That is the Phase 2 localization ablation, and it is the
    same question Phase 1 asked of RANSAC: is the robust estimator's implicit
    rejection enough, or does the scene need to be cleaned first?
    """

    def __init__(
        self,
        scanner: LidarScanner,
        lmap: LandmarkMap,
        mask: SimulatedMask = MASK_OFF,
        rng: np.random.Generator | None = None,
        min_matched: int = 60,
        max_rms: float = 0.45,
        min_match_fraction: float = 0.3,
    ) -> None:
        self.scanner = scanner
        self.map = lmap
        self.mask = mask
        self.rng = rng or np.random.default_rng(0)
        self.min_matched = min_matched
        self.max_rms = max_rms
        self.min_match_fraction = min_match_fraction
        self._tree = lmap.tree()
        self._steps = set(int(k) for k in scanner.step_indices)

    def wants(self, index: int) -> bool:
        return index in self._steps

    def measure(self, index: int, R_b_est: np.ndarray, p_b_est: np.ndarray) -> MapMatch | None:
        """Scan at ``index``, register from the filter's estimated body pose."""
        s = self.scanner.scan(index, noisy=True)
        keep = self.mask.keep(s.is_dynamic, self.rng)
        pts, sig = s.points[keep], s.sigma[keep]
        if len(pts) < self.min_matched:
            return None

        p_bl = self.scanner.spec.p_bl
        R0 = R_b_est
        t0 = p_b_est + R_b_est @ p_bl
        res = icp_point_to_point(
            pts, sig, self._tree, self.map.points, self.map.sigma, R0, t0,
            min_matched=self.min_matched,
        )
        if (
            not res.converged
            or res.rms_residual > self.max_rms
            or res.n_matched < self.min_match_fraction * len(pts)
        ):
            # Refusing to answer beats answering wrongly with confidence; the
            # filter coasts on inertial + constraints until the next scan.
            # The load-bearing gate is rms, and its threshold is set by
            # measurement, not taste: a correct registration in this world
            # converges to rms ~ 0.06-0.23, while an ICP that snapped to the
            # wrong local minimum lands at rms ~ 0.9 and above; 0.45 sits in
            # the empty margin between them. Match fraction is deliberately a
            # weak floor, not a discriminator: a lead vehicle can honestly be
            # a third of the scan, and those points correctly matching nothing
            # in a clean map must not veto a centimetre-accurate fix. So
            # lost-in-map shows up as missing updates, never confident wrong
            # ones, and a heavy dynamic load costs information, not truth.
            return None

        # Sensor pose -> body pose (identity mounting, lever arm only).
        p_b = res.t - res.R @ p_bl
        # Position covariance includes the lever-arm coupling of rotation error.
        from ..frames import skew

        J = np.hstack([skew(res.R @ p_bl), np.eye(3)])
        cov_p = J @ res.cov @ J.T
        return MapMatch(
            index=index,
            p_b=p_b,
            R_b=res.R,
            cov_p=0.5 * (cov_p + cov_p.T),
            cov_r=res.cov[0:3, 0:3],
            n_matched=res.n_matched,
            rms=res.rms_residual,
        )
