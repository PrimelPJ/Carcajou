"""LiDAR sensor model over the landmark world.

Same attribution contract as the vision front end: no rendering, no meshes,
no ray casting. A scan is the set of world landmarks inside the range gate,
expressed in the sensor frame, with range-proportional noise. With zero noise
a scan is an exact rigid transform of the world, so registration must recover
the sensor pose exactly (``test_lidar.py``), and every metre of scan-matching
error in the benchmark is attributable to injected noise, geometry, or the
dynamic returns that Phase 2 exists to remove.

What is deliberately *not* modelled: beam divergence, incidence-angle dropout,
intensity, multi-echo returns, motion distortion within a sweep. Motion
distortion is the one that matters most in practice at automotive speed; it is
absorbed into the per-point noise floor here and stated rather than hidden,
the same treatment coning got in the IMU path.

Frames: the sensor frame is the body frame with a lever arm. Automotive roof
LiDARs are mounted close to the body axes, and unlike the camera there is no
axis-permutation convention to trip over, so the identity mounting keeps every
registration result directly interpretable as a body pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..datasets.synthetic import Trajectory
from ..vision.world import LandmarkWorld


@dataclass(frozen=True)
class LidarSpec:
    """Sensor parameters.

    Attributes
    ----------
    rate_hz : scan rate. 10 Hz is the automotive norm, and matching the VO
        rate keeps the two aiding sources comparable on the same harness.
    max_range : metres. Returns beyond this are dropped.
    min_range : metres. Inside this the sensor sees the roof rack.
    range_sigma : one-sigma range noise at reference range, metres.
    range_sigma_scale : additional noise per metre of range, dimensionless.
        Together these give ``sigma(r) = range_sigma + r * range_sigma_scale``,
        the affine model that fits spinning-LiDAR datasheets well enough.
    max_points : cap per scan, standing in for angular resolution. Applied
        uniformly at random so masking does not change point density.
    p_bl : lever arm from body origin to the sensor, body frame (FRD, so a
        roof mount is negative down).
    """

    rate_hz: float = 10.0
    max_range: float = 60.0
    min_range: float = 2.0
    range_sigma: float = 0.02
    range_sigma_scale: float = 0.001
    max_points: int = 1500
    p_bl: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.0, -1.8]))

    def sigma_at(self, ranges: np.ndarray) -> np.ndarray:
        return self.range_sigma + ranges * self.range_sigma_scale


@dataclass
class Scan:
    """One LiDAR sweep in the sensor frame."""

    t: float
    index: int  # trajectory index the scan was taken at
    points: np.ndarray  # (N, 3) sensor-frame points
    sigma: np.ndarray  # (N,) one-sigma isotropic noise per point
    is_dynamic: np.ndarray  # (N,) truth labels, used only by masks and tests


class LidarScanner:
    """Generate scans of a :class:`LandmarkWorld` along a trajectory."""

    def __init__(
        self,
        spec: LidarSpec,
        world: LandmarkWorld,
        traj: Trajectory,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.spec = spec
        self.world = world
        self.traj = traj
        self.rng = rng or np.random.default_rng(0)
        self._tree = cKDTree(world.p_static)
        stride = max(1, int(round((1.0 / spec.rate_hz) / traj.dt)))
        self.step_indices = np.arange(0, len(traj.t), stride)

    def _sensor_pose(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        R = self.traj.R[k]
        return R, self.traj.p[k] + R @ self.spec.p_bl

    def scan(self, k: int, noisy: bool = True) -> Scan:
        """One sweep at trajectory index ``k``."""
        sp = self.spec
        R_nl, p_nl = self._sensor_pose(k)

        near = np.asarray(self._tree.query_ball_point(p_nl, sp.max_range), dtype=int)
        p_stat = self.world.p_static[near] if near.size else np.zeros((0, 3))
        p_dyn = self.world.actor_points(k)

        p_nav = np.vstack([p_stat, p_dyn])
        is_dyn = np.concatenate(
            [np.zeros(len(p_stat), bool), np.ones(len(p_dyn), bool)]
        )

        P_l = (p_nav - p_nl) @ R_nl  # nav -> sensor
        r = np.linalg.norm(P_l, axis=1)
        keep = (r > sp.min_range) & (r < sp.max_range)
        P_l, r, is_dyn = P_l[keep], r[keep], is_dyn[keep]

        if len(P_l) > sp.max_points:
            sel = self.rng.choice(len(P_l), sp.max_points, replace=False)
            P_l, r, is_dyn = P_l[sel], r[sel], is_dyn[sel]

        sigma = sp.sigma_at(r)
        if noisy:
            # Isotropic 3D noise at the range-dependent sigma. Real range noise
            # is radial; isotropic is a mild pessimisation of the tangential
            # directions and keeps the registration covariance model simple
            # enough to defend.
            P_l = P_l + self.rng.normal(0.0, 1.0, P_l.shape) * sigma[:, None]

        return Scan(
            t=float(self.traj.t[k]), index=int(k), points=P_l, sigma=sigma, is_dynamic=is_dyn
        )

    def run(self, noisy: bool = True) -> list[Scan]:
        return [self.scan(int(k), noisy=noisy) for k in self.step_indices]
