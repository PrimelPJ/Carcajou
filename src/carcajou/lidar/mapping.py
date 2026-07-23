"""Map building and persistence: the two-pass structure.

Pass 1 (mapping): drive the route with GNSS available, pose each scan at the
filter's estimate, remove dynamic returns with the segmentation mask, and
consolidate what survives into a voxel map. The map inherits the mapping
pass's pose error — a map is not truth, it is a snapshot of your best past
answer, which is exactly what makes the second pass an honest experiment.

Pass 2 (localization): drive the same route with GNSS cut and register each
scan against the persisted map. The claim under test is the one in the
roadmap: same sensors, better answer, because prior structure turns relative
odometry into absolute positioning. Scan-to-scan odometry drifts; scan-to-map
does not, because the map does not move.

Dynamic removal happens at *mapping* time by mask and at *localization* time
by trimming. The asymmetry is deliberate: a parked-then-departed car baked
into the map is a permanent phantom that corrupts every future localization,
while the same car during localization is one scan's outliers. Mapping is
where contamination compounds, so mapping is where the mask spends its
budget. This mirrors the "AI-labelled HD map" claim being benchmarked: the
labelling earns its keep at map-build time.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..vision.segmentation import SimulatedMask
from .registration import voxel_downsample
from .scanner import Scan

MAP_FORMAT_VERSION = 1


@dataclass
class LandmarkMap:
    """Persisted static structure in the nav frame."""

    points: np.ndarray  # (M, 3) voxel centroids
    sigma: np.ndarray  # (M,) one-sigma per centroid
    voxel: float

    def tree(self) -> cKDTree:
        return cKDTree(self.points)

    # ---------------------------------------------------------- persistence
    def save(self, path: str | pathlib.Path) -> None:
        np.savez_compressed(
            path,
            version=MAP_FORMAT_VERSION,
            points=self.points,
            sigma=self.sigma,
            voxel=self.voxel,
        )

    @classmethod
    def load(cls, path: str | pathlib.Path) -> LandmarkMap:
        d = np.load(path)
        if int(d["version"]) != MAP_FORMAT_VERSION:
            raise ValueError(
                f"map format {int(d['version'])} != {MAP_FORMAT_VERSION}; rebuild the map"
            )
        return cls(points=d["points"], sigma=d["sigma"], voxel=float(d["voxel"]))


def build_map(
    scans: list[Scan],
    poses: dict[int, tuple[np.ndarray, np.ndarray]],
    mask: SimulatedMask,
    rng: np.random.Generator,
    voxel: float = 0.5,
) -> LandmarkMap:
    """Accumulate posed scans into a voxel map, dynamic returns removed.

    ``poses`` maps trajectory index to the (R, p) *sensor* pose used to place
    each scan — in the benchmark this is the mapping filter's estimate, never
    truth, so the map honestly carries the mapping pass's error.

    The mask is applied per scan in label space, the same abstraction and the
    same operating points as Phase 1. A missed-object scan leaks the whole
    actor into the map for that sweep; voxel consolidation then dilutes it
    against every clean sweep of the same voxel, which is the mechanism that
    makes mapping tolerant of occasional segmentation failures and single-scan
    localization not.
    """
    pts, sig = [], []
    for s in scans:
        pose = poses.get(s.index)
        if pose is None:
            continue
        R, p = pose
        keep = mask.keep(s.is_dynamic, rng)
        if not keep.any():
            continue
        pts.append(s.points[keep] @ R.T + p)
        sig.append(s.sigma[keep])
    if not pts:
        return LandmarkMap(np.zeros((0, 3)), np.zeros(0), voxel)
    points = np.vstack(pts)
    sigma = np.concatenate(sig)
    cent, cent_sigma = voxel_downsample(points, voxel, sigma)
    return LandmarkMap(points=cent, sigma=cent_sigma, voxel=voxel)
