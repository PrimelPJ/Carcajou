"""Scan-to-map registration.

Trimmed, weighted point-to-point ICP against a voxel-consolidated map, with a
covariance from the converged normal equations. NDT was the other candidate;
point-to-point against a consolidated map was chosen because its covariance
falls out of the same machinery already defended in the vision front end
(``rigid_covariance``), and a second registration method with a second
covariance argument is complexity the benchmark does not pay for. The module
boundary is the pose and its covariance, so swapping NDT in later changes no
caller.

The trimming is not a tuning nicety. During the mapping pass, dynamic returns
are removed by the segmentation mask *before* points enter the map; during
localization, the scan still contains them, and the map (correctly) has
nothing for them to match. Trimming plus the residual gate is what keeps a
lead vehicle's returns from dragging the pose toward it — the same failure
Phase 1 measured for VO, arriving here through the correspondence stage
instead of the consensus stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..frames import exp_so3, log_so3, skew


def voxel_downsample(
    points: np.ndarray, voxel: float, sigma: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Consolidate points to per-voxel centroids.

    Returns centroids and, if per-point sigmas were given, the sigma of each
    centroid (reduced by sqrt(count), floored at a quarter voxel so the
    discretisation itself is never reported as better than it is).
    """
    if len(points) == 0:
        return points, np.zeros(0)
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    n_vox = len(counts)
    cent = np.zeros((n_vox, 3))
    np.add.at(cent, inv, points)
    cent /= counts[:, None]
    if sigma is None:
        return cent, np.zeros(n_vox)
    s = np.zeros(n_vox)
    np.add.at(s, inv, sigma**2)
    s = np.sqrt(s) / counts  # sigma of the mean
    return cent, np.maximum(s, 0.25 * voxel)


@dataclass
class RegistrationResult:
    """Pose of the scan frame in the map frame, with everything needed to
    either trust it or reject it."""

    R: np.ndarray  # scan -> map rotation
    t: np.ndarray  # scan origin in map frame
    cov: np.ndarray  # 6x6 covariance of [dtheta, dt] in the map frame
    n_matched: int
    rms_residual: float
    converged: bool


def icp_point_to_point(
    scan: np.ndarray,
    scan_sigma: np.ndarray,
    map_tree: cKDTree,
    map_points: np.ndarray,
    map_sigma: np.ndarray,
    R0: np.ndarray,
    t0: np.ndarray,
    iters: int = 30,
    max_corr: float = 2.0,
    trim: float = 0.85,
    min_matched: int = 60,
) -> RegistrationResult:
    """Trimmed weighted ICP with an initial pose from the filter.

    ``max_corr`` bounds the correspondence search; it is what makes the
    method local, and it is why the filter's own prediction is the initial
    pose rather than identity. A navigation filter mid-outage is wrong by
    metres, not by tens of metres, so a 2 m gate holds; a cold-start global
    localizer is a different problem this repository does not claim to solve.

    ``trim`` keeps the best fraction of correspondences by residual each
    iteration. Its job is the returns the map has no counterpart for:
    dynamic objects during localization, and structure that appeared or
    vanished between passes.
    """
    R, t = R0.copy(), t0.copy()
    n = len(scan)
    fail = RegistrationResult(R, t, np.eye(6) * 1e6, 0, np.inf, False)
    if n < min_matched:
        return fail

    inl = np.zeros(0, np.intp)
    w = np.zeros(0)
    q = np.zeros((0, 3))
    for _ in range(iters):
        P = scan @ R.T + t
        dist, j = map_tree.query(P, k=1, distance_upper_bound=max_corr)
        ok = np.isfinite(dist)
        if ok.sum() < min_matched:
            return fail
        idx = np.where(ok)[0]
        # Trim: keep the best `trim` fraction by residual.
        order = np.argsort(dist[idx])
        keep = idx[order[: max(min_matched, int(trim * len(idx)))]]
        q = map_points[j[keep]]
        p = scan[keep]
        sig = np.sqrt(scan_sigma[keep] ** 2 + map_sigma[j[keep]] ** 2) + 1e-6
        w = 1.0 / sig**2
        wn = w / w.sum()

        # Weighted Kabsch step on the trimmed set.
        P_k = p @ R.T + t
        ca, cb = wn @ P_k, wn @ q
        H = ((P_k - ca) * wn[:, None]).T @ (q - cb)
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        dR = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        R, t = dR @ R, dR @ t + (cb - dR @ ca)
        inl = keep
        if np.linalg.norm(log_so3(dR)) < 1e-8 and np.linalg.norm(cb - dR @ ca) < 1e-8:
            break

    P = scan[inl] @ R.T + t
    resid = q - P
    rms = float(np.sqrt(np.mean(np.sum(resid**2, axis=1))))

    # Covariance from the converged normal equations, chi-square rescaled the
    # same way as the vision front end: residuals the noise model cannot
    # explain grow the reported covariance instead of being averaged away.
    dof = max(3 * len(inl) - 6, 1)
    chi2_n = float((w[:, None] * resid**2).sum() / dof)
    scale = max(chi2_n, 1.0)
    S = np.zeros((len(inl), 3, 3))
    S[:, 0, 1], S[:, 0, 2] = -P[:, 2], P[:, 1]
    S[:, 1, 0], S[:, 1, 2] = P[:, 2], -P[:, 0]
    S[:, 2, 0], S[:, 2, 1] = -P[:, 1], P[:, 0]
    info = np.zeros((6, 6))
    info[0:3, 0:3] = np.einsum("n,nji,njk->ik", w, S, S)
    info[0:3, 3:6] = -np.einsum("n,nji->ij", w, S).T
    info[3:6, 0:3] = info[0:3, 3:6].T
    info[3:6, 3:6] = w.sum() * np.eye(3)
    try:
        cov = scale * np.linalg.inv(info + np.eye(6) * 1e-9)
    except np.linalg.LinAlgError:
        return fail
    return RegistrationResult(R, t, 0.5 * (cov + cov.T), int(len(inl)), rms, True)


def perturb_pose(R: np.ndarray, t: np.ndarray, dtheta: np.ndarray, dt: np.ndarray):
    """Left-perturb a pose, matching the covariance convention above."""
    return exp_so3(dtheta) @ R, t + dt


__all__ = [
    "RegistrationResult",
    "icp_point_to_point",
    "perturb_pose",
    "voxel_downsample",
    "skew",
]
