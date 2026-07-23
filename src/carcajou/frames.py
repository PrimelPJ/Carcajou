"""Rotation and coordinate-frame utilities.

Conventions used throughout carcajou
------------------------------------
Navigation frame : local-level NED (North, East, Down) tangent plane, origin
                   pinned to the first valid GNSS fix.
Body frame       : FRD (Forward, Right, Down), the automotive/aerospace standard.
Attitude         : ``R`` maps body vectors into the nav frame, ``v_n = R @ v_b``.
Gravity          : ``g_n = [0, 0, +g]`` (down positive), so a level, stationary
                   accelerometer reads a specific force of ``[0, 0, -g]``.
Attitude error   : *global* (left) parametrisation, ``R_true = Exp(dtheta) @ R_est``.
                   The ESKF Jacobians in :mod:`carcajou.eskf` assume this and
                   nothing else; changing it means rederiving F.
"""

from __future__ import annotations

import numpy as np

# WGS-84
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
EARTH_RATE = 7.292115e-5  # rad/s


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix such that ``skew(a) @ b == np.cross(a, b)``."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_so3(phi: np.ndarray) -> np.ndarray:
    """Exponential map from a rotation vector to SO(3) (Rodrigues)."""
    theta = float(np.linalg.norm(phi))
    if theta < 1e-12:
        # Second-order series keeps this accurate and orthogonal near zero.
        K = skew(phi)
        return np.eye(3) + K + 0.5 * K @ K
    axis = phi / theta
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Logarithm map from SO(3) to a rotation vector."""
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-12:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (theta / (2.0 * np.sin(theta)))


def euler_to_dcm(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX (yaw-pitch-roll) Euler angles to a body-to-nav DCM."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def dcm_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Body-to-nav DCM to ZYX Euler angles ``(roll, pitch, yaw)`` in radians."""
    pitch = float(np.arcsin(-np.clip(R[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Cheap Newton step back onto SO(3): ``R (3I - R^T R) / 2``.

    Quadratically convergent and exact to second order for matrices already
    close to orthogonal, which is every matrix this sees in the hot loop. It
    is roughly 20x faster than an SVD and the residual after one step is below
    1e-16 for a per-epoch rotation increment. Use :func:`project_so3` if you
    ever need to repair a badly degraded matrix.
    """
    return R @ (1.5 * np.eye(3) - 0.5 * (R.T @ R))


def project_so3(R: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix in the Frobenius sense (SVD, determinant +1)."""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def gravity_ned(lat_rad: float, height: float = 0.0) -> np.ndarray:
    """Somigliana normal gravity with a free-air correction, expressed in NED."""
    s2 = np.sin(lat_rad) ** 2
    g0 = 9.7803253359 * (1.0 + 0.001931853 * s2) / np.sqrt(1.0 - WGS84_E2 * s2)
    g = g0 * (1.0 - 2.0 * height / WGS84_A)
    return np.array([0.0, 0.0, g])


def earth_rate_ned(lat_rad: float) -> np.ndarray:
    """Earth rotation rate resolved in the local-level NED frame."""
    return EARTH_RATE * np.array([np.cos(lat_rad), 0.0, -np.sin(lat_rad)])


def llh_to_ecef(lat: float, lon: float, h: float) -> np.ndarray:
    """Geodetic latitude/longitude (radians) and height (m) to ECEF."""
    sl, cl = np.sin(lat), np.cos(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
    return np.array(
        [(N + h) * cl * np.cos(lon), (N + h) * cl * np.sin(lon), (N * (1.0 - WGS84_E2) + h) * sl]
    )


def ecef_to_ned_dcm(lat: float, lon: float) -> np.ndarray:
    """DCM rotating an ECEF vector into the local NED frame at ``(lat, lon)``."""
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    return np.array([[-sl * co, -sl * so, cl], [-so, co, 0.0], [-cl * co, -cl * so, -sl]])


class LocalTangentPlane:
    """Fixed-origin NED tangent plane.

    Valid to well under a centimetre of linearisation error over the few-km
    trajectories this stack targets. Anything continental-scale wants full
    ECEF mechanization instead.
    """

    def __init__(self, lat0: float, lon0: float, h0: float) -> None:
        self.lat0, self.lon0, self.h0 = lat0, lon0, h0
        self._r0 = llh_to_ecef(lat0, lon0, h0)
        self._C = ecef_to_ned_dcm(lat0, lon0)

    def llh_to_ned(self, lat: float, lon: float, h: float) -> np.ndarray:
        return self._C @ (llh_to_ecef(lat, lon, h) - self._r0)

    def ned_to_llh(self, ned: np.ndarray) -> tuple[float, float, float]:
        ecef = self._C.T @ np.asarray(ned) + self._r0
        x, y, z = ecef
        lon = float(np.arctan2(y, x))
        p = np.hypot(x, y)
        lat = float(np.arctan2(z, p * (1.0 - WGS84_E2)))
        for _ in range(8):  # Bowring fixed-point, converges in ~3
            N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
            h = p / np.cos(lat) - N
            lat = float(np.arctan2(z, p * (1.0 - WGS84_E2 * N / (N + h))))
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        h = float(p / np.cos(lat) - N)
        return lat, lon, h
