"""Strapdown INS mechanization in a local-level NED frame.

This is the dead-reckoning core: given specific force and angular rate in the
body frame, propagate attitude, velocity and position forward. Everything else
in carcajou exists to stop this from drifting.

Modelled
--------
* Earth rotation (``omega_ie``) in both the attitude and Coriolis terms.
* Somigliana normal gravity with a free-air height correction.
* Second-order rotation-vector attitude update.

Deliberately omitted
--------------------
* Transport rate (``omega_en``). On a fixed tangent plane this is zero by
  construction; it reappears only if you move to curvilinear mechanization.
* Coning and sculling compensation. At the 100 Hz sample rates used here the
  residual is well below the MEMS noise floor. Add it before you go to a
  tactical-grade IMU at low output rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .frames import earth_rate_ned, exp_so3, gravity_ned, orthonormalize, skew


@dataclass
class NavState:
    """Full inertial navigation state at one epoch."""

    t: float = 0.0
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))  # NED position, m
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))  # NED velocity, m/s
    R: np.ndarray = field(default_factory=lambda: np.eye(3))  # body -> nav
    b_a: np.ndarray = field(default_factory=lambda: np.zeros(3))  # accel bias, m/s^2
    b_g: np.ndarray = field(default_factory=lambda: np.zeros(3))  # gyro bias, rad/s

    def copy(self) -> NavState:
        return replace(
            self,
            p=self.p.copy(),
            v=self.v.copy(),
            R=self.R.copy(),
            b_a=self.b_a.copy(),
            b_g=self.b_g.copy(),
        )


@dataclass(frozen=True)
class ImuSample:
    """One IMU epoch. ``f`` is specific force (m/s^2), ``w`` angular rate (rad/s)."""

    t: float
    f: np.ndarray
    w: np.ndarray


class Mechanizer:
    """Propagates a :class:`NavState` using raw IMU samples."""

    def __init__(self, lat_rad: float, height: float = 0.0) -> None:
        self.lat = lat_rad
        self.height = height
        self.w_ie = earth_rate_ned(lat_rad)
        self.g = gravity_ned(lat_rad, height)

    def propagate(self, state: NavState, imu: ImuSample, dt: float) -> NavState:
        """Advance ``state`` by ``dt`` seconds using one IMU sample."""
        if dt <= 0.0:
            return state.copy()

        f_b = imu.f - state.b_a
        w_b = imu.w - state.b_g

        # --- attitude -------------------------------------------------------
        # Body rate relative to the nav frame, expressed in body axes.
        w_nb_b = w_b - state.R.T @ self.w_ie
        R_new = orthonormalize(state.R @ exp_so3(w_nb_b * dt))

        # --- velocity -------------------------------------------------------
        # Trapezoidal on the rotated specific force keeps first-order sculling
        # error out of the velocity increment.
        f_n = 0.5 * (state.R @ f_b + R_new @ f_b)
        a_n = f_n + self.g - 2.0 * np.cross(self.w_ie, state.v)
        v_new = state.v + a_n * dt

        # --- position -------------------------------------------------------
        p_new = state.p + 0.5 * (state.v + v_new) * dt

        return NavState(
            t=state.t + dt,
            p=p_new,
            v=v_new,
            R=R_new,
            b_a=state.b_a.copy(),
            b_g=state.b_g.copy(),
        )

    def transition_matrix(self, state: NavState, imu: ImuSample, tau_a: float, tau_g: float):
        """Continuous-time error-state dynamics matrix ``F`` (15x15).

        Error state ordering: ``[dp, dv, dtheta, db_a, db_g]``.
        Attitude error is global/left: ``R_true = Exp(dtheta) @ R_est``, which
        gives ``d(dv)/dt = -skew(R f) dtheta - R db_a``.
        """
        F = np.zeros((15, 15))
        R = state.R
        f_n = R @ (imu.f - state.b_a)

        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -skew(f_n)
        F[3:6, 9:12] = -R
        F[3:6, 3:6] = -2.0 * skew(self.w_ie)
        F[6:9, 6:9] = -skew(self.w_ie)
        F[6:9, 12:15] = -R
        if np.isfinite(tau_a) and tau_a > 0:
            F[9:12, 9:12] = -np.eye(3) / tau_a
        if np.isfinite(tau_g) and tau_g > 0:
            F[12:15, 12:15] = -np.eye(3) / tau_g
        return F

    def noise_gain(self, state: NavState) -> np.ndarray:
        """Process-noise input matrix ``G`` (15x12) for ``[n_a, n_g, w_ba, w_bg]``."""
        G = np.zeros((15, 12))
        G[3:6, 0:3] = -state.R
        G[6:9, 3:6] = -state.R
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)
        return G


def dead_reckon(mech: Mechanizer, state: NavState, imus, times) -> list[NavState]:
    """Pure INS integration over a batch of samples. No aiding, no filter."""
    out = [state.copy()]
    cur = state
    for imu, t_prev in zip(imus, times, strict=False):
        cur = mech.propagate(cur, imu, imu.t - t_prev)
        out.append(cur)
    return out
