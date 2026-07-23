"""15-state error-state Kalman filter for GNSS/INS integration.

Error state (15): ``[dp(3), dv(3), dtheta(3), db_a(3), db_g(3)]``

The nominal state lives in :class:`~carcajou.mechanization.NavState` and is
propagated by the full nonlinear mechanization. The filter only ever tracks
the *error*, which is small, so the linearisation stays honest. After every
measurement update the error is injected into the nominal state and reset to
zero, with the covariance rotated through the reset Jacobian.

Aiding updates implemented here are the ones that need no extra hardware:

* **GNSS position / velocity** -- the primary update, with a chi-square gate
  so a multipath outlier cannot poison the biases.
* **ZUPT** (zero-velocity) -- fires when the vehicle is detected stationary.
  This is the single highest-leverage trick during a GNSS outage: it observes
  the accelerometer bias directly and stops velocity error from integrating
  into a quadratic position ramp.
* **NHC** (non-holonomic constraint) -- a wheeled vehicle does not slide
  sideways or leave the road surface, so lateral and vertical body-frame
  velocity are ~0. Nearly free, and it bounds heading drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .frames import exp_so3, log_so3, orthonormalize, skew
from .mechanization import ImuSample, Mechanizer, NavState
from .sensors import GnssSpec, ImuSpec

IDX_P = slice(0, 3)
IDX_V = slice(3, 6)
IDX_TH = slice(6, 9)
IDX_BA = slice(9, 12)
IDX_BG = slice(12, 15)


@dataclass
class EskfConfig:
    imu: ImuSpec
    gnss: GnssSpec
    use_gnss_velocity: bool = True
    use_zupt: bool = True
    use_nhc: bool = False
    use_vo: bool = False
    # Implemented but off by default: the relative-rotation channel's reported
    # noise is optimistic because successive VO intervals share an epoch and
    # their errors are therefore correlated, which this filter cannot carry
    # without stochastic cloning. Enabling it improves the gyro-bias estimate
    # and degrades position unless the covariance is inflated by a factor with
    # no principled derivation. Cloning is Phase 3; until then the honest
    # default is off. See update_vo_rotation.
    use_vo_rotation: bool = False
    # ZUPT detector thresholds, tuned for a road vehicle at 100 Hz.
    zupt_window: int = 20
    zupt_accel_thresh: float = 0.25  # m/s^2, deviation of |f| from |g|
    zupt_gyro_thresh: float = 0.02  # rad/s
    zupt_sigma: float = 0.02  # m/s
    nhc_sigma: float = 0.15  # m/s
    vo_sigma_scale: float = 1.0  # multiplier on the front end's reported R
    innovation_gate_p: float = 0.999  # chi-square gate confidence
    # Initial uncertainty
    p0_pos: float = 5.0
    p0_vel: float = 1.0
    p0_att_horiz: float = np.deg2rad(1.0)
    p0_att_yaw: float = np.deg2rad(5.0)


class Eskf:
    """Error-state KF wrapping a :class:`Mechanizer`."""

    def __init__(self, mech: Mechanizer, cfg: EskfConfig, state: NavState) -> None:
        self.mech = mech
        self.cfg = cfg
        self.state = state.copy()
        self.Qc = cfg.imu.process_noise()

        P = np.zeros((15, 15))
        P[IDX_P, IDX_P] = np.eye(3) * cfg.p0_pos**2
        P[IDX_V, IDX_V] = np.eye(3) * cfg.p0_vel**2
        P[IDX_TH, IDX_TH] = np.diag(
            [cfg.p0_att_horiz**2, cfg.p0_att_horiz**2, cfg.p0_att_yaw**2]
        )
        P[IDX_BA, IDX_BA] = np.eye(3) * cfg.imu.b0_a**2
        P[IDX_BG, IDX_BG] = np.eye(3) * cfg.imu.b0_g**2
        self.P = P

        self._imu_buf: list[ImuSample] = []
        self._gates = {d: float(chi2.ppf(cfg.innovation_gate_p, df=d)) for d in (1, 2, 3, 6)}
        self._static: bool | None = None  # invalidated on every predict()
        self.stats = {
            "gnss_applied": 0, "gnss_rejected": 0, "zupt": 0, "nhc": 0,
            "vo_applied": 0, "vo_rejected": 0,
            "vo_rot_applied": 0, "vo_rot_rejected": 0,
        }

    # ------------------------------------------------------------------ time
    def predict(self, imu: ImuSample, dt: float) -> None:
        if dt <= 0.0:
            return
        F = self.mech.transition_matrix(self.state, imu, self.cfg.imu.tau_a, self.cfg.imu.tau_g)
        G = self.mech.noise_gain(self.state)

        Phi = np.eye(15) + F * dt + 0.5 * (F @ F) * dt * dt
        Qd_inst = G @ self.Qc @ G.T
        # Trapezoidal van Loan approximation. Cheap, and stable at 100 Hz.
        Qd = 0.5 * (Phi @ Qd_inst @ Phi.T + Qd_inst) * dt

        self.P = Phi @ self.P @ Phi.T + Qd
        self.P = 0.5 * (self.P + self.P.T)
        self.state = self.mech.propagate(self.state, imu, dt)

        self._imu_buf.append(imu)
        if len(self._imu_buf) > self.cfg.zupt_window:
            self._imu_buf.pop(0)
        self._static = None

    # ------------------------------------------------------------ correction
    def _update(self, H: np.ndarray, r: np.ndarray, R: np.ndarray, gate: bool = True) -> bool:
        S = H @ self.P @ H.T + R
        if gate:
            try:
                nis = float(r @ np.linalg.solve(S, r))
            except np.linalg.LinAlgError:
                return False
            gate = self._gates.get(len(r)) or float(
                chi2.ppf(self.cfg.innovation_gate_p, df=len(r))
            )
            if nis > gate:
                return False

        K = np.linalg.solve(S, H @ self.P).T  # == P H^T S^-1, without forming S^-1
        dx = K @ r

        # Joseph form: stays positive definite even with a badly conditioned S.
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        self._inject(dx)
        return True

    def _inject(self, dx: np.ndarray) -> None:
        """Fold the error estimate into the nominal state, then reset."""
        self.state.p = self.state.p + dx[IDX_P]
        self.state.v = self.state.v + dx[IDX_V]
        dtheta = dx[IDX_TH]
        # Global/left error convention, matching the F matrix.
        self.state.R = orthonormalize(exp_so3(dtheta) @ self.state.R)
        self.state.b_a = self.state.b_a + dx[IDX_BA]
        self.state.b_g = self.state.b_g + dx[IDX_BG]

        # Reset Jacobian: the attitude error frame just moved.
        G = np.eye(15)
        G[IDX_TH, IDX_TH] = np.eye(3) - 0.5 * skew(dtheta)
        self.P = G @ self.P @ G.T

    # -------------------------------------------------------------- aiding
    def update_gnss_position(self, p_meas: np.ndarray, R: np.ndarray | None = None) -> bool:
        H = np.zeros((3, 15))
        H[:, IDX_P] = np.eye(3)
        r = np.asarray(p_meas, float) - self.state.p
        ok = self._update(H, r, self.cfg.gnss.R_position() if R is None else R)
        self.stats["gnss_applied" if ok else "gnss_rejected"] += 1
        return ok

    def update_gnss_velocity(self, v_meas: np.ndarray, R: np.ndarray | None = None) -> bool:
        H = np.zeros((3, 15))
        H[:, IDX_V] = np.eye(3)
        r = np.asarray(v_meas, float) - self.state.v
        return self._update(H, r, self.cfg.gnss.R_velocity() if R is None else R)

    def is_stationary(self) -> bool:
        """Variance-based static detector over the recent IMU window.

        Memoized per epoch: both the ZUPT and NHC paths ask, and recomputing
        window statistics twice per 100 Hz sample is the single most expensive
        thing this filter can do for no reason.
        """
        if self._static is not None:
            return self._static
        if len(self._imu_buf) < self.cfg.zupt_window:
            self._static = False
            return False
        f = np.array([s.f for s in self._imu_buf])
        w = np.array([s.w for s in self._imu_buf])
        g_mag = float(np.linalg.norm(self.mech.g))
        accel_ok = abs(np.linalg.norm(f.mean(axis=0)) - g_mag) < self.cfg.zupt_accel_thresh
        accel_ok &= float(np.linalg.norm(f.std(axis=0))) < self.cfg.zupt_accel_thresh
        gyro_ok = float(np.linalg.norm(w.mean(axis=0))) < self.cfg.zupt_gyro_thresh
        self._static = bool(accel_ok and gyro_ok)
        return self._static

    def update_zupt(self) -> bool:
        H = np.zeros((3, 15))
        H[:, IDX_V] = np.eye(3)
        r = -self.state.v
        ok = self._update(H, r, np.eye(3) * self.cfg.zupt_sigma**2)
        self.stats["zupt"] += int(ok)
        return ok

    def update_nhc(self) -> bool:
        """Constrain lateral and vertical velocity in the body frame to zero."""
        Rbn = self.state.R
        v_b = Rbn.T @ self.state.v
        sel = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # body y and z
        H = np.zeros((2, 15))
        H[:, IDX_V] = sel @ Rbn.T
        H[:, IDX_TH] = sel @ Rbn.T @ skew(self.state.v)
        r = -sel @ v_b
        ok = self._update(H, r, np.eye(2) * self.cfg.nhc_sigma**2)
        self.stats["nhc"] += int(ok)
        return ok

    def update_vo_velocity(self, v_b_meas: np.ndarray, R: np.ndarray) -> bool:
        """Body-frame velocity update from stereo visual odometry.

        Same measurement function as NHC, without the selector: NHC asserts that
        two components of ``R^T v`` are zero, VO measures all three. The shared
        structure is not a coincidence. Both are body-frame velocity
        constraints, which is also why the attitude coupling is identical and
        why VO bounds heading drift for the same reason NHC does.

        ``R`` comes from the front end, per epoch, and is not a tuning constant.
        A frame that tracked forty far-field points on a wet road reports a
        worse covariance than one that tracked three hundred at fifteen metres,
        and the filter should believe it less. Any front end wired in here must
        supply a covariance it can defend; the chi-square gate will reject an
        optimistic one, which shows up as a rejection rate rather than as drift,
        and that is the failure mode you want.
        """
        Rbn = self.state.R
        H = np.zeros((3, 15))
        H[:, IDX_V] = Rbn.T
        H[:, IDX_TH] = Rbn.T @ skew(self.state.v)
        r = np.asarray(v_b_meas, float) - Rbn.T @ self.state.v
        ok = self._update(H, r, np.asarray(R, float) * self.cfg.vo_sigma_scale**2)
        self.stats["vo_applied" if ok else "vo_rejected"] += 1
        return ok

    def update_vo_rotation(
        self, dR_b_meas: np.ndarray, G_raw: np.ndarray, dt: float, R_rot: np.ndarray
    ) -> bool:
        """Relative-rotation update from stereo VO, observing gyro bias.

        VO's rotation between two epochs and the gyro's integrated rotation
        over the same interval measure the same physical quantity through
        independent errors. Their disagreement is, to first order, the gyro
        bias error integrated over the interval, which is exactly the term
        NHC and the velocity update observe only weakly. This is the channel
        that bounds heading drift, the limiting error source for consumer
        MEMS once velocity is aided.

        ``G_raw`` is the *uncorrected* gyro rotation over the interval,
        ``prod Exp(w_k dt)``, and the current bias estimate is removed here:
        ``G_corr = G_raw Exp(-b_g T)``. Comparing against raw gyro rather
        than the filter's own attitude history is deliberate. Between two VO
        epochs the filter applies a dozen NHC and velocity updates, each of
        which nudges attitude; a prediction built from filter attitudes
        inherits those state-correlated corrections and the residual stops
        being the clean bias observation the H matrix claims. Raw gyro keeps
        the two sides of the comparison independent, which is the property
        the whole update rests on.

        First-order approximations, stated: the bias correction is applied as
        a single right-multiplied rotation rather than per sample (commutator
        error ~ |w| |b| T^2, nanoradians here), and the measurement noise
        carries the gyro ARW over the interval so the residual is never
        trusted below the sensor's own floor.
        """
        T = dt
        G_corr = np.asarray(G_raw, float) @ exp_so3(-self.state.b_g * T)
        dR_pred = G_corr.T  # matches dR_meas = R_curr^T R_prev
        r = log_so3(dR_pred.T @ np.asarray(dR_b_meas, float))
        H = np.zeros((3, 15))
        H[:, IDX_BG] = np.eye(3) * dt
        Rm = np.asarray(R_rot, float) + np.eye(3) * (self.cfg.imu.arw**2 * dt)
        Rm = Rm * self.cfg.vo_sigma_scale**2 + np.eye(3) * 1e-10
        ok = self._update(H, r, 0.5 * (Rm + Rm.T))
        self.stats["vo_rot_applied" if ok else "vo_rot_rejected"] += 1
        return ok

    # -------------------------------------------------------------- helpers
    def sigma(self) -> np.ndarray:
        """One-sigma marginal uncertainty for the 15 error states."""
        return np.sqrt(np.clip(np.diag(self.P), 0.0, None))
