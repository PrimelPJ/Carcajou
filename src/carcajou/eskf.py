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
# 24-state extension (cfg.extended_state): accel/gyro scale factor and a
# GNSS Gauss-Markov position error state.
IDX_SA = slice(15, 18)
IDX_SG = slice(18, 21)
IDX_GM = slice(21, 24)


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
    use_map: bool = False
    # Phase 3: extend the error state to 24 = 15 + [ds_a(3), ds_g(3), dgm(3)].
    # Scale factor and axis gain error are what the README has blamed for
    # covariance optimism since Phase 0; the GM state models time-correlated
    # GNSS error (multipath) instead of pretending it is white. Off by
    # default so every published table reproduces exactly; the consistency
    # study in scripts/run_consistency.py is where this earns its keep.
    extended_state: bool = False
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
        self.n = 24 if cfg.extended_state else 15
        # GNSS GM error estimate lives beside the NavState: it is receiver
        # environment, not vehicle state, so NavState stays a pure INS state.
        self.gm = np.zeros(3)

        P = np.zeros((self.n, self.n))
        P[IDX_P, IDX_P] = np.eye(3) * cfg.p0_pos**2
        P[IDX_V, IDX_V] = np.eye(3) * cfg.p0_vel**2
        P[IDX_TH, IDX_TH] = np.diag(
            [cfg.p0_att_horiz**2, cfg.p0_att_horiz**2, cfg.p0_att_yaw**2]
        )
        P[IDX_BA, IDX_BA] = np.eye(3) * cfg.imu.b0_a**2
        P[IDX_BG, IDX_BG] = np.eye(3) * cfg.imu.b0_g**2
        if cfg.extended_state:
            P[IDX_SA, IDX_SA] = np.eye(3) * cfg.imu.sf_a**2
            P[IDX_SG, IDX_SG] = np.eye(3) * cfg.imu.sf_g**2
            P[IDX_GM, IDX_GM] = np.eye(3) * cfg.gnss.sigma_gm**2
        self.P = P

        self._imu_buf: list[ImuSample] = []
        self._gates = {d: float(chi2.ppf(cfg.innovation_gate_p, df=d)) for d in (1, 2, 3, 6)}
        self._static: bool | None = None  # invalidated on every predict()
        self.stats = {
            "gnss_applied": 0, "gnss_rejected": 0, "zupt": 0, "nhc": 0,
            "vo_applied": 0, "vo_rejected": 0,
            "vo_rot_applied": 0, "vo_rot_rejected": 0,
            "map_pos_applied": 0, "map_pos_rejected": 0,
            "map_rot_applied": 0, "map_rot_rejected": 0,
        }

    # ------------------------------------------------------------------ time
    def predict(self, imu: ImuSample, dt: float) -> None:
        if dt <= 0.0:
            return
        F15 = self.mech.transition_matrix(self.state, imu, self.cfg.imu.tau_a, self.cfg.imu.tau_g)
        G15 = self.mech.noise_gain(self.state)
        if self.cfg.extended_state:
            F = np.zeros((24, 24))
            F[0:15, 0:15] = F15
            # Scale-factor coupling. With measured = (1+s)*true + b, the
            # residual specific-force error from ds_a is -diag(f_b) ds_a in
            # the body frame, rotated into nav exactly like a bias error:
            #   d(dv)/dt += -R diag(f_b) ds_a,  d(dtheta)/dt += -R diag(w_b) ds_g
            # The states themselves are random constants (no dynamics, no
            # process noise): a gain error does not wander in-run.
            R = self.state.R
            f_b = (1.0 - self.state.s_a) * (imu.f - self.state.b_a)
            w_b = (1.0 - self.state.s_g) * (imu.w - self.state.b_g)
            F[3:6, IDX_SA] = -R @ np.diag(f_b)
            F[6:9, IDX_SG] = -R @ np.diag(w_b)
            # GNSS GM error: first-order Gauss-Markov, tau from the receiver
            # spec, driven so its stationary variance is sigma_gm^2.
            tau = self.cfg.gnss.tau_gm
            F[IDX_GM, IDX_GM] = -np.eye(3) / tau
            G = np.zeros((24, 15))
            G[0:15, 0:12] = G15
            G[IDX_GM, 12:15] = np.eye(3)
            q_gm = 2.0 * self.cfg.gnss.sigma_gm**2 / tau
            Qc = np.zeros((15, 15))
            Qc[0:12, 0:12] = self.Qc
            Qc[12:15, 12:15] = np.eye(3) * q_gm
        else:
            F, G, Qc = F15, G15, self.Qc

        Phi = np.eye(self.n) + F * dt + 0.5 * (F @ F) * dt * dt
        Qd_inst = G @ Qc @ G.T
        # Trapezoidal van Loan approximation. Cheap, and stable at 100 Hz.
        Qd = 0.5 * (Phi @ Qd_inst @ Phi.T + Qd_inst) * dt

        self.P = Phi @ self.P @ Phi.T + Qd
        self.P = 0.5 * (self.P + self.P.T)
        self.state = self.mech.propagate(self.state, imu, dt)
        if self.cfg.extended_state:
            # The nominal GM estimate decays with the same time constant its
            # error model assumes; an estimate of a mean-reverting process
            # that never reverts contradicts its own covariance.
            self.gm = self.gm * float(np.exp(-dt / self.cfg.gnss.tau_gm))

        self._imu_buf.append(imu)
        if len(self._imu_buf) > self.cfg.zupt_window:
            self._imu_buf.pop(0)
        self._static = None

    # ------------------------------------------------------------ correction
    def _update(
        self, H: np.ndarray, r: np.ndarray, R: np.ndarray,
        gate: bool = True, gate_scale: float = 1.0,
    ) -> bool:
        S = H @ self.P @ H.T + R
        if gate:
            try:
                nis = float(r @ np.linalg.solve(S, r))
            except np.linalg.LinAlgError:
                return False
            thr = self._gates.get(len(r)) or float(
                chi2.ppf(self.cfg.innovation_gate_p, df=len(r))
            )
            if nis > thr * gate_scale:
                return False

        K = np.linalg.solve(S, H @ self.P).T  # == P H^T S^-1, without forming S^-1
        dx = K @ r

        # Joseph form: stays positive definite even with a badly conditioned S.
        I_KH = np.eye(self.n) - K @ H
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
        if self.cfg.extended_state:
            self.state.s_a = self.state.s_a + dx[IDX_SA]
            self.state.s_g = self.state.s_g + dx[IDX_SG]
            self.gm = self.gm + dx[IDX_GM]

        # Reset Jacobian: the attitude error frame just moved.
        G = np.eye(self.n)
        G[IDX_TH, IDX_TH] = np.eye(3) - 0.5 * skew(dtheta)
        self.P = G @ self.P @ G.T

    # -------------------------------------------------------------- aiding
    def update_gnss_position(self, p_meas: np.ndarray, R: np.ndarray | None = None) -> bool:
        H = np.zeros((3, self.n))
        H[:, IDX_P] = np.eye(3)
        r = np.asarray(p_meas, float) - self.state.p
        if self.cfg.extended_state:
            # Measured = true + gm + white: the innovation carries the GM
            # estimate and H exposes the GM error alongside position, so
            # correlated receiver error is attributed to a state with the
            # right time constant instead of being averaged into position
            # as if it were white. This is the whole mechanism.
            H[:, IDX_GM] = np.eye(3)
            r = r - self.gm
        ok = self._update(H, r, self.cfg.gnss.R_position() if R is None else R)
        self.stats["gnss_applied" if ok else "gnss_rejected"] += 1
        return ok

    def update_gnss_velocity(self, v_meas: np.ndarray, R: np.ndarray | None = None) -> bool:
        H = np.zeros((3, self.n))
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
        H = np.zeros((3, self.n))
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
        H = np.zeros((2, self.n))
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
        H = np.zeros((3, self.n))
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
        H = np.zeros((3, self.n))
        H[:, IDX_BG] = np.eye(3) * dt
        Rm = np.asarray(R_rot, float) + np.eye(3) * (self.cfg.imu.arw**2 * dt)
        Rm = Rm * self.cfg.vo_sigma_scale**2 + np.eye(3) * 1e-10
        ok = self._update(H, r, 0.5 * (Rm + Rm.T))
        self.stats["vo_rot_applied" if ok else "vo_rot_rejected"] += 1
        return ok

    def update_map_position(self, p_meas: np.ndarray, R: np.ndarray) -> bool:
        """Absolute position from scan-to-map registration.

        Structurally identical to the GNSS position update — an absolute nav-
        frame position with its own covariance — which is the entire point of
        the two-pass design: a persisted map turns a relative sensor into an
        absolute one, and the filter consumes it through the same 3-DOF
        measurement model GNSS uses, gate and all.
        """
        H = np.zeros((3, self.n))
        H[:, IDX_P] = np.eye(3)
        r = np.asarray(p_meas, float) - self.state.p
        # Wider gate than GNSS, deliberately. The matcher's own acceptance
        # gates already vet correctness by consensus (a wrong-basin pose
        # cannot explain 500 points at scan noise), so the failure the
        # innovation gate exists to catch is largely pre-filtered. Meanwhile
        # this filter's covariance is documented as mildly optimistic, and a
        # tight gate on an optimistic P turns a few metres of honest drift
        # into a rejection cascade: fix refused -> drift grows -> next fix
        # refused harder -> map lost. Absolute re-acquisition must survive
        # its own prior being wrong; that is what makes it re-acquisition.
        ok = self._update(H, r, np.asarray(R, float), gate_scale=25.0)
        self.stats["map_pos_applied" if ok else "map_pos_rejected"] += 1
        return ok

    def update_map_rotation(self, R_meas: np.ndarray, R: np.ndarray) -> bool:
        """Absolute attitude from scan-to-map registration.

        With the global/left error convention, ``R_true = Exp(dtheta) R_est``,
        so ``Log(R_meas R_est^T)`` measures ``dtheta`` directly and H is the
        identity on the attitude block. This is the update GNSS cannot
        provide at all: it observes heading without needing motion, which is
        what keeps yaw bounded through the long, straight tunnel segments
        where NHC's observability is weakest.
        """
        r = log_so3(np.asarray(R_meas, float) @ self.state.R.T)
        H = np.zeros((3, self.n))
        H[:, IDX_TH] = np.eye(3)
        ok = self._update(H, r, np.asarray(R, float), gate_scale=25.0)
        self.stats["map_rot_applied" if ok else "map_rot_rejected"] += 1
        return ok

    # -------------------------------------------------------------- helpers
    def sigma(self) -> np.ndarray:
        """One-sigma marginal uncertainty for the 15 error states."""
        return np.sqrt(np.clip(np.diag(self.P), 0.0, None))
