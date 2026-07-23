"""End-to-end run harness.

Two entry points:

``run_aided_pass``
    Filter the whole sequence with GNSS available, snapshotting the state and
    covariance at candidate outage start times. This is the "steady state" a
    real vehicle is in when a tunnel or urban canyon arrives.

``run_outage_study``
    Resume from each snapshot with GNSS cut, propagate for the outage duration
    using inertial plus whatever no-cost aiding is enabled, and score the drift.

Resuming from snapshots rather than re-filtering from scratch is not an
optimisation shortcut. It is the methodologically correct setup: every outage
is evaluated from a converged, calibrated filter, so the numbers reflect
outage performance and not initial-alignment transients.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .benchmark.metrics import OutageResult
from .datasets.synthetic import GnssFix, Trajectory
from .eskf import Eskf, EskfConfig
from .frames import dcm_to_euler, exp_so3
from .mechanization import ImuSample, Mechanizer
from .vision.frontend import VoMeasurement


@dataclass
class Snapshot:
    t: float
    index: int
    filter_state: object


@dataclass
class PassResult:
    t: np.ndarray
    p: np.ndarray
    v: np.ndarray
    sigma_p: np.ndarray
    snapshots: list[Snapshot]
    stats: dict


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def build_filter(traj: Trajectory, cfg: EskfConfig) -> Eskf:
    mech = Mechanizer(traj.lat0, traj.h0)
    return Eskf(mech, cfg, traj.initial_state())


def index_vo(measurements: list[VoMeasurement]) -> dict[int, VoMeasurement]:
    """Key VO measurements by the trajectory index they land on.

    VO is computed from truth, once, independently of any filter. That is what
    keeps the ablation clean: mask-on and mask-off differ only in which
    correspondences the pose estimator saw, and every other input to every
    variant is byte-identical.
    """
    return {m.index: m for m in measurements}


def _gyro_delta(imus: list[ImuSample], m: VoMeasurement, dt: float) -> np.ndarray:
    """Uncorrected gyro rotation over a VO interval, ``prod Exp(w_k dt)``.

    Sample ``k`` rotates the body from epoch ``k`` to ``k+1``, so the interval
    ``[index_prev, index)`` covers exactly the two epochs VO compared.
    """
    G = np.eye(3)
    for j in range(m.index_prev, m.index):
        G = G @ exp_so3(imus[j].w * dt)
    return G


def run_aided_pass(
    traj: Trajectory,
    imus: list[ImuSample],
    fixes: list[GnssFix],
    cfg: EskfConfig,
    snapshot_times: np.ndarray | None = None,
    vo: dict[int, VoMeasurement] | None = None,
) -> PassResult:
    """Filter the full sequence with GNSS on, recording snapshots."""
    ekf = build_filter(traj, cfg)
    dt = traj.dt

    fix_iter = iter(fixes)
    next_fix = next(fix_iter, None)
    snap_targets = list(snapshot_times) if snapshot_times is not None else []
    snap_i = 0

    n = len(imus)
    p_hist = np.zeros((n + 1, 3))
    v_hist = np.zeros((n + 1, 3))
    s_hist = np.zeros((n + 1, 3))
    p_hist[0], v_hist[0] = ekf.state.p, ekf.state.v
    s_hist[0] = ekf.sigma()[0:3]

    snapshots: list[Snapshot] = []

    for k, imu in enumerate(imus):
        ekf.predict(imu, dt)

        while next_fix is not None and next_fix.t <= imu.t + 1e-9:
            ekf.update_gnss_position(next_fix.p)
            if cfg.use_gnss_velocity:
                ekf.update_gnss_velocity(next_fix.v)
            next_fix = next(fix_iter, None)

        if cfg.use_zupt and ekf.is_stationary():
            ekf.update_zupt()
        if cfg.use_nhc and not ekf.is_stationary():
            ekf.update_nhc()
        # VO is not gated on the static detector. A constant-velocity cruise
        # is indistinguishable from standstill to an IMU-variance test (|f| ~ g,
        # gyro ~ 0), so gating VO on it would drop the update on exactly the
        # straight segments where forward drift accumulates fastest. The camera
        # can tell 14 m/s from parked; let it say so, and let the chi-square
        # gate arbitrate.
        if cfg.use_vo and vo is not None:
            m = vo.get(k + 1)
            if m is not None:
                ekf.update_vo_velocity(m.v_b, m.R_v)
                if cfg.use_vo_rotation:
                    ekf.update_vo_rotation(m.dR_b, _gyro_delta(imus, m, dt), m.dt, m.R_rot)

        p_hist[k + 1] = ekf.state.p
        v_hist[k + 1] = ekf.state.v
        s_hist[k + 1] = ekf.sigma()[0:3]

        while snap_i < len(snap_targets) and imu.t >= snap_targets[snap_i] - 1e-9:
            snapshots.append(
                Snapshot(t=imu.t, index=k + 1, filter_state=copy.deepcopy((ekf.state, ekf.P)))
            )
            snap_i += 1

    return PassResult(
        t=np.concatenate([[traj.t[0]], [s.t for s in imus]]),
        p=p_hist,
        v=v_hist,
        sigma_p=s_hist,
        snapshots=snapshots,
        stats=dict(ekf.stats),
    )


def run_outage(
    traj: Trajectory,
    imus: list[ImuSample],
    snap: Snapshot,
    cfg: EskfConfig,
    duration: float,
    vo: dict[int, VoMeasurement] | None = None,
) -> OutageResult:
    """Propagate from a snapshot with GNSS cut and score the resulting drift."""
    ekf = build_filter(traj, cfg)
    state, P = copy.deepcopy(snap.filter_state)
    ekf.state, ekf.P = state, P

    dt = traj.dt
    n_steps = int(round(duration / dt))
    start, stop = snap.index, min(snap.index + n_steps, len(imus))

    max_h = 0.0
    for k in range(start, stop):
        ekf.predict(imus[k], dt)
        if cfg.use_zupt and ekf.is_stationary():
            ekf.update_zupt()
        if cfg.use_nhc and not ekf.is_stationary():
            ekf.update_nhc()
        # VO is not gated on the static detector. A constant-velocity cruise
        # is indistinguishable from standstill to an IMU-variance test (|f| ~ g,
        # gyro ~ 0), so gating VO on it would drop the update on exactly the
        # straight segments where forward drift accumulates fastest. The camera
        # can tell 14 m/s from parked; let it say so, and let the chi-square
        # gate arbitrate.
        if cfg.use_vo and vo is not None:
            m = vo.get(k + 1)
            if m is not None:
                ekf.update_vo_velocity(m.v_b, m.R_v)
                if cfg.use_vo_rotation:
                    ekf.update_vo_rotation(m.dR_b, _gyro_delta(imus, m, dt), m.dt, m.R_rot)
        err = ekf.state.p - traj.p[k + 1]
        max_h = max(max_h, float(np.linalg.norm(err[:2])))

    idx = min(stop, len(traj.p) - 1)
    err = ekf.state.p - traj.p[idx]
    dist = float(
        np.sum(np.linalg.norm(np.diff(traj.p[snap.index : idx + 1], axis=0), axis=1))
    )
    _, _, yaw_err = dcm_to_euler(ekf.state.R @ traj.R[idx].T)

    return OutageResult(
        t_start=snap.t,
        duration=float((idx - snap.index) * dt),
        distance=dist,
        final_error_3d=float(np.linalg.norm(err)),
        final_error_horizontal=float(np.linalg.norm(err[:2])),
        final_error_vertical=float(abs(err[2])),
        max_error_horizontal=max_h,
        heading_error_deg=float(np.rad2deg(_wrap_pi(yaw_err))),
    )


def run_outage_study(
    traj: Trajectory,
    imus: list[ImuSample],
    fixes: list[GnssFix],
    aided_cfg: EskfConfig,
    outage_cfgs: dict[str, EskfConfig],
    durations: list[float],
    window_spacing: float = 60.0,
    warmup: float = 120.0,
    vo_sources: dict[str, dict[int, VoMeasurement]] | None = None,
) -> tuple[PassResult, dict[str, dict[float, list[OutageResult]]]]:
    """Sweep every ablation over every outage duration.

    All ablations share a single GNSS-aided pass, so each variant resumes from
    exactly the same converged state and covariance. The only thing that varies
    is which aiding sources survive the outage, which is the whole question.
    """
    horizon = max(durations)
    t_end = float(traj.t[-1]) - horizon - 1.0
    starts = np.arange(warmup, t_end, window_spacing)
    if len(starts) == 0:
        raise ValueError(
            f"no valid outage windows: trajectory is {traj.t[-1]:.0f} s, but "
            f"warmup ({warmup:.0f} s) + longest outage ({horizon:.0f} s) leaves "
            f"nothing to sample. Need at least "
            f"{warmup + horizon + 1.0:.0f} s. Lengthen the trajectory (--laps), "
            f"shorten the outages (--durations), or reduce --warmup."
        )

    pass_result = run_aided_pass(traj, imus, fixes, aided_cfg, snapshot_times=starts)

    out: dict[str, dict[float, list[OutageResult]]] = {}
    for name, cfg in outage_cfgs.items():
        vo = (vo_sources or {}).get(name)
        out[name] = {
            d: [run_outage(traj, imus, s, cfg, d, vo=vo) for s in pass_result.snapshots]
            for d in durations
        }
    return pass_result, out
